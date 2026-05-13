"""
Data Loaders for EmberVLM Training

Provides data loaders for different training stages.
Memory-safe implementation with distributed training support.
"""

import os
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from PIL import Image, ImageFile
import torchvision.transforms as transforms

# Enable loading of truncated images - prevents crashes from corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Limit PIL's decompression bomb prevention (for very large images)
Image.MAX_IMAGE_PIXELS = 178956970  # ~13K x 13K

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False


# ============================================================================
# MEMORY SAFETY CONFIGURATION
# ============================================================================
# Hard limits to prevent OOM crashes on shared servers
# These values are tuned for 2x A200 GPUs with shared system RAM
# Conservative increases to avoid kernel page allocation failures

# Maximum samples per individual dataset source
MAX_SAMPLES_PER_DATASET = 175000  # 175k per dataset source (was 150k)
MAX_SAMPLES_PER_DATASET_TRIAL = 1000  # 1k for trial mode

# Maximum total samples for Stage 1 alignment (Vision-Language)
# Increased to handle full CC3M (~3.3M) plus other datasets
# CC3M uses lazy loading so memory is manageable
MAX_STAGE1_TOTAL_SAMPLES = 4000000  # 4M total for stage 1 (handles full CC3M + GQA + RefCOCO)
MAX_STAGE1_TOTAL_SAMPLES_TRIAL = 5000  # 5k for trial mode

# Maximum total samples for Stage 2 instruction tuning
MAX_STAGE2_TOTAL_SAMPLES = 250000  # 250k for stage 2 (was 200k)
MAX_STAGE2_TOTAL_SAMPLES_TRIAL = 2000  # 2k for trial mode

# Maximum CC3M samples to load
# CC3M uses LAZY LOADING - only indices stored in RAM, images loaded on-demand
# This makes it safe to use large sample counts (3M+) without OOM
# Full CC3M dataset has ~3.3M samples - use all for better vision-language alignment
MAX_CC3M_SAMPLES = 3300000  # ~3.3M from CC3M (uses lazy loading, memory-safe)

# Maximum GQA samples per JSON file
# GQA teaches scene reasoning - important for robot selection
MAX_GQA_SAMPLES_PER_FILE = 120000  # 120k per GQA file (was 100k)

# Maximum GQA files to process
# Use balanced files + train for best coverage
MAX_GQA_FILES = 5  # Keep at 5 GQA files to be safe

# Safe number of DataLoader workers (prevents CPU oversubscription)
SAFE_NUM_WORKERS = 2  # 2 workers per GPU is usually safe

# Whether to use lazy/streaming dataset (memory efficient but slower)
USE_LAZY_LOADING = True


def _get_rank() -> int:
    """Get current distributed rank (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    """Get world size (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _is_main_process() -> bool:
    """Check if this is rank 0 (main process)."""
    return _get_rank() == 0


def _barrier():
    """Synchronize all distributed processes."""
    if dist.is_initialized():
        dist.barrier()


def _broadcast_object(obj, src=0):
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_initialized() or _get_world_size() == 1:
        return obj

    object_list = [obj] if _get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0.0


def _log_memory_warning(logger, context: str):
    """Log a warning if memory usage is high."""
    mem_mb = _get_memory_usage_mb()
    if mem_mb > 50000:  # > 50GB
        logger.warning(f"⚠️ HIGH MEMORY USAGE: {mem_mb:.0f} MB during {context}")
    elif mem_mb > 30000:  # > 30GB
        logger.info(f"Memory usage: {mem_mb:.0f} MB during {context}")


def _safe_load_image(image_input, transform, image_size: int, logger=None) -> Optional[torch.Tensor]:
    """
    Safely load and transform an image with robust error handling.

    Returns None on failure instead of crashing.
    """
    try:
        # Suppress PIL warnings during loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            if isinstance(image_input, Image.Image):
                # Already a PIL Image
                if image_input.mode != 'RGB':
                    image = image_input.convert('RGB')
                else:
                    image = image_input.copy()  # Copy to avoid modifying original
            elif isinstance(image_input, bytes):
                # Raw bytes
                import io
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            elif isinstance(image_input, (str, Path)):
                # File path
                image = Image.open(str(image_input)).convert('RGB')
            else:
                return None

            # Verify image is valid by loading data
            image.load()

            # Apply transform
            return transform(image)

    except Exception as e:
        if logger:
            logger.debug(f"Failed to load image: {e}")
        return None


class BaseVLMDataset(Dataset):
    """Base dataset class for vision-language data."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        split: str = 'train',
        max_length: int = 512,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        trial_mode: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.trial_mode = trial_mode

        # Image transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # Load data
        self.samples = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data samples. Override in subclasses."""
        raise NotImplementedError

    def _load_image(self, image_input) -> torch.Tensor:
        """Load and transform image from path or PIL Image safely."""
        result = _safe_load_image(image_input, self.transform, self.image_size)
        if result is not None:
            return result
        # Return black image on error
        return torch.zeros(3, self.image_size, self.image_size)

    def _tokenize(
        self,
        text: str,
        add_labels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if add_labels:
            labels = encoding['input_ids'].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class AlignmentDataset(BaseVLMDataset):
    """Dataset for Stage 1 visual-language alignment - supports multiple dataset formats."""

    def __init__(self, *args, **kwargs):
        # Store CC3M dataset reference for lazy loading
        self._cc3m_dataset = None
        self._cc3m_indices = None
        super().__init__(*args, **kwargs)

    def _load_cc3m_hf(self, logger) -> List[Dict[str, Any]]:
        """
        Load CC3M data using HuggingFace datasets library.

        MEMORY SAFE: Only loads metadata, not actual images.
        Images are loaded lazily at __getitem__ time.
        """
        samples = []

        # Only rank 0 should log detailed info
        is_main = _is_main_process()

        # CC3M is stored in HuggingFace WebDataset format
        cc3m_dir = self.data_dir / 'cc3m'

        if not cc3m_dir.exists():
            if is_main:
                logger.debug("CC3M directory not found")
            return samples

        try:
            # Try to import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                if is_main:
                    logger.warning("HuggingFace datasets library not available - skipping CC3M")
                return samples

            # Check if dataset is already cached locally
            cached_dataset_path = cc3m_dir / 'dataset' / 'pixparse___cc3m-wds' / 'default' / '0.0.0'

            dataset = None
            if cached_dataset_path.exists():
                hash_dirs = [d for d in cached_dataset_path.iterdir() if d.is_dir() and len(d.name) == 40]
                if hash_dirs:
                    local_dataset_path = hash_dirs[0]
                    if is_main:
                        logger.info(f"Loading CC3M dataset from cached path: {local_dataset_path}")

                    try:
                        dataset = load_dataset(
                            "arrow",
                            data_files={
                                "train": str(local_dataset_path / "cc3m-wds-train-*.arrow")
                            },
                            split="train"
                        )
                        if is_main:
                            logger.info(f"✓ Successfully loaded {len(dataset):,} samples from cached CC3M dataset")
                    except Exception as e:
                        if is_main:
                            logger.warning(f"Failed to load from cached path: {e}")
                        dataset = None

            if dataset is None:
                if is_main:
                    logger.info("Loading CC3M dataset from HuggingFace...")
                dataset = load_dataset(
                    "pixparse/cc3m-wds",
                    cache_dir=str(cc3m_dir / 'dataset'),
                    split="train"
                )

            if is_main:
                logger.info(f"Found {len(dataset):,} samples in CC3M dataset")

            # MEMORY SAFETY: Use much smaller sample limit
            # TRIAL MODE: Use even smaller limit for quick validation
            if self.trial_mode:
                samples_limit = min(MAX_SAMPLES_PER_DATASET_TRIAL, len(dataset))
                if is_main:
                    logger.info(f"🧪 TRIAL MODE: Limiting CC3M to {samples_limit:,} samples (from {len(dataset):,})")
            else:
                samples_limit = min(MAX_CC3M_SAMPLES, len(dataset))
                if is_main:
                    logger.info(f"⚠️ MEMORY SAFE: Limiting CC3M to {samples_limit:,} samples (from {len(dataset):,})")

            # Sample indices uniformly
            import numpy as np
            if len(dataset) > samples_limit:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(len(dataset), samples_limit, replace=False)
                indices = sorted(indices)  # Sort for sequential access
            else:
                indices = list(range(len(dataset)))

            # Store dataset reference for lazy loading
            self._cc3m_dataset = dataset
            self._cc3m_indices = indices

            # Create lightweight sample references (NO image data stored!)
            # Memory safety: Check memory every 100k samples
            MEMORY_LIMIT_MB = 80000  # 80GB hard limit
            for i, idx in enumerate(indices):
                samples.append({
                    'type': 'cc3m',
                    'cc3m_idx': idx,
                    'caption': None,  # Will be loaded lazily
                })

                # Progress logging every 10k
                if is_main and (i + 1) % 10000 == 0:
                    logger.info(f"  Indexed {i + 1:,} CC3M samples...")

                # Memory safety check every 100k samples
                if (i + 1) % 100000 == 0:
                    mem_mb = _get_memory_usage_mb()
                    if mem_mb > MEMORY_LIMIT_MB:
                        if is_main:
                            logger.warning(f"⚠️ MEMORY LIMIT REACHED: {mem_mb:.0f} MB > {MEMORY_LIMIT_MB} MB")
                            logger.warning(f"   Stopping CC3M indexing at {i + 1:,} samples to prevent OOM")
                        break

            if is_main:
                logger.info(f"✓ Indexed {len(samples):,} CC3M samples (lazy loading enabled)")
                _log_memory_warning(logger, "CC3M indexing")

        except Exception as e:
            if is_main:
                logger.warning(f"Failed to load CC3M dataset: {e}")

        return samples

    def _get_cc3m_sample(self, cc3m_idx: int) -> Dict[str, Any]:
        """Lazily load a CC3M sample by index."""
        if self._cc3m_dataset is None:
            return None

        try:
            item = self._cc3m_dataset[cc3m_idx]
            image = item.get('jpg') or item.get('image')
            caption = item.get('txt') or item.get('caption') or item.get('text', '')

            if image is None or not caption:
                return None

            return {
                'image': image,
                'caption': str(caption).strip(),
            }
        except Exception:
            return None

    def _load_refcoco_arrow(self, logger) -> List[Dict[str, Any]]:
        """Load RefCOCO/RefCOCO+/RefCOCOg data from Arrow files."""
        samples = []

        if not ARROW_AVAILABLE or not PANDAS_AVAILABLE:
            return samples

        # Check for RefCOCO variants - they're in nested directories
        refcoco_dirs = [
            ('refcoco', self.data_dir / 'refcoco'),
            ('refcoco_plus', self.data_dir / 'refcoco_plus'),
            ('refcocog', self.data_dir / 'refcocog'),
        ]

        # Find COCO images directory
        coco_img_dirs = []
        coco_dir = self.data_dir / 'coco'
        if coco_dir.exists():
            for pattern in ['train2014', 'val2014', 'train2017', 'val2017']:
                found = list(coco_dir.glob(pattern))
                coco_img_dirs.extend(found)

        if not coco_img_dirs:
            logger.debug("COCO image directories not found for RefCOCO")
            return samples

        for dataset_name, refcoco_dir in refcoco_dirs:
            if not refcoco_dir.exists():
                continue

            # Arrow files are directly in the refcoco directory (no nested subdirs in your structure)
            arrow_files = sorted(refcoco_dir.glob('*.arrow'))

            if not arrow_files:
                logger.debug(f"No Arrow files found in {dataset_name}")
                continue

            logger.info(f"Loading {dataset_name} from {len(arrow_files)} Arrow files...")

            for arrow_file in arrow_files:
                # Skip dataset_info.json or lock files
                if 'dataset_info' in arrow_file.name or '.lock' in arrow_file.name:
                    continue

                try:
                    table = pa.ipc.open_file(str(arrow_file)).read_all()
                    df = table.to_pandas()

                    for idx in range(len(df)):
                        try:
                            row = df.iloc[idx]

                            # Extract referring expression
                            caption = None
                            for col in ['sent', 'caption', 'sentence', 'text']:
                                if col in df.columns and pd.notna(row[col]):
                                    caption = str(row[col]).strip()
                                    break

                            # Extract image ID
                            image_id = None
                            for col in ['image_id', 'imageId', 'image']:
                                if col in df.columns and pd.notna(row[col]):
                                    image_id = row[col]
                                    break

                            if caption and image_id:
                                # Try to find image - RefCOCO uses COCO 2014 images
                                image_filename = f"COCO_train2014_{int(image_id):012d}.jpg"
                                image_filename_val = f"COCO_val2014_{int(image_id):012d}.jpg"

                                for img_dir in coco_img_dirs:
                                    for fname in [image_filename, image_filename_val]:
                                        candidate = img_dir / fname
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': caption,
                                            })
                                            break
                                    else:
                                        continue
                                    break

                        except Exception as e:
                            logger.debug(f"Failed to process row: {e}")
                            continue

                except Exception as e:
                    logger.debug(f"Failed to load {arrow_file.name}: {e}")
                    continue

            if samples:
                logger.info(f"✓ Loaded {len(samples):,} samples from {dataset_name}")

        return samples

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load image-text pairs from multiple dataset formats.

        MEMORY SAFE: Implements hard limits on samples and efficient loading.
        Only rank 0 does heavy I/O, then broadcasts metadata to other ranks.
        """
        import logging
        logger = logging.getLogger(__name__)

        is_main = _is_main_process()
        samples = []

        # Memory safety: Check initial memory state
        initial_mem_mb = _get_memory_usage_mb()
        if is_main and initial_mem_mb > 50000:  # > 50GB already
            logger.warning(f"⚠️ HIGH INITIAL MEMORY: {initial_mem_mb:.0f} MB before data loading")
            logger.warning("   Consider reducing batch size or clearing other processes")

        # Track samples by source for limiting
        samples_by_source = {}
        gqa_files_processed = 0
        
        # Use trial mode limits if enabled
        max_samples_per_dataset = MAX_SAMPLES_PER_DATASET_TRIAL if self.trial_mode else MAX_SAMPLES_PER_DATASET
        max_stage1_total = MAX_STAGE1_TOTAL_SAMPLES_TRIAL if self.trial_mode else MAX_STAGE1_TOTAL_SAMPLES

        if is_main:
            logger.info("="*60)
            if self.trial_mode:
                logger.info("🧪 TRIAL MODE: Using reduced dataset")
            logger.info("MEMORY-SAFE DATASET LOADING")
            logger.info(f"  Initial memory usage: {initial_mem_mb:.0f} MB")
            logger.info(f"  Max samples per dataset: {max_samples_per_dataset:,}")
            logger.info(f"  Max total Stage 1 samples: {max_stage1_total:,}")
            logger.info(f"  Max CC3M samples: {MAX_CC3M_SAMPLES:,}")
            logger.info(f"  Max GQA files: {MAX_GQA_FILES}")
            logger.info("="*60)

        # Load CC3M from HuggingFace format (memory safe - lazy loading)
        cc3m_samples = self._load_cc3m_hf(logger)
        samples.extend(cc3m_samples)
        samples_by_source['cc3m'] = len(cc3m_samples)

        # Load RefCOCO variants from Arrow files (limited)
        refcoco_samples = self._load_refcoco_arrow(logger)
        # Limit RefCOCO samples
        if len(refcoco_samples) > max_samples_per_dataset:
            refcoco_samples = refcoco_samples[:max_samples_per_dataset]
            if is_main:
                logger.info(f"⚠️ Limited RefCOCO to {max_samples_per_dataset:,} samples")
        samples.extend(refcoco_samples)
        samples_by_source['refcoco'] = len(refcoco_samples)

        # Recursively search for JSON files in subdirectories
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files and is_main:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")

        if is_main:
            logger.info(f"Found {len(json_files)} JSON files to process")

        # Skip metadata files and non-VL datasets
        skip_patterns = [
            'download_summary', 'dataset_info', 'instances_', 'person_keypoints_',
            '__MACOSX', '.lock', '_builder.lock', '_incomplete', 'dataset.json',
            'readme.txt', 'LICENCE.txt', 'loadDataset.py', '.py', '.csv', '.download_attempted',
        ]

        # Also skip annotation-only files that don't contain usable data
        skip_exact = [
            'v2_mscoco_train2014_annotations.json',
            'v2_mscoco_val2014_annotations.json',
            'mscoco_train2014_annotations.json',
            'mscoco_val2014_annotations.json',
        ]

        # Prioritize certain files (balanced/smaller datasets first)
        priority_patterns = ['balanced', 'val', 'train']

        def get_priority(f):
            name = f.name.lower()
            for i, p in enumerate(priority_patterns):
                if p in name:
                    return i
            return len(priority_patterns)

        json_files = sorted(json_files, key=get_priority)

        for json_file in json_files:
            # Check if we've hit the total sample limit
            if len(samples) >= max_stage1_total:
                if is_main:
                    logger.info(f"⚠️ Reached max total samples ({max_stage1_total:,}), stopping dataset loading")
                break

            # Skip metadata files
            file_name_lower = json_file.name.lower()
            if any(pattern in file_name_lower for pattern in skip_patterns):
                continue

            # Skip exact matches of problematic files
            if json_file.name in skip_exact:
                if is_main:
                    logger.debug(f"Skipping annotation-only file: {json_file.name}")
                continue

            # Skip files in __MACOSX directories
            if '__MACOSX' in str(json_file):
                continue

            # Check if this is a GQA file and if we've processed enough
            is_gqa = 'gqa' in str(json_file).lower()
            if is_gqa:
                if gqa_files_processed >= MAX_GQA_FILES:
                    if is_main:
                        logger.debug(f"Skipping GQA file (already processed {MAX_GQA_FILES} files): {json_file.name}")
                    continue

            if is_main:
                logger.info(f"Processing file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                if is_main:
                    logger.warning(f"Failed to load {json_file}: {e}")
                continue

            # Get the parent directory for resolving image paths
            json_parent = json_file.parent
            before_count = len(samples)
            file_sample_count = 0
            max_samples_this_file = max_samples_per_dataset

            if isinstance(data, list):
                # Handle list format (LLaVA, CC3M, etc.)
                for item in data:
                    # Check per-file limit
                    if file_sample_count >= max_samples_this_file:
                        break

                    text = None
                    image_path = None

                    # Extract text (caption, question+answer, instruction, conversations, etc.)
                    if 'conversations' in item:
                        # LLaVA format: conversations = [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
                        convos = item['conversations']
                        if len(convos) >= 2:
                            human_text = convos[0].get('value', '').replace('<image>\n', '').replace('<image>', '').strip()
                            gpt_text = convos[1].get('value', '').strip()
                            if human_text and gpt_text:
                                text = f"Question: {human_text} Answer: {gpt_text}"
                    elif 'caption' in item:
                        text = item['caption']
                    elif 'text' in item:
                        text = item['text']
                    elif 'question' in item and 'answer' in item:
                        text = f"Question: {item['question']} Answer: {item['answer']}"
                    elif 'question' in item:
                        text = item['question']

                    # Extract image path
                    if 'image' in item:
                        image_path = item['image']
                        # LLaVA often uses COCO image IDs like "coco/train2017/000000123456.jpg"
                        if not os.path.isabs(image_path):
                            # Try multiple potential base dirs
                            potential_paths = [
                                json_parent / image_path,
                                self.data_dir / image_path,
                                self.data_dir / 'coco' / image_path.split('/')[-1] if '/' in image_path else None,
                            ]
                            for p in potential_paths:
                                if p and p.exists():
                                    image_path = str(p)
                                    break
                            else:
                                image_path = str(json_parent / image_path)  # Fallback
                    elif 'image_id' in item:
                        image_path = f"{item['image_id']}.jpg"
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)
                    elif 'file_name' in item:
                        image_path = item['file_name']
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)

                    if text and image_path:
                        samples.append({
                            'image': image_path,
                            'caption': text,
                        })
                        file_sample_count += 1

            elif isinstance(data, dict):
                # ===== COCO Captions Format =====
                if 'annotations' in data and 'images' in data and any('caption' in ann for ann in data['annotations'][:5]):
                    logger.info(f"Detected COCO Captions format: {len(data['images'])} images, {len(data['annotations'])} annotations")

                    image_map = {img['id']: img['file_name'] for img in data['images']}

                    # Find image directories
                    image_dirs = []
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['train2017', 'val2017', 'train2014', 'val2014', 'images']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    if not image_dirs:
                        if is_main:
                            logger.warning(f"No image directories found for {json_file}")
                    else:
                        if is_main:
                            logger.info(f"Found image directories: {[str(d) for d in image_dirs]}")

                        for ann in data['annotations']:
                            # Check per-file limit
                            if file_sample_count >= max_samples_this_file:
                                break

                            image_id = ann.get('image_id')
                            caption = ann.get('caption', '')

                            if caption and image_id in image_map:
                                image_filename = image_map[image_id]
                                image_path = None
                                for img_dir in image_dirs:
                                    candidate = img_dir / image_filename
                                    if candidate.exists():
                                        image_path = str(candidate)
                                        break

                                if image_path:
                                    samples.append({
                                        'image': image_path,
                                        'caption': caption,
                                    })
                                    file_sample_count += 1

                # ===== VQA Format (questions + annotations) =====
                elif 'questions' in data or ('annotations' in data and any('question' in ann or 'answer' in ann for ann in data.get('annotations', [])[:5])):
                    if is_main:
                        logger.info(f"Detected VQA format")

                    # VQA v2 format has separate questions and annotations
                    questions = data.get('questions', data.get('annotations', []))

                    # Find image directory
                    image_dirs = []
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['train2014', 'val2014', 'test2015', 'train2017', 'val2017', 'images']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    if not image_dirs:
                        # Try COCO directory structure
                        coco_dir = self.data_dir / 'coco'
                        if coco_dir.exists():
                            for pattern in ['train2014', 'val2014', 'train2017', 'val2017']:
                                found_dirs = list(coco_dir.glob(pattern))
                                image_dirs.extend(found_dirs)

                    for item in questions:
                        # Check per-file limit
                        if file_sample_count >= max_samples_this_file:
                            break

                        question = item.get('question', '')
                        answer = item.get('answer', item.get('multiple_choice_answer', ''))
                        image_id = item.get('image_id', '')

                        if question and image_id:
                            # Construct text as Q&A pair or just question
                            if answer:
                                text = f"Question: {question} Answer: {answer}"
                            else:
                                text = f"Question: {question}"

                            # Try multiple COCO image naming conventions
                            possible_filenames = [
                                f"COCO_train2014_{image_id:012d}.jpg",
                                f"COCO_val2014_{image_id:012d}.jpg",
                                f"COCO_train2017_{image_id:012d}.jpg",
                                f"COCO_val2017_{image_id:012d}.jpg",
                                f"{image_id:012d}.jpg",  # Sometimes just the ID
                            ]
                            if 'image_name' in item:
                                possible_filenames.insert(0, item['image_name'])

                            found_image = False
                            if image_dirs:
                                for img_dir in image_dirs:
                                    if found_image:
                                        break
                                    for image_filename in possible_filenames:
                                        candidate = img_dir / image_filename
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': text,
                                            })
                                            file_sample_count += 1
                                            found_image = True
                                            break

                # ===== GQA Format =====
                elif isinstance(data, dict) and len(data) > 0:
                    # Check if it's GQA format: dict of dicts with question/answer structure
                    first_values = list(data.values())[:5]
                    is_gqa_format = all(isinstance(v, dict) and any(k in v for k in ['question', 'answer', 'imageId']) for v in first_values if isinstance(v, dict))

                    if is_gqa_format:
                        # GQA is dict of dicts: {question_id: {question, answer, imageId, ...}}
                        if is_main:
                            logger.info(f"Detected GQA format: {len(data)} questions")

                        # Find GQA images directory
                        image_dirs = []
                        for search_dir in [json_parent, json_parent.parent]:
                            for pattern in ['images', 'allImages']:
                                found_dirs = list(search_dir.glob(pattern))
                                image_dirs.extend(found_dirs)

                        # MEMORY SAFETY: Use much smaller limit per GQA file
                        gqa_limit = min(MAX_GQA_SAMPLES_PER_FILE, max_samples_this_file - file_sample_count, len(data))

                        if is_main:
                            logger.info(f"⚠️ GQA limit for this file: {gqa_limit:,} (from {len(data):,})")

                        for qid, item in list(data.items())[:gqa_limit]:
                            if file_sample_count >= max_samples_this_file:
                                break

                            if not isinstance(item, dict):
                                continue

                            question = item.get('question', '')
                            answer = item.get('answer', '')
                            image_id = item.get('imageId', '')

                            if question and image_id:
                                text = f"Question: {question} Answer: {answer}" if answer else f"Question: {question}"
                                image_filename = f"{image_id}.jpg"

                                if image_dirs:
                                    for img_dir in image_dirs:
                                        candidate = img_dir / image_filename
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': text,
                                            })
                                            file_sample_count += 1
                                            break

                        # Mark that we've processed a GQA file
                        if is_gqa:
                            gqa_files_processed += 1
                            if is_main:
                                logger.info(f"GQA files processed: {gqa_files_processed}/{MAX_GQA_FILES}")

                # ===== RefCOCO Format =====
                elif 'refs' in data or any('ref_id' in str(k) for k in list(data.keys())[:5]):
                    if is_main:
                        logger.info(f"Detected RefCOCO format")

                    refs = data.get('refs', data.get('annotations', []))

                    # Find image directory (RefCOCO uses COCO images)
                    image_dirs = []
                    coco_dir = self.data_dir / 'coco'
                    if coco_dir.exists():
                        for pattern in ['train2014', 'train2017', 'val2014', 'val2017']:
                            found_dirs = list(coco_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    for ref in refs:
                        if file_sample_count >= max_samples_this_file:
                            break

                        # RefCOCO has referring expressions
                        sentences = ref.get('sentences', [])
                        image_id = ref.get('image_id', '')

                        for sent in sentences:
                            if file_sample_count >= max_samples_this_file:
                                break
                            text = sent.get('sent', sent.get('raw', ''))
                            if text and image_id:
                                image_filename = f"COCO_train2014_{image_id:012d}.jpg"

                                if image_dirs:
                                    for img_dir in image_dirs:
                                        candidate = img_dir / image_filename
                                        if candidate.exists():
                                            samples.append({
                                                'image': str(candidate),
                                                'caption': text,
                                            })
                                            file_sample_count += 1
                                            break

                # ===== OCR-VQA Format =====
                # Check if it's actually OCR-VQA data (has questions, not just metadata)
                elif 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                    # Verify first item has OCR-VQA structure
                    first_item = data['data'][0] if isinstance(data['data'], list) else {}
                    if 'question' in first_item or 'imageURL' in first_item:
                        if is_main:
                            logger.info(f"Detected OCR-VQA format")

                        items = data['data']
                        ocr_limit = min(max_samples_this_file, len(items))

                        for item in items[:ocr_limit]:
                            if file_sample_count >= max_samples_this_file:
                                break

                            if not isinstance(item, dict):
                                continue

                            question = item.get('question', item.get('text', ''))
                            answer = item.get('answer', '')
                            if 'answers' in item:
                                if isinstance(item['answers'], list) and len(item['answers']) > 0:
                                    answer = item['answers'][0]
                                elif isinstance(item['answers'], dict) and 'answer' in item['answers']:
                                    answer = item['answers']['answer']

                            image_id = item.get('imageURL', item.get('image_id', item.get('image', '')))

                            if question and image_id:
                                text = f"Question: {question} Answer: {answer}" if answer else f"Question: {question}"

                                # Try to find image
                                if isinstance(image_id, str) and ('/' in image_id or '.' in image_id):
                                    image_path = image_id
                                else:
                                    image_path = f"{image_id}.jpg"

                                if not os.path.isabs(image_path):
                                    image_path = str(json_parent / image_path)

                                samples.append({
                                    'image': image_path,
                                    'caption': text,
                                })
                                file_sample_count += 1

            # Log how many samples were added from this file
            after_count = len(samples)
            added_count = after_count - before_count

            if added_count > 0:
                if is_main:
                    logger.info(f"✓ Loaded {added_count:,} samples from {json_file.name}")
            else:
                if is_main:
                    logger.debug(f"No samples loaded from {json_file.name}")

            # Free memory after processing each file
            del data
            gc.collect()

        # Filter by split if needed
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        if is_main:
            logger.info("="*80)
            logger.info(f"FINAL DATASET: Loaded {len(samples):,} total samples for {self.split} split")
            logger.info(f"Memory usage: {_get_memory_usage_mb():.0f} MB")
            logger.info("="*80)

        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Handle CC3M lazy loading
        if sample.get('type') == 'cc3m':
            cc3m_data = self._get_cc3m_sample(sample['cc3m_idx'])
            if cc3m_data:
                pixel_values = self._load_image(cc3m_data['image'])
                caption = cc3m_data['caption']
            else:
                # Fallback to black image
                pixel_values = torch.zeros(3, self.image_size, self.image_size)
                caption = ""
        else:
            # Load image from path
            pixel_values = self._load_image(sample['image'])
            caption = sample.get('caption', '')

        # Tokenize caption
        text_data = self._tokenize(caption)

        return {
            'pixel_values': pixel_values,
            **text_data,
        }


class InstructionDataset(BaseVLMDataset):
    """Dataset for Stage 2 instruction tuning with proper VQA support."""

    # Diverse instruction prompts for captioning (to avoid overfitting to single prompt)
    CAPTION_PROMPTS = [
        "Describe this image.",
        "What do you see in this image?",
        "Provide a description of this image.",
        "What is shown in this picture?",
        "Describe what you observe in the image.",
        "Give a brief description of this image.",
        "What's happening in this image?",
        "Explain what this image contains.",
    ]

    # Diverse prompts for VQA (wrapping questions)
    VQA_PROMPTS = [
        "{question}",  # Direct question
        "Answer the following question about the image: {question}",
        "Based on the image, {question}",
        "Looking at the image, {question}",
        "Question: {question}",
    ]

    def _parse_qa_from_caption(self, caption: str) -> tuple:
        """
        Parse question-answer pairs from caption strings.

        Handles formats like:
        - "Question: What color is the car? Answer: Red"
        - "Q: What color is the car? A: Red"

        Returns:
            (question, answer) tuple, or (None, None) if not a Q&A format
        """
        import re

        # Pattern 1: "Question: X Answer: Y"
        match = re.match(r'^Question:\s*(.+?)\s*Answer:\s*(.+)$', caption, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Pattern 2: "Q: X A: Y"
        match = re.match(r'^Q:\s*(.+?)\s*A:\s*(.+)$', caption, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Pattern 3: Check if it looks like a question (starts with question word)
        # but doesn't have an answer - skip these
        question_starters = ['what', 'where', 'when', 'who', 'why', 'how', 'is', 'are', 'does', 'do', 'can', 'could', 'would', 'will']
        lower_caption = caption.lower().strip()
        if any(lower_caption.startswith(q) for q in question_starters) and '?' in caption:
            # This looks like a question without answer - not usable for training
            return None, None

        return None, None

    def _looks_like_code_text(self, text: Optional[str]) -> bool:
        """Detect obvious code-like responses that should be excluded from VLM instruction tuning."""
        if not text:
            return False

        lowered = text.lower()
        code_markers = [
            'def ',
            'import ',
            'class ',
            'return ',
            'self.',
            'cv2.',
            'torch.',
            'numpy',
            'from ',
        ]
        hits = sum(1 for marker in code_markers if marker in lowered)
        return hits >= 2

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load instruction data with memory safety limits and proper VQA parsing."""
        import logging
        logger = logging.getLogger(__name__)

        is_main = _is_main_process()
        samples = []
        vqa_count = 0
        caption_count = 0

        # Use trial mode limits if enabled
        max_stage2_total = MAX_STAGE2_TOTAL_SAMPLES_TRIAL if self.trial_mode else MAX_STAGE2_TOTAL_SAMPLES
        max_samples_per_dataset = MAX_SAMPLES_PER_DATASET_TRIAL if self.trial_mode else MAX_SAMPLES_PER_DATASET

        # Recursively search for JSON files
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            if is_main:
                logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        if is_main:
            logger.info(f"Found {len(json_files)} JSON files to process for instruction tuning")
            logger.info(f"⚠️ MEMORY SAFE: Max samples = {max_stage2_total:,}")

        for json_file in json_files:
            # Check total sample limit
            if len(samples) >= max_stage2_total:
                if is_main:
                    logger.info(f"⚠️ Reached max Stage 2 samples ({max_stage2_total:,})")
                break

            if is_main:
                logger.info(f"Processing instruction file: {json_file}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                if is_main:
                    logger.warning(f"Failed to load {json_file}: {e}")
                continue

            json_parent = json_file.parent
            file_samples = 0
            file_vqa = 0
            file_caption = 0
            max_per_file = min(max_samples_per_dataset, max_stage2_total - len(samples))

            # Handle list format (most common)
            if isinstance(data, list):
                for item in data:
                    if file_samples >= max_per_file:
                        break

                    if 'image' in item:
                        image_path = item['image']
                        # Resolve relative image paths
                        if not os.path.isabs(image_path):
                            image_path = str(json_parent / image_path)

                        # Priority 1: LLaVA conversation format
                        if 'conversations' in item:
                            instruction = ""
                            response = ""
                            for conv in item['conversations']:
                                if conv.get('from') == 'human':
                                    instruction = conv.get('value', '')
                                elif conv.get('from') == 'gpt':
                                    response = conv.get('value', '')
                            if instruction and response:
                                samples.append({
                                    'image': image_path,
                                    'instruction': instruction,
                                    'response': response,
                                    'type': 'conversation',
                                })
                                file_samples += 1
                                file_vqa += 1

                        # Priority 2: Explicit instruction/question format
                        elif 'instruction' in item or 'question' in item:
                            instruction = item.get('instruction', item.get('question', ''))
                            response = item.get('response', item.get('answer', ''))
                            if instruction and response:
                                samples.append({
                                    'image': image_path,
                                    'instruction': instruction,
                                    'response': response,
                                    'type': 'vqa',
                                })
                                file_samples += 1
                                file_vqa += 1

                        # Priority 3: Caption with embedded Q&A (parse "Question: X Answer: Y")
                        elif 'caption' in item or 'text' in item:
                            caption = item.get('caption', item.get('text', ''))
                            if caption:
                                # Try to parse as Q&A first
                                question, answer = self._parse_qa_from_caption(caption)
                                if question and answer:
                                    # This is VQA data - use question as instruction
                                    samples.append({
                                        'image': image_path,
                                        'instruction': question,
                                        'response': answer,
                                        'type': 'vqa_parsed',
                                    })
                                    file_samples += 1
                                    file_vqa += 1
                                else:
                                    # This is pure caption data - use varied prompts
                                    samples.append({
                                        'image': image_path,
                                        'instruction': None,  # Will be filled with random prompt
                                        'response': caption,
                                        'type': 'caption',
                                    })
                                    file_samples += 1
                                    file_caption += 1

            # Handle dict format (GQA, VQA v2, etc.)
            elif isinstance(data, dict):
                # Check for GQA format: {question_id: {question, answer, imageId}}
                first_values = list(data.values())[:5]
                is_gqa_format = all(
                    isinstance(v, dict) and any(k in v for k in ['question', 'answer', 'imageId'])
                    for v in first_values if isinstance(v, dict)
                )

                if is_gqa_format:
                    if is_main:
                        logger.info(f"Detected GQA format: {len(data)} questions")

                    # Find GQA images directory
                    image_dirs = []
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['images', 'allImages']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    gqa_limit = min(MAX_GQA_SAMPLES_PER_FILE, max_per_file - file_samples, len(data))

                    for qid, item in list(data.items())[:gqa_limit]:
                        if file_samples >= max_per_file:
                            break

                        if not isinstance(item, dict):
                            continue

                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        image_id = item.get('imageId', '')

                        if question and answer and image_id:
                            image_filename = f"{image_id}.jpg"
                            image_path = None

                            for img_dir in image_dirs:
                                candidate = img_dir / image_filename
                                if candidate.exists():
                                    image_path = str(candidate)
                                    break

                            if image_path:
                                samples.append({
                                    'image': image_path,
                                    'instruction': question,
                                    'response': answer,
                                    'type': 'gqa',
                                })
                                file_samples += 1
                                file_vqa += 1

                # Check for VQA v2 format
                elif 'questions' in data or 'annotations' in data:
                    questions_list = data.get('questions', data.get('annotations', []))
                    if is_main:
                        logger.info(f"Detected VQA format: {len(questions_list)} items")

                    # Find image directories
                    image_dirs = []
                    for search_dir in [json_parent, json_parent.parent]:
                        for pattern in ['train2014', 'val2014', 'train2017', 'val2017', 'images']:
                            found_dirs = list(search_dir.glob(pattern))
                            image_dirs.extend(found_dirs)

                    for item in questions_list:
                        if file_samples >= max_per_file:
                            break

                        question = item.get('question', '')
                        answer = item.get('answer', item.get('multiple_choice_answer', ''))
                        image_id = item.get('image_id', '')

                        if question and answer and image_id:
                            # Try to find image
                            possible_filenames = [
                                f"COCO_train2014_{image_id:012d}.jpg",
                                f"COCO_val2014_{image_id:012d}.jpg",
                                f"{image_id:012d}.jpg",
                                f"{image_id}.jpg",
                            ]

                            image_path = None
                            for img_dir in image_dirs:
                                for fname in possible_filenames:
                                    candidate = img_dir / fname
                                    if candidate.exists():
                                        image_path = str(candidate)
                                        break
                                if image_path:
                                    break

                            if image_path:
                                samples.append({
                                    'image': image_path,
                                    'instruction': question,
                                    'response': answer,
                                    'type': 'vqa',
                                })
                                file_samples += 1
                                file_vqa += 1

            if is_main:
                logger.info(f"✓ Loaded {file_samples} samples from {json_file.name} (VQA: {file_vqa}, Caption: {file_caption})")

            vqa_count += file_vqa
            caption_count += file_caption

        # Split train/val
        pre_filter_count = len(samples)
        samples = [
            sample for sample in samples
            if not self._looks_like_code_text(sample.get('response'))
        ]
        removed_code_like = pre_filter_count - len(samples)

        if is_main and removed_code_like > 0:
            logger.info(f"Filtered {removed_code_like:,} code-like instruction responses")

        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        if is_main:
            logger.info("=" * 60)
            logger.info(f"INSTRUCTION DATASET SUMMARY ({self.split} split):")
            logger.info(f"  Total samples: {len(samples):,}")
            logger.info(f"  VQA samples: {vqa_count:,}")
            logger.info(f"  Caption samples: {caption_count:,}")
            logger.info(f"  VQA ratio: {vqa_count / max(1, vqa_count + caption_count) * 100:.1f}%")
            logger.info("=" * 60)

        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        pixel_values = self._load_image(sample['image'])

        instruction = sample['instruction']
        response = sample['response']
        sample_type = sample.get('type', 'unknown')

        # For caption samples, use varied prompts to avoid overfitting
        if instruction is None or sample_type == 'caption':
            # Deterministic but varied prompt selection based on idx
            prompt_idx = idx % len(self.CAPTION_PROMPTS)
            instruction = self.CAPTION_PROMPTS[prompt_idx]

        # For VQA samples, optionally wrap with varied prompt templates
        elif sample_type in ['vqa', 'vqa_parsed', 'gqa']:
            # Use direct question most of the time (70%), varied prompts otherwise
            if idx % 10 < 7:
                # Direct question (most common for VQA)
                pass  # Keep instruction as-is
            else:
                # Wrap with varied prompt template
                prompt_idx = (idx // 10) % (len(self.VQA_PROMPTS) - 1) + 1  # Skip first (direct) template
                instruction = self.VQA_PROMPTS[prompt_idx].format(question=instruction)

        # Format as instruction-response for training
        # Use a clear format that separates instruction from response
        text = f"User: {instruction}\nAssistant: {response}"
        text_data = self._tokenize(text)

        return {
            'pixel_values': pixel_values,
            'raw_text': text,
            'instruction': instruction,
            'response': response,
            **text_data,
        }


class ReasoningDataset(BaseVLMDataset):
    """
    Dataset for Stage 4 reasoning training (DeepSeek-R1 style).

    Supports:
    1. Legacy format (reasoning_chain list)
    2. XML format (<reasoning>...</reasoning><answer>...</answer>)
    3. Robot selection data (auto-generates reasoning chains from task + subtasks)
    """

    # Robot capabilities for auto-generating reasoning
    ROBOT_KNOWLEDGE = {
        'Drone': {
            'capabilities': ['aerial navigation', 'surveillance', 'fast movement', 'hard to reach areas', 'aerial inspection'],
            'limitations': ['limited payload', 'weather dependent', 'battery life'],
            'environments': ['outdoor', 'large indoor spaces', 'tall structures'],
        },
        'Humanoid': {
            'capabilities': ['manipulation', 'human interaction', 'complex tasks', 'tool use', 'walking'],
            'limitations': ['slow movement', 'balance issues', 'high power consumption'],
            'environments': ['indoor', 'human environments', 'stairs'],
        },
        'Robot with Wheels': {
            'capabilities': ['fast movement', 'good payload', 'stable platform', 'efficient'],
            'limitations': ['flat surfaces only', 'limited climbing'],
            'environments': ['indoor', 'warehouse', 'flat outdoor areas', 'roads'],
        },
        'Robot with Legs': {
            'capabilities': ['rough terrain navigation', 'stability', 'load carrying', 'stairs'],
            'limitations': ['limited manipulation', 'height restrictions'],
            'environments': ['outdoor', 'uneven terrain', 'industrial sites', 'search and rescue'],
        },
        'Underwater Robot': {
            'capabilities': ['underwater navigation', 'deep sea exploration', 'marine inspection'],
            'limitations': ['water environments only', 'pressure constraints'],
            'environments': ['underwater', 'marine', 'pools', 'pipes', 'ocean'],
        },
    }

    def _generate_reasoning_chain(self, task: str, robot: str, subtasks: List[Dict] = None) -> List[str]:
        """Auto-generate reasoning chain from task and selected robot."""
        chain = []

        # Extract task name
        task_clean = task.replace('Task:', '').strip()
        chain.append(f"Analyzing task: {task_clean}")

        # Get robot knowledge
        robot_info = self.ROBOT_KNOWLEDGE.get(robot, {})
        capabilities = robot_info.get('capabilities', [])
        environments = robot_info.get('environments', [])

        # Generate reasoning based on task and robot match
        if capabilities:
            relevant_caps = capabilities[:3]  # Top 3 capabilities
            chain.append(f"{robot} has relevant capabilities: {', '.join(relevant_caps)}")

        if environments:
            chain.append(f"{robot} is suitable for environments like: {', '.join(environments[:2])}")

        # If we have subtasks, use them for more detailed reasoning
        if subtasks:
            chain.append(f"Breaking down into {len(subtasks)} subtasks for optimal execution")
            for st in subtasks[:3]:  # Max 3 subtasks in reasoning
                subtask_desc = st.get('subtask', '')
                assigned = st.get('assigned_robot', robot)
                chain.append(f"Subtask '{subtask_desc[:50]}...' assigned to {assigned}")

        chain.append(f"Conclusion: {robot} is the best choice for this task")

        return chain

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load reasoning data with chains. Auto-generates chains for robot selection data."""
        import logging
        logger = logging.getLogger(__name__)

        samples = []

        # Recursively search for JSON files
        json_files = list(self.data_dir.rglob('*.json'))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir} or its subdirectories")
            return samples

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

            json_parent = json_file.parent

            if isinstance(data, list):
                for item in data:
                    # Detect data type and handle accordingly

                    # Case 1: Robot selection data (single robot)
                    if 'output' in item and 'input' in item and 'Task:' in item.get('input', ''):
                        # This is single robot selection format
                        task = item.get('input', '')
                        selected_robot = item.get('output', '')

                        # Handle multi-robot output (e.g., "Drone, Robot with Legs")
                        if ',' in selected_robot:
                            primary_robot = selected_robot.split(',')[0].strip()
                        else:
                            primary_robot = selected_robot.strip()

                        # Auto-generate reasoning chain
                        reasoning_chain = self._generate_reasoning_chain(task, primary_robot)

                        sample = {
                            'instruction': item.get('instruction', ''),
                            'reasoning_chain': reasoning_chain,
                            'response': selected_robot,
                            'robot_target': primary_robot,
                            'task': task,
                            'format': 'auto_generated',
                        }
                        samples.append(sample)

                    # Case 2: Multi-robot selection data (with subtasks)
                    elif 'subtasks' in item and 'input' in item:
                        task = item.get('input', '')
                        subtasks = item.get('subtasks', [])
                        original_output = item.get('original_single_robot_output', '')

                        # Handle original_output being either a string or list
                        if isinstance(original_output, list):
                            # If it's a list, join and take first robot
                            original_output_str = ', '.join(str(r) for r in original_output) if original_output else ''
                        else:
                            original_output_str = str(original_output) if original_output else ''

                        # Get primary robot from subtasks or original output
                        if subtasks:
                            first_robot_fallback = original_output_str.split(',')[0].strip() if original_output_str else 'Drone'
                            primary_robot = subtasks[0].get('assigned_robot', first_robot_fallback)
                        else:
                            primary_robot = original_output_str.split(',')[0].strip() if original_output_str else 'Drone'

                        # Auto-generate reasoning chain using subtasks
                        reasoning_chain = self._generate_reasoning_chain(task, primary_robot, subtasks)

                        # Build response with all assigned robots
                        assigned_robots = list(set([st.get('assigned_robot', '') for st in subtasks if st.get('assigned_robot')]))
                        response = ', '.join(assigned_robots) if assigned_robots else original_output_str

                        sample = {
                            'instruction': item.get('instruction', ''),
                            'reasoning_chain': reasoning_chain,
                            'response': response,
                            'robot_target': primary_robot,
                            'task': task,
                            'subtasks': subtasks,
                            'format': 'auto_generated',
                        }
                        samples.append(sample)

                    # Case 3: Pre-formatted reasoning data (original format)
                    else:
                        sample = {
                            'instruction': item.get('instruction', ''),
                            'reasoning_chain': item.get('reasoning_chain', []),
                            'response': item.get('response', ''),
                            'robot_target': item.get('selected_robot', item.get('robot_target', item.get('answer'))),
                            'task': item.get('task', ''),
                            'format': item.get('format', 'legacy'),
                        }

                        # Handle image if present
                        if 'image' in item:
                            image_path = item['image']
                            if not os.path.isabs(image_path):
                                image_path = str(json_parent / image_path)
                            sample['image'] = image_path

                        # Handle chat-style prompt (DeepSeek-R1 style)
                        if 'prompt' in item:
                            sample['prompt'] = item['prompt']

                        samples.append(sample)

        # Split
        if self.split == 'train':
            samples = samples[:int(len(samples) * 0.9)]
        else:
            samples = samples[int(len(samples) * 0.9):]

        logger.info(f"Loaded {len(samples)} samples for {self.split} split from {self.data_dir}")
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image if present
        if 'image' in sample and sample['image']:
            pixel_values = self._load_image(sample['image'])
        else:
            # Create placeholder image for text-only samples
            pixel_values = torch.zeros(3, self.image_size, self.image_size)

        # Check if this is XML format (DeepSeek-R1 style)
        is_xml_format = sample.get('format') == 'xml' or '<reasoning>' in sample.get('response', '')

        if is_xml_format:
            # XML format: response already contains <reasoning>...</reasoning><answer>...</answer>
            text = f"{sample['instruction']}\n\n{sample['response']}"
        else:
            # Legacy format: Build text from reasoning chain list
            reasoning_text = ""
            if sample['reasoning_chain']:
                reasoning_steps = "\n".join([
                    f"Step {i+1}: {step}"
                    for i, step in enumerate(sample['reasoning_chain'])
                ])
                reasoning_text = f"<reasoning>\n{reasoning_steps}\n</reasoning>\n<answer>\n{sample.get('robot_target', '')}\n</answer>"
                text = f"{sample['instruction']}\n\n{reasoning_text}"
            else:
                text = f"{sample['instruction']}\n{sample['response']}"

        text_data = self._tokenize(text)

        result = {
            'pixel_values': pixel_values,
            **text_data,
        }

        # Add robot target if available
        robot_mapping = {
            'Drone': 0, 'drone': 0,
            'Humanoid': 1, 'humanoid': 1,
            'Wheeled': 2, 'Robot with Wheels': 2, 'robot with wheels': 2, 'wheeled robot': 2,
            'Legged': 3, 'Robot with Legs': 3, 'robot with legs': 3, 'legged robot': 3,
            'Underwater': 4, 'Underwater Robot': 4, 'underwater robot': 4,
        }

        robot_target = sample.get('robot_target')
        if robot_target is not None:
            if isinstance(robot_target, str):
                # Try to normalize the robot name
                robot_target_lower = robot_target.lower().strip()
                robot_target = robot_mapping.get(robot_target, robot_mapping.get(robot_target_lower, 0))
            result['robot_target'] = torch.tensor(robot_target, dtype=torch.long)
            result['robot_target_names'] = sample.get('robot_target', '')  # Keep string name for rewards

        return result


def get_alignment_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = None,  # Will use SAFE_NUM_WORKERS if not specified
    trial_mode: bool = False,  # If True, use much smaller dataset
    **kwargs,
) -> DataLoader:
    """
    Create dataloader for alignment stage.

    MEMORY SAFE: Uses limited num_workers and persistent_workers to prevent
    memory duplication and CPU oversubscription.
    
    Args:
        trial_mode: If True, limits data to 5k samples for quick validation
    """
    import logging
    logger = logging.getLogger(__name__)

    # Use safe default for num_workers
    if num_workers is None:
        num_workers = SAFE_NUM_WORKERS
    else:
        # Cap at safe maximum
        num_workers = min(num_workers, SAFE_NUM_WORKERS)

    if _is_main_process():
        logger.info(f"Creating alignment dataloader with {num_workers} workers")

    dataset = AlignmentDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        trial_mode=trial_mode,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),  # Keep workers alive between batches
        prefetch_factor=2 if num_workers > 0 else None,  # Limit prefetching
    )


def get_instruction_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = None,
    trial_mode: bool = False,  # If True, use much smaller dataset
    **kwargs,
) -> DataLoader:
    """
    Create dataloader for instruction tuning.

    MEMORY SAFE: Uses limited num_workers.
    
    Args:
        trial_mode: If True, limits data to 2k samples for quick validation
    """
    import logging
    logger = logging.getLogger(__name__)

    if num_workers is None:
        num_workers = SAFE_NUM_WORKERS
    else:
        num_workers = min(num_workers, SAFE_NUM_WORKERS)

    if _is_main_process():
        logger.info(f"Creating instruction dataloader with {num_workers} workers")

    dataset = InstructionDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        trial_mode=trial_mode,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def get_reasoning_dataloader(
    data_dir: str,
    tokenizer: Any,
    batch_size: int = 32,
    split: str = 'train',
    distributed: bool = False,
    num_workers: int = None,
    **kwargs,
) -> DataLoader:
    """
    Create dataloader for reasoning training.

    MEMORY SAFE: Uses limited num_workers.
    """
    import logging
    logger = logging.getLogger(__name__)

    if num_workers is None:
        num_workers = SAFE_NUM_WORKERS
    else:
        num_workers = min(num_workers, SAFE_NUM_WORKERS)

    if _is_main_process():
        logger.info(f"Creating reasoning dataloader with {num_workers} workers")

    dataset = ReasoningDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        **kwargs,
    )

    sampler = None
    if distributed and split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function with error handling."""
    result = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch if key in item]

        if len(values) == 0:
            continue

        if isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values)
            except Exception:
                # If stacking fails, skip this key
                pass
        else:
            result[key] = values

    return result

