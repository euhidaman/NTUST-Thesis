#!/usr/bin/env python3
"""
MicroVLM-E Complete Dataset Downloader with Enhanced Logging & Monitoring
Downloads all required datasets with comprehensive tracking and WandB integration.

Usage:
    python download-new.py

Optional flags:
    --skip-cc3m          Skip CC3M (dead URLs)
    --skip-laion         Skip LAION (large download)
    --minimal            Download only core datasets (COCO, VQA, LLaVA)
    --data-dir PATH      Custom data directory (default: ./data)
    --log-level LEVEL    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    --no-wandb           Disable WandB logging
    --retry-limit N      Number of retries for failed downloads (default: 3)
"""

import os
import sys
import json
import zipfile
import tarfile
import shutil
import argparse
import logging
import hashlib
import psutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse
from datetime import datetime
from collections import defaultdict
import time
import threading

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests tqdm psutil")
    import requests
    from tqdm import tqdm

# HuggingFace datasets - install if needed
try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    print("Installing HuggingFace libraries...")
    os.system(f"{sys.executable} -m pip install datasets huggingface_hub")
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True

# WandB - install if needed and enabled
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Constants
WANDB_PROJECT = "MicroVLM-E-datasets-logs"
COUNTER_FILE = ".download_counter.json"
LOG_FILE = "download_session.log"

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: str = LOG_FILE) -> logging.Logger:
    """Setup comprehensive logging with file and console handlers."""
    logger = logging.getLogger("DatasetDownloader")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


class DownloadCounter:
    """Persistent counter for tracking download statistics across runs."""

    def __init__(self, counter_file: str = COUNTER_FILE):
        self.counter_file = Path(counter_file)
        self.data = self._load()
        self.current_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'downloads_attempted': 0,
            'downloads_successful': 0,
            'downloads_failed': 0,
            'total_bytes_downloaded': 0,
            'total_bytes_attempted': 0,
            'files': [],
            'errors': []
        }

    def _load(self) -> Dict:
        """Load counter from file."""
        if self.counter_file.exists():
            try:
                with open(self.counter_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {'total_runs': 0, 'sessions': []}
        return {'total_runs': 0, 'sessions': []}

    def save(self):
        """Save counter to file."""
        self.current_session['end_time'] = datetime.now().isoformat()
        self.data['sessions'].append(self.current_session)
        self.data['total_runs'] += 1

        with open(self.counter_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def log_download(self, filename: str, size: int, success: bool, url: str = "", error: str = ""):
        """Log a download attempt."""
        self.current_session['downloads_attempted'] += 1
        self.current_session['total_bytes_attempted'] += size if size > 0 else 0

        if success:
            self.current_session['downloads_successful'] += 1
            self.current_session['total_bytes_downloaded'] += size
        else:
            self.current_session['downloads_failed'] += 1
            if error:
                self.current_session['errors'].append({
                    'filename': filename,
                    'url': url,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })

        self.current_session['files'].append({
            'filename': filename,
            'size_bytes': size,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'url': url
        })


class DatasetDownloader:
    def __init__(self, data_dir: str = "./data", logger: logging.Logger = None,
                 use_wandb: bool = True, retry_limit: int = 3):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("DatasetDownloader")
        self.retry_limit = retry_limit

        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Initialize counter
        self.counter = DownloadCounter()

        # Initialize WandB if available and enabled
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("WandB not available. Install with: pip install wandb")

        # Performance monitoring
        self.start_time = time.time()
        self.download_stats = defaultdict(lambda: {'size': 0, 'time': 0, 'speed': 0})

        self.logger.info(f"Initialized DatasetDownloader - Data dir: {self.data_dir.absolute()}")
        self.logger.info(f"Session ID: {self.counter.current_session['session_id']}")
        self.logger.info(f"Retry limit: {self.retry_limit}")

    def _init_wandb(self):
        """Initialize WandB logging."""
        try:
            run_number = self.counter.data.get('total_runs', 0) + 1
            run_name = f"download-session-{run_number}"

            self.wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                config={
                    'data_dir': str(self.data_dir),
                    'retry_limit': self.retry_limit,
                    'session_id': self.counter.current_session['session_id']
                },
                tags=['dataset_download', f'run_{run_number}']
            )

            self.logger.info(f"WandB initialized: {run_name}")
            self.logger.info(f"View logs at: {self.wandb_run.url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            self.wandb_run = None

    def _log_to_wandb(self, metrics: Dict):
        """Log metrics to WandB."""
        if self.wandb_run:
            try:
                wandb.log(metrics)
            except Exception as e:
                self.logger.debug(f"WandB logging failed: {e}")

    def _verify_path(self, path: Path, expected_parent: str = None) -> bool:
        """Verify file path is in expected location."""
        try:
            # Check path is under data_dir
            path.resolve().relative_to(self.data_dir.resolve())

            if expected_parent:
                # Check immediate parent matches expected
                if path.parent.name != expected_parent:
                    self.logger.warning(f"Path verification: {path} not in expected parent '{expected_parent}'")
                    return False

            self.logger.debug(f"Path verified: {path}")
            return True
        except ValueError:
            self.logger.error(f"Path verification failed: {path} is outside data directory")
            return False

    def _calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> Optional[str]:
        """Calculate file checksum for verification."""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            checksum = hash_obj.hexdigest()
            self.logger.debug(f"Checksum ({algorithm}): {checksum} for {file_path.name}")
            return checksum
        except Exception as e:
            self.logger.error(f"Checksum calculation failed for {file_path}: {e}")
            return None

    def _check_disk_space(self, required_bytes: int = 0) -> Tuple[bool, int]:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage(str(self.data_dir))
            available_gb = disk_usage.free / (1024**3)
            required_gb = required_bytes / (1024**3)

            self.logger.debug(f"Disk space: {available_gb:.2f} GB available")

            if required_bytes > 0 and disk_usage.free < required_bytes:
                self.logger.warning(f"Low disk space: {available_gb:.2f} GB available, {required_gb:.2f} GB required")
                return False, disk_usage.free

            return True, disk_usage.free
        except Exception as e:
            self.logger.error(f"Disk space check failed: {e}")
            return True, 0  # Continue anyway

    def _monitor_resources(self) -> Dict:
        """Monitor system resources during download."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
            }
        except Exception as e:
            self.logger.debug(f"Resource monitoring failed: {e}")
            return {}

    def download_file(self, url: str, dest_path: Path, desc: str = None,
                      expected_size: int = 0, verify_checksum: bool = False) -> bool:
        """
        Download a file with comprehensive logging, retry logic, and monitoring.

        Args:
            url: Source URL
            dest_path: Destination file path
            desc: Description for progress bar
            expected_size: Expected file size in bytes (for validation)
            verify_checksum: Whether to calculate and log checksum

        Returns:
            True if download successful, False otherwise
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Verify path
        if not self._verify_path(dest_path):
            self.logger.error(f"Path verification failed for {dest_path}")
            return False

        # Check if already downloaded
        if dest_path.exists():
            file_size = dest_path.stat().st_size
            self.logger.info(f"✓ Already exists: {dest_path.name} ({file_size / (1024**2):.2f} MB)")

            # Log to counter as successful (existing file)
            self.counter.log_download(dest_path.name, file_size, True, url)

            # Verify checksum if requested
            if verify_checksum:
                checksum = self._calculate_checksum(dest_path)
                self.logger.info(f"Checksum: {checksum}")

            return True

        # Check disk space
        has_space, available = self._check_disk_space(expected_size)
        if not has_space:
            self.logger.error(f"Insufficient disk space for {dest_path.name}")
            self.counter.log_download(dest_path.name, 0, False, url, "Insufficient disk space")
            return False

        temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')

        # Retry loop
        for attempt in range(1, self.retry_limit + 1):
            try:
                self.logger.info(f"Downloading {dest_path.name} (attempt {attempt}/{self.retry_limit})")
                self.logger.debug(f"URL: {url}")
                self.logger.debug(f"Destination: {dest_path}")

                # Check for resume
                resume_pos = temp_path.stat().st_size if temp_path.exists() else 0
                if resume_pos > 0:
                    self.logger.info(f"Resuming from byte {resume_pos}")

                headers = {'Range': f'bytes={resume_pos}-'} if resume_pos > 0 else {}

                # Start download
                download_start = time.time()
                response = self.session.get(url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0)) + resume_pos
                mode = 'ab' if resume_pos > 0 else 'wb'

                self.logger.debug(f"Total size: {total_size / (1024**2):.2f} MB")

                # Monitor resources
                resources_start = self._monitor_resources()

                bytes_downloaded = 0
                with open(temp_path, mode) as f:
                    with tqdm(total=total_size, initial=resume_pos, unit='B',
                              unit_scale=True, desc=desc or dest_path.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                bytes_downloaded += len(chunk)

                download_time = time.time() - download_start
                download_speed = bytes_downloaded / download_time if download_time > 0 else 0

                # Rename temp to final
                temp_path.rename(dest_path)

                # Verify size
                actual_size = dest_path.stat().st_size
                if expected_size > 0 and actual_size != expected_size:
                    self.logger.warning(f"Size mismatch: expected {expected_size}, got {actual_size}")

                # Calculate checksum if requested
                checksum = None
                if verify_checksum:
                    checksum = self._calculate_checksum(dest_path)

                # Monitor resources after download
                resources_end = self._monitor_resources()

                # Log success
                self.logger.info(f"✓ Downloaded: {dest_path.name} ({actual_size / (1024**2):.2f} MB)")
                self.logger.info(f"  Speed: {download_speed / (1024**2):.2f} MB/s")
                self.logger.info(f"  Time: {download_time:.1f}s")
                if checksum:
                    self.logger.info(f"  Checksum: {checksum[:16]}...")

                # Update counter
                self.counter.log_download(dest_path.name, actual_size, True, url)

                # Log to WandB
                self._log_to_wandb({
                    'download_success': 1,
                    'download_size_mb': actual_size / (1024**2),
                    'download_speed_mbps': download_speed / (1024**2),
                    'download_time_sec': download_time,
                    'cpu_percent': resources_end.get('cpu_percent', 0),
                    'memory_percent': resources_end.get('memory_percent', 0)
                })

                # Store stats
                self.download_stats[dest_path.name] = {
                    'size': actual_size,
                    'time': download_time,
                    'speed': download_speed,
                    'checksum': checksum
                }

                return True

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"✗ Download failed (attempt {attempt}/{self.retry_limit}): {error_msg}")
                self.logger.debug(f"Exception details: {type(e).__name__}: {e}")

                # Clean up partial download on last attempt
                if attempt == self.retry_limit:
                    if temp_path.exists():
                        self.logger.debug(f"Cleaning up partial download: {temp_path}")
                        try:
                            temp_path.unlink()
                        except Exception as cleanup_error:
                            self.logger.debug(f"Cleanup failed: {cleanup_error}")

                    # Log failure
                    self.counter.log_download(dest_path.name, 0, False, url, error_msg)

                    # Log to WandB
                    self._log_to_wandb({
                        'download_failure': 1,
                        'error': error_msg[:100]  # Truncate long errors
                    })

                    return False

                # Wait before retry
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract zip or tar.gz archive with logging and monitoring.

        Args:
            archive_path: Path to archive file
            extract_to: Destination directory

        Returns:
            True if extraction successful, False otherwise
        """
        if not archive_path.exists():
            self.logger.error(f"✗ Archive not found: {archive_path}")
            return False

        # Check if already extracted
        marker_file = extract_to / f".extracted_{archive_path.name}.marker"
        if marker_file.exists():
            self.logger.info(f"✓ Already extracted: {archive_path.name}")
            return True

        self.logger.info(f"Extracting {archive_path.name}...")
        self.logger.debug(f"Archive: {archive_path}")
        self.logger.debug(f"Destination: {extract_to}")

        extract_to.mkdir(parents=True, exist_ok=True)

        extract_start = time.time()
        resources_start = self._monitor_resources()

        try:
            # Count files to extract
            file_count = 0
            total_size = 0

            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    file_count = len(zf.namelist())
                    total_size = sum(info.file_size for info in zf.infolist())
                    self.logger.debug(f"ZIP contains {file_count} files, {total_size / (1024**2):.2f} MB")

                    # Extract with progress
                    for member in tqdm(zf.namelist(), desc=f"Extracting {archive_path.name}"):
                        zf.extract(member, extract_to)

            elif archive_path.suffix == '.gz' or '.tar' in archive_path.suffixes:
                with tarfile.open(archive_path, 'r:*') as tf:
                    members = tf.getmembers()
                    file_count = len(members)
                    total_size = sum(m.size for m in members)
                    self.logger.debug(f"TAR contains {file_count} files, {total_size / (1024**2):.2f} MB")

                    # Extract with progress
                    for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                        tf.extract(member, extract_to)
            else:
                self.logger.error(f"✗ Unknown archive format: {archive_path}")
                return False

            extract_time = time.time() - extract_start
            resources_end = self._monitor_resources()

            # Create marker file
            marker_file.touch()
            with open(marker_file, 'w') as f:
                json.dump({
                    'archive': archive_path.name,
                    'extracted_at': datetime.now().isoformat(),
                    'file_count': file_count,
                    'total_size_bytes': total_size,
                    'extraction_time_sec': extract_time
                }, f, indent=2)

            self.logger.info(f"✓ Extracted: {archive_path.name}")
            self.logger.info(f"  Files: {file_count}")
            self.logger.info(f"  Size: {total_size / (1024**2):.2f} MB")
            self.logger.info(f"  Time: {extract_time:.1f}s")

            # Log to WandB
            self._log_to_wandb({
                'extraction_success': 1,
                'extracted_files': file_count,
                'extracted_size_mb': total_size / (1024**2),
                'extraction_time_sec': extract_time,
                'cpu_percent': resources_end.get('cpu_percent', 0)
            })

            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"✗ Failed to extract {archive_path}: {error_msg}")
            self.logger.debug(f"Exception details: {type(e).__name__}: {e}")

            # Log to WandB
            self._log_to_wandb({
                'extraction_failure': 1,
                'error': error_msg[:100]
            })

            return False

    def download_coco(self) -> bool:
        """Download COCO 2017 dataset with comprehensive logging."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DOWNLOADING COCO 2017 (Primary Dataset - ~25GB)")
        self.logger.info("=" * 80)

        dataset_start = time.time()
        coco_dir = self.data_dir / "coco"
        coco_dir.mkdir(exist_ok=True)

        self.logger.debug(f"COCO directory: {coco_dir.absolute()}")

        urls = [
            ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip", 18 * 1024**3),  # ~18GB
            ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip", 800 * 1024**2),  # ~800MB
            ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip", 250 * 1024**2),  # ~250MB
        ]

        success_count = 0
        for url, filename, expected_size in urls:
            archive_path = coco_dir / filename
            self.logger.info(f"\nProcessing: {filename}")

            if not self.download_file(url, archive_path, f"COCO {filename}", expected_size=expected_size):
                self.logger.error(f"Failed to download {filename}")
                return False
            success_count += 1

            # Extract if not already extracted
            extract_name = filename.replace('.zip', '')
            if not (coco_dir / extract_name).exists():
                if not self.extract_archive(archive_path, coco_dir):
                    self.logger.warning(f"Extraction failed for {filename}, but continuing...")

        dataset_time = time.time() - dataset_start
        self.logger.info(f"\n✓ COCO 2017 completed in {dataset_time:.1f}s ({success_count}/{len(urls)} files)")

        # Log to WandB
        self._log_to_wandb({
            'coco_complete': 1,
            'coco_files': success_count,
            'coco_duration_sec': dataset_time
        })

        return True

    def download_vqav2(self) -> bool:
        """Download VQA v2 dataset with comprehensive logging."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DOWNLOADING VQA v2 (~200MB)")
        self.logger.info("=" * 80)

        dataset_start = time.time()
        vqa_dir = self.data_dir / "vqa"
        vqa_dir.mkdir(exist_ok=True)

        self.logger.debug(f"VQA directory: {vqa_dir.absolute()}")

        urls = [
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        ]

        success_count = 0
        for url in urls:
            filename = Path(urlparse(url).path).name
            archive_path = vqa_dir / filename
            self.logger.info(f"\nProcessing: {filename}")

            if not self.download_file(url, archive_path, f"VQA {filename}"):
                self.logger.error(f"Failed to download {filename}")
                return False
            success_count += 1

            if not self.extract_archive(archive_path, vqa_dir):
                self.logger.warning(f"Extraction failed for {filename}, but continuing...")

        dataset_time = time.time() - dataset_start
        self.logger.info(f"\n✓ VQA v2 completed in {dataset_time:.1f}s ({success_count}/{len(urls)} files)")

        self._log_to_wandb({
            'vqav2_complete': 1,
            'vqav2_files': success_count,
            'vqav2_duration_sec': dataset_time
        })

        return True

    def download_okvqa(self) -> bool:
        """Download OK-VQA dataset."""
        print("\n" + "=" * 80)
        print("DOWNLOADING OK-VQA (~50MB)")
        print("=" * 80)

        okvqa_dir = self.data_dir / "okvqa"
        okvqa_dir.mkdir(exist_ok=True)

        urls = [
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
            "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
            "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
        ]

        for url in urls:
            filename = Path(urlparse(url).path).name
            archive_path = okvqa_dir / filename
            if not self.download_file(url, archive_path, f"OK-VQA {filename}"):
                return False
            self.extract_archive(archive_path, okvqa_dir)

        print("✓ OK-VQA completed")
        return True

    def download_aokvqa(self) -> bool:
        """Download A-OKVQA dataset."""
        print("\n" + "=" * 80)
        print("DOWNLOADING A-OKVQA (~100MB)")
        print("=" * 80)

        aokvqa_dir = self.data_dir / "aokvqa"
        aokvqa_dir.mkdir(exist_ok=True)

        url = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
        archive_path = aokvqa_dir / "aokvqa_v1p0.tar.gz"

        if self.download_file(url, archive_path, "A-OKVQA"):
            self.extract_archive(archive_path, aokvqa_dir)
            print("✓ A-OKVQA completed")
            return True
        return False

    def download_gqa(self) -> bool:
        """Download GQA dataset."""
        print("\n" + "=" * 80)
        print("DOWNLOADING GQA (~20GB)")
        print("=" * 80)

        gqa_dir = self.data_dir / "gqa"
        gqa_dir.mkdir(exist_ok=True)

        urls = [
            ("https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip", "questions1.2.zip"),
            ("https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip", "images.zip"),
        ]

        for url, filename in urls:
            archive_path = gqa_dir / filename
            if not self.download_file(url, archive_path, f"GQA {filename}"):
                return False
            self.extract_archive(archive_path, gqa_dir)

        print("✓ GQA completed")
        return True

    def download_ocrvqa(self) -> bool:
        """Download OCR-VQA from Google Drive."""
        print("\n" + "=" * 80)
        print("DOWNLOADING OCR-VQA (~5GB)")
        print("=" * 80)

        try:
            import gdown
        except ImportError:
            print("Installing gdown for Google Drive downloads...")
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown

        ocrvqa_dir = self.data_dir / "ocrvqa"
        ocrvqa_dir.mkdir(exist_ok=True)

        # Google Drive folder ID
        folder_id = "1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_"
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

        print(f"Downloading from Google Drive: {folder_url}")
        try:
            gdown.download_folder(folder_url, output=str(ocrvqa_dir), quiet=False, use_cookies=False)
            print("✓ OCR-VQA completed")
            return True
        except Exception as e:
            print(f"✗ OCR-VQA failed: {e}")
            print("  Note: May require manual download from Google Drive")
            return False

    def download_refcoco_datasets(self) -> bool:
        """Download RefCOCO, RefCOCO+, RefCOCOg from HuggingFace."""
        print("\n" + "=" * 80)
        print("DOWNLOADING RefCOCO DATASETS (~150MB total)")
        print("=" * 80)

        datasets_info = [
            ("lmms-lab/RefCOCO", "refcoco"),
            ("lmms-lab/RefCOCOplus", "refcoco_plus"),
            ("lmms-lab/RefCOCOg", "refcocog"),
        ]

        for hf_name, local_name in datasets_info:
            print(f"\nDownloading {local_name}...")
            local_dir = self.data_dir / local_name
            local_dir.mkdir(exist_ok=True)

            try:
                dataset = load_dataset(hf_name, cache_dir=str(local_dir))
                print(f"✓ {local_name} completed")
            except Exception as e:
                print(f"✗ {local_name} failed: {e}")
                return False

        return True

    def download_llava_instruct(self) -> bool:
        """Download LLaVA instruction tuning dataset."""
        print("\n" + "=" * 80)
        print("DOWNLOADING LLaVA Instruct (~300MB)")
        print("=" * 80)

        llava_dir = self.data_dir / "llava"
        llava_dir.mkdir(exist_ok=True)

        url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
        json_path = llava_dir / "llava_instruct_150k.json"

        if self.download_file(url, json_path, "LLaVA Instruct"):
            print("✓ LLaVA Instruct completed")
            return True
        return False

    def download_laion(self) -> bool:
        """Download LAION-COCO subset from HuggingFace."""
        print("\n" + "=" * 80)
        print("DOWNLOADING LAION-COCO (~50-100GB)")
        print("Warning: This is a large download and may take several hours")
        print("=" * 80)

        laion_dir = self.data_dir / "laion"
        laion_dir.mkdir(exist_ok=True)

        # Create a marker to skip if already attempted
        marker = laion_dir / ".download_attempted"
        if marker.exists():
            print("⚠ LAION download previously attempted. Skipping.")
            print("  Delete data/laion/.download_attempted to retry")
            return True

        print("Downloading LAION-COCO from HuggingFace...")
        print("This will be stored in HuggingFace cache format.")

        try:
            dataset = load_dataset(
                "laion/relaion2B-en-research-safe",
                cache_dir=str(laion_dir / "dataset"),
                streaming=False
            )
            marker.touch()
            print("✓ LAION-COCO completed")
            return True
        except Exception as e:
            print(f"✗ LAION download failed: {e}")
            print("  Continuing with other datasets...")
            marker.touch()  # Mark as attempted
            return False

    def download_cc3m(self) -> bool:
        """Download CC3M from HuggingFace WebDataset format."""
        print("\n" + "=" * 80)
        print("DOWNLOADING CC3M from HuggingFace (~100-200GB)")
        print("Using pixparse/cc3m-wds (WebDataset format with images)")
        print("=" * 80)

        cc3m_dir = self.data_dir / "cc3m"
        cc3m_dir.mkdir(exist_ok=True)

        # Create a marker to skip if already attempted
        marker = cc3m_dir / ".download_attempted"
        if marker.exists():
            print("⚠ CC3M download previously attempted. Skipping.")
            print("  Delete data/cc3m/.download_attempted to retry")
            return True

        print("Downloading CC3M WebDataset from HuggingFace...")
        print("This will be stored in HuggingFace cache format with images included.")

        try:
            # Download the dataset - images are included in WebDataset format
            dataset = load_dataset(
                "pixparse/cc3m-wds",
                cache_dir=str(cc3m_dir / "dataset"),
                streaming=False  # Download full dataset
            )
            marker.touch()

            # Save info about the dataset
            with open(cc3m_dir / "README.txt", "w") as f:
                f.write("""CC3M Dataset - HuggingFace WebDataset Format

Downloaded from: pixparse/cc3m-wds
Format: WebDataset with images included
Location: data/cc3m/dataset/

This dataset includes:
- Images embedded in the dataset
- Captions for each image
- ~3M image-text pairs

To load in your training code:
    from datasets import load_dataset
    dataset = load_dataset("pixparse/cc3m-wds", cache_dir="data/cc3m/dataset")
    image = dataset['train'][0]['image']  # PIL Image
    caption = dataset['train'][0]['caption']  # Text
""")

            print("✓ CC3M completed")
            return True

        except Exception as e:
            print(f"✗ CC3M download failed: {e}")
            print("  Continuing with other datasets...")
            marker.touch()  # Mark as attempted
            return False

    def download_flickr30k(self) -> bool:
        """Download Flickr30k from HuggingFace."""
        print("\n" + "=" * 80)
        print("DOWNLOADING Flickr30k from HuggingFace (~5-10GB)")
        print("Using nlphuji/flickr30k (with images)")
        print("=" * 80)

        flickr_dir = self.data_dir / "flickr30k"
        flickr_dir.mkdir(exist_ok=True)

        # Create a marker to skip if already attempted
        marker = flickr_dir / ".download_attempted"
        if marker.exists():
            print("⚠ Flickr30k download previously attempted. Skipping.")
            print("  Delete data/flickr30k/.download_attempted to retry")
            return True

        print("Downloading Flickr30k from HuggingFace...")
        print("This includes images and captions.")

        try:
            # Download the dataset - images are included
            dataset = load_dataset(
                "nlphuji/flickr30k",
                cache_dir=str(flickr_dir / "dataset"),
                streaming=False
            )
            marker.touch()

            # Save info about the dataset
            with open(flickr_dir / "README.txt", "w") as f:
                f.write("""Flickr30k Dataset - HuggingFace Format

Downloaded from: nlphuji/flickr30k
Format: HuggingFace dataset with images included
Location: data/flickr30k/dataset/

This dataset includes:
- Images embedded in the dataset
- Multiple captions per image (5 captions each)
- ~31K images

To load in your training code:
    from datasets import load_dataset
    dataset = load_dataset("nlphuji/flickr30k", cache_dir="data/flickr30k/dataset")
    image = dataset['test'][0]['image']  # PIL Image
    captions = dataset['test'][0]['caption']  # List of 5 captions
""")

            print("✓ Flickr30k completed")
            return True

        except Exception as e:
            print(f"✗ Flickr30k download failed: {e}")
            print("  Continuing with other datasets...")
            marker.touch()  # Mark as attempted
            return False

    def generate_summary_report(self) -> Dict:
        """Generate comprehensive download summary report."""
        total_time = time.time() - self.start_time

        # Calculate totals
        total_downloaded = self.counter.current_session['total_bytes_downloaded']
        total_attempted = self.counter.current_session['total_bytes_attempted']
        success_count = self.counter.current_session['downloads_successful']
        failed_count = self.counter.current_session['downloads_failed']
        total_count = self.counter.current_session['downloads_attempted']

        # Calculate average speed
        avg_speed = total_downloaded / total_time if total_time > 0 else 0

        # Get final resource usage
        final_resources = self._monitor_resources()

        report = {
            'session_id': self.counter.current_session['session_id'],
            'start_time': self.counter.current_session['start_time'],
            'end_time': datetime.now().isoformat(),
            'total_duration_sec': total_time,
            'total_duration_readable': f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s",
            'downloads': {
                'attempted': total_count,
                'successful': success_count,
                'failed': failed_count,
                'success_rate': (success_count / total_count * 100) if total_count > 0 else 0
            },
            'data_transfer': {
                'total_downloaded_bytes': total_downloaded,
                'total_downloaded_gb': total_downloaded / (1024**3),
                'average_speed_mbps': avg_speed / (1024**2),
                'peak_speed_mbps': max([s['speed'] / (1024**2) for s in self.download_stats.values()], default=0)
            },
            'resources': final_resources,
            'errors': self.counter.current_session['errors']
        }

        return report

    def print_summary(self):
        """Print comprehensive download summary with statistics."""
        report = self.generate_summary_report()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("DOWNLOAD SESSION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Session ID: {report['session_id']}")
        self.logger.info(f"Duration: {report['total_duration_readable']}")
        self.logger.info("")
        self.logger.info("DOWNLOAD STATISTICS:")
        self.logger.info(f"  Total Attempts: {report['downloads']['attempted']}")
        self.logger.info(f"  Successful: {report['downloads']['successful']}")
        self.logger.info(f"  Failed: {report['downloads']['failed']}")
        self.logger.info(f"  Success Rate: {report['downloads']['success_rate']:.1f}%")
        self.logger.info("")
        self.logger.info("DATA TRANSFER:")
        self.logger.info(f"  Total Downloaded: {report['data_transfer']['total_downloaded_gb']:.2f} GB")
        self.logger.info(f"  Average Speed: {report['data_transfer']['average_speed_mbps']:.2f} MB/s")
        self.logger.info(f"  Peak Speed: {report['data_transfer']['peak_speed_mbps']:.2f} MB/s")

        if report['resources']:
            self.logger.info("")
            self.logger.info("SYSTEM RESOURCES:")
            self.logger.info(f"  CPU Usage: {report['resources'].get('cpu_percent', 0):.1f}%")
            self.logger.info(f"  Memory Usage: {report['resources'].get('memory_percent', 0):.1f}%")
            self.logger.info(f"  Memory Available: {report['resources'].get('memory_available_gb', 0):.2f} GB")

        if report['errors']:
            self.logger.info("")
            self.logger.info(f"ERRORS ({len(report['errors'])}):")
            for error in report['errors'][:5]:  # Show first 5 errors
                self.logger.info(f"  - {error['filename']}: {error['error'][:80]}")
            if len(report['errors']) > 5:
                self.logger.info(f"  ... and {len(report['errors']) - 5} more errors")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("DATASET STATUS")
        self.logger.info("=" * 80)

        datasets = {
            "coco": "COCO 2017 (Primary)",
            "vqa": "VQA v2",
            "okvqa": "OK-VQA",
            "aokvqa": "A-OKVQA",
            "gqa": "GQA",
            "ocrvqa": "OCR-VQA",
            "refcoco": "RefCOCO",
            "refcoco_plus": "RefCOCO+",
            "refcocog": "RefCOCOg",
            "llava": "LLaVA Instruct",
            "laion": "LAION-COCO",
            "cc3m": "CC3M",
        }

        self.logger.info("\nDataset Status:")
        for dir_name, display_name in datasets.items():
            dataset_dir = self.data_dir / dir_name
            if dataset_dir.exists():
                # Calculate size
                try:
                    size_bytes = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
                    size_gb = size_bytes / (1024 ** 3)
                    file_count = sum(1 for f in dataset_dir.rglob('*') if f.is_file())
                    status = f"✓ {size_gb:.2f} GB ({file_count} files)"
                except Exception as e:
                    status = f"✓ Present (size calculation failed)"
                    self.logger.debug(f"Size calculation failed for {dir_name}: {e}")
            else:
                status = "✗ Not downloaded"
            self.logger.info(f"  {display_name:30s} {status}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING READINESS")
        self.logger.info("=" * 80)

        required = [
            ("coco", "COCO images", "REQUIRED for most tasks"),
            ("vqa", "VQA annotations", "REQUIRED for VQA tasks"),
            ("llava", "LLaVA instructions", "REQUIRED for instruction tuning"),
        ]

        optional = [
            ("gqa", "GQA", "Optional: Advanced reasoning"),
            ("okvqa", "OK-VQA", "Optional: Knowledge-based QA"),
            ("aokvqa", "A-OKVQA", "Optional: Augmented knowledge QA"),
            ("ocrvqa", "OCR-VQA", "Optional: Text recognition"),
            ("refcoco", "RefCOCO", "Optional: Referring expressions"),
            ("laion", "LAION", "Optional: Additional image-text pairs"),
        ]

        self.logger.info("\nRequired Datasets:")
        all_required_ready = True
        for dir_name, name, desc in required:
            exists = (self.data_dir / dir_name).exists()
            status = "✓ Ready" if exists else "✗ Missing"
            self.logger.info(f"  {status} {name:20s} - {desc}")
            if not exists:
                all_required_ready = False

        self.logger.info("\nOptional Datasets:")
        for dir_name, name, desc in optional:
            status = "✓ Ready" if (self.data_dir / dir_name).exists() else "○ Not downloaded"
            self.logger.info(f"  {status} {name:20s} - {desc}")

        self.logger.info("\n" + "=" * 80)
        if all_required_ready:
            self.logger.info("✓ ALL REQUIRED DATASETS READY FOR TRAINING!")
        else:
            self.logger.warning("⚠ Some required datasets are missing!")

        self.logger.info("\nNext Steps:")
        self.logger.info(f"  1. Review logs: {LOG_FILE}")
        self.logger.info(f"  2. Check counter: {COUNTER_FILE}")
        if self.wandb_run:
            self.logger.info(f"  3. View WandB dashboard: {self.wandb_run.url}")
            self.logger.info(f"  4. Verify paths and start training")
        else:
            self.logger.info(f"  3. Verify paths and start training")
        self.logger.info("=" * 80 + "\n")

        # Save summary to file
        summary_file = self.data_dir / "download_summary.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Summary saved to: {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download MicroVLM-E datasets with comprehensive logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets with full logging
  python download-new.py
  
  # Download only core datasets
  python download-new.py --minimal
  
  # Skip large datasets
  python download-new.py --skip-cc3m --skip-laion
  
  # Debug mode with verbose logging
  python download-new.py --log-level DEBUG
  
  # Without WandB logging
  python download-new.py --no-wandb
        """
    )
    parser.add_argument('--data-dir', default='./data', help='Data directory (default: ./data)')
    parser.add_argument('--minimal', action='store_true', help='Download only core datasets')
    parser.add_argument('--skip-cc3m', action='store_true', help='Skip CC3M dataset')
    parser.add_argument('--skip-laion', action='store_true', help='Skip LAION dataset')
    parser.add_argument('--skip-ocrvqa', action='store_true', help='Skip OCR-VQA dataset')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--retry-limit', type=int, default=3,
                       help='Number of retries for failed downloads (default: 3)')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level, LOG_FILE)

    logger.info("=" * 80)
    logger.info("MicroVLM-E Dataset Downloader - Enhanced Version")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Mode: {'MINIMAL' if args.minimal else 'FULL'}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"WandB logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    logger.info(f"Retry limit: {args.retry_limit}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Counter file: {COUNTER_FILE}")
    logger.info("=" * 80 + "\n")

    # Check if WandB should be installed
    if not args.no_wandb and not WANDB_AVAILABLE:
        logger.info("Installing WandB for experiment tracking...")
        try:
            os.system(f"{sys.executable} -m pip install wandb")
            import wandb
            globals()['WANDB_AVAILABLE'] = True
            logger.info("✓ WandB installed successfully")
        except Exception as e:
            logger.warning(f"Failed to install WandB: {e}")
            logger.warning("Continuing without WandB logging...")

    downloader = DatasetDownloader(
        args.data_dir,
        logger=logger,
        use_wandb=not args.no_wandb,
        retry_limit=args.retry_limit
    )

    # Core datasets (always download)
    print("PHASE 1: Core Datasets (Required)")
    downloader.download_coco()
    downloader.download_vqav2()
    downloader.download_llava_instruct()

    if not args.minimal:
        print("\nPHASE 2: VQA Datasets")
        downloader.download_okvqa()
        downloader.download_aokvqa()
        downloader.download_gqa()

        if not args.skip_ocrvqa:
            downloader.download_ocrvqa()

        print("\nPHASE 3: Referring Expression Datasets")
        downloader.download_refcoco_datasets()

        print("\nPHASE 4: Large-Scale Datasets")
        if not args.skip_laion:
            downloader.download_laion()

        if not args.skip_cc3m:
            downloader.download_cc3m()

    # Print summary
    downloader.print_summary()

    # Save counter
    try:
        downloader.counter.save()
        logger.info(f"✓ Session statistics saved to: {COUNTER_FILE}")
    except Exception as e:
        logger.error(f"Failed to save counter: {e}")

    # Log final metrics to WandB
    if downloader.wandb_run:
        try:
            report = downloader.generate_summary_report()
            wandb.log({
                'session_complete': 1,
                'total_downloads': report['downloads']['attempted'],
                'successful_downloads': report['downloads']['successful'],
                'failed_downloads': report['downloads']['failed'],
                'total_gb_downloaded': report['data_transfer']['total_downloaded_gb'],
                'total_duration_sec': report['total_duration_sec'],
                'success_rate': report['downloads']['success_rate']
            })

            # Save summary as artifact
            artifact = wandb.Artifact('download_summary', type='summary')
            summary_file = Path(args.data_dir) / "download_summary.json"
            if summary_file.exists():
                artifact.add_file(str(summary_file))
                wandb.log_artifact(artifact)

            wandb.finish()
            logger.info("✓ WandB session closed")
        except Exception as e:
            logger.error(f"Failed to finalize WandB: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL DOWNLOADS COMPLETED!")
    logger.info(f"Total data location: {Path(args.data_dir).absolute()}")
    logger.info(f"Review detailed logs: {LOG_FILE}")
    logger.info(f"Session statistics: {COUNTER_FILE}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        print(f"Session logs saved to: {LOG_FILE}")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger("DatasetDownloader")
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Fatal error occurred. Check logs: {LOG_FILE}")
        sys.exit(1)
