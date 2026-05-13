# Dataset Overview

**Complete Guide to All Datasets Used in EmberVLM Training**

## 📋 Table of Contents

- [Overview](#overview)
- [Stage 1 Datasets](#stage-1-datasets)
- [Stage 2 Datasets](#stage-2-datasets)
- [Stage 3 Datasets](#stage-3-datasets)
- [Stage 4 Datasets](#stage-4-datasets)
- [Data Preparation](#data-preparation)
- [Data Format Specifications](#data-format-specifications)
- [Download Instructions](#download-instructions)

## 🎯 Overview

EmberVLM training uses carefully curated datasets across 4 stages, progressing from general vision-language alignment to domain-specific robot selection with reasoning capabilities.

### Dataset Summary

| Stage | Datasets | Total Samples | Purpose |
|-------|----------|---------------|---------|
| **1** | COCO, Flickr30k, CC3M | 330,000 | Visual-language alignment |
| **2** | LLaVA, VQA-v2, OK-VQA | 300,000 | Instruction following |
| **3** | Robot Selection | 1,000 → 10,000 | Robot fleet selection |
| **4** | Reasoning-Annotated | 50,000 | Chain-of-thought reasoning |
| **Total** | **8 datasets** | **390,000 unique** | **Progressive training** |

## 📊 Stage 1 Datasets

### Visual-Language Alignment Datasets

#### 1. COCO Captions (100K samples)

**Description**: Microsoft COCO 2017 - Common Objects in Context

**Statistics**:
- **Images**: 100,000 training images
- **Captions**: 5 captions per image = 500,000 total
- **Vocabulary**: ~12,000 unique words
- **Avg Caption Length**: 10.5 words
- **Domains**: Everyday scenes, objects, people

**Download**:
```bash
# Download COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip -d data/coco/images/
unzip annotations_trainval2017.zip -d data/coco/
```

**Data Format**:
```json
{
  "image_id": 123456,
  "file_name": "COCO_train2017_000000123456.jpg",
  "captions": [
    "A person riding a bike down the street",
    "Someone cycling on an urban road",
    "A cyclist traveling through the city",
    "A man on a bicycle in traffic",
    "Bicyclist riding on a city street"
  ]
}
```

#### 2. Flickr30k (30K samples)

**Description**: Flickr30k Entities - detailed image descriptions

**Statistics**:
- **Images**: 30,000 images
- **Captions**: 5 captions per image = 150,000 total
- **Vocabulary**: ~18,000 unique words
- **Avg Caption Length**: 12.3 words
- **Domains**: People, activities, scenes

**Download**:
```bash
# Request access at: https://shannon.cs.illinois.edu/DenotationGraph/
# Download flickr30k-images.tar
tar -xf flickr30k-images.tar -C data/flickr30k/images/

# Download annotations
wget https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip
unzip annotations.zip -d data/flickr30k/
```

**Data Format**:
```json
{
  "image_id": "2489498453",
  "file_name": "2489498453.jpg",
  "captions": [
    "A young girl climbing a set of stairs in an entry way",
    "A little girl in pink climbing stairs",
    "Girl ascending the staircase",
    "A small child walks up stairs",
    "Young child going up the steps"
  ]
}
```

#### 3. Conceptual Captions 3M - Subset (200K samples)

**Description**: Web-crawled image-caption pairs

**Statistics**:
- **Images**: 200,000 images (subset of 3M)
- **Captions**: 1 caption per image
- **Vocabulary**: ~25,000 unique words
- **Avg Caption Length**: 8.7 words
- **Domains**: Diverse web images

**Download**:
```bash
# Download from Google
# https://ai.google.com/research/ConceptualCaptions/download

# Or use our preprocessed subset
wget https://your-server.com/cc3m_subset_200k.tar.gz
tar -xzf cc3m_subset_200k.tar.gz -C data/cc3m/
```

**Data Format**:
```json
{
  "image_url": "https://example.com/image.jpg",
  "caption": "beautiful sunset over mountains",
  "image_id": "cc3m_123456"
}
```

## 🎓 Stage 2 Datasets

### Instruction Tuning Datasets

#### 1. LLaVA-Instruct-150K

**Description**: GPT-4 generated instruction-following data

**Statistics**:
- **Samples**: 150,000 instruction-response pairs
- **Images**: From COCO dataset
- **Avg Instruction Length**: 15.2 words
- **Avg Response Length**: 45 words
- **Task Types**: Description, reasoning, conversation

**Download**:
```bash
# Download from LLaVA repository
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json
mkdir -p data/llava/
mv llava_instruct_150k.json data/llava/
```

**Data Format**:
```json
{
  "id": "conversation_001",
  "image": "COCO_train2017_000000123456.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is unusual about this image?"
    },
    {
      "from": "gpt",
      "value": "The unusual aspect of this image is that there is a person water skiing while being pulled by a horse on land near the water, which is not a typical water skiing setup. Normally, water skiing involves being pulled by a motorboat across the water's surface."
    }
  ]
}
```

#### 2. VQA-v2 (100K subset)

**Description**: Visual Question Answering v2.0

**Statistics**:
- **Samples**: 100,000 question-answer pairs
- **Images**: From COCO dataset
- **Question Types**: Yes/No (40%), Number (10%), Other (50%)
- **Avg Question Length**: 6.2 words
- **Avg Answer Length**: 1.5 words

**Download**:
```bash
# Download VQA v2.0
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

# Extract
unzip v2_Questions_Train_mscoco.zip -d data/vqa/
unzip v2_Annotations_Train_mscoco.zip -d data/vqa/
```

**Data Format**:
```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question": "What color is the umbrella?",
  "answer": "red",
  "answer_type": "color",
  "answers": [
    {"answer": "red", "confidence": "yes"},
    {"answer": "red", "confidence": "yes"},
    {"answer": "red", "confidence": "yes"}
  ]
}
```

#### 3. OK-VQA (50K subset)

**Description**: Outside Knowledge VQA - requires external knowledge

**Statistics**:
- **Samples**: 50,000 question-answer pairs
- **Knowledge Domains**: Science, history, culture, geography
- **Avg Question Length**: 7.8 words
- **Requires Reasoning**: 85% of samples

**Download**:
```bash
# Download OK-VQA
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip

# Extract
unzip mscoco_train2014_annotations.json.zip -d data/okvqa/
unzip OpenEnded_mscoco_train2014_questions.json.zip -d data/okvqa/
```

**Data Format**:
```json
{
  "question_id": 5255080,
  "image_id": 525508,
  "question": "What architectural style is this building?",
  "answers": [
    "gothic",
    "gothic architecture",
    "neo-gothic"
  ],
  "rationale": "The pointed arches and flying buttresses are characteristic of Gothic architecture"
}
```

## 🤖 Stage 3 Datasets

### Robot Fleet Selection Dataset

#### Robot Selection Dataset (1K → 10K augmented)

**Description**: Task-based robot selection with multimodal inputs

**Statistics**:
- **Base Samples**: 1,000 manually curated
- **Augmented Samples**: 10,000 (10x augmentation)
- **Robot Types**: 5 (Drone, Underwater, Humanoid, Wheeled, Legged)
- **Task Categories**: Inspection, manipulation, transportation, rescue
- **Input Modalities**: Image + text instruction OR text-only

**Directory Structure**:
```
robot-selection-dataset/
├── single_robot_selection.json      # 250 samples
├── multi_robot_selection_dataset.json  # 250 samples
├── images/
│   ├── warehouse/
│   ├── outdoor/
│   ├── underwater/
│   └── disaster/
└── augmented/
    ├── augmented_train.json         # 9,000 samples
    └── augmented_val.json           # 1,000 samples
```

**Data Format**:
```json
{
  "id": "robot_sel_001",
  "task": "Pick and place boxes from high shelves in warehouse",
  "image": "warehouse/scene_001.jpg",
  "environment": {
    "type": "indoor_warehouse",
    "constraints": ["height_3m", "heavy_loads"],
    "hazards": []
  },
  "robots": {
    "primary": "Humanoid",
    "secondary": ["Robot with Wheels"],
    "reasoning": "Humanoid robots have the reach and strength needed for high shelf access. Wheeled robots can assist with ground-level transportation."
  },
  "confidence": 0.95
}
```

**Augmentation Strategies**:
1. **Synonym Replacement**: Replace task keywords with synonyms
2. **Task Variation**: Modify task parameters (height, weight, distance)
3. **Environmental Changes**: Add/modify constraints
4. **Reasoning Paraphrase**: Rephrase reasoning steps

**Generation Script**:
```bash
python scripts/augment_robot_data.py \
    --input robot-selection-dataset/single_robot_selection.json \
    --output robot-selection-dataset/augmented/augmented_train.json \
    --augmentation_factor 10 \
    --strategies synonym,variation,environmental
```

## 🧠 Stage 4 Datasets

### Reasoning-Annotated Dataset

#### Teacher-Generated Reasoning (50K samples)

**Description**: Chain-of-thought reasoning for robot selection

**Statistics**:
- **Samples**: 50,000 reasoning-annotated examples
- **Generation Method**: Teacher model (DeepSeek-R1 style)
- **Avg Reasoning Steps**: 4.2 steps
- **Avg Reasoning Length**: 120 words
- **Validation Rate**: 92% pass logical consistency check

**Directory Structure**:
```
data/stage4/
├── reasoning_data.json           # Main reasoning dataset
├── reasoning_samples/
│   ├── high_quality/            # Top 10% by quality
│   ├── medium_quality/          # Middle 80%
│   └── low_quality/             # Bottom 10% (filtered)
└── validation/
    ├── consistency_scores.json  # Logical consistency
    └── human_validation.json    # Human-verified subset (500 samples)
```

**Data Format**:
```json
{
  "id": "reasoning_001",
  "image": "warehouse_complex.jpg",
  "instruction": "Which robot combination is best for sorting packages by size and weight?",
  "reasoning": {
    "steps": [
      {
        "step_num": 1,
        "type": "problem_analysis",
        "content": "Task Requirements Analysis:\n- Need to sort packages by TWO criteria: size AND weight\n- Requires sensors for both measurements\n- Must handle various package sizes (small to large)\n- Need manipulation capability for sorting",
        "key_insights": ["dual_criteria", "sensor_requirement", "manipulation_needed"]
      },
      {
        "step_num": 2,
        "type": "robot_evaluation",
        "content": "Robot Capability Assessment:\n- Humanoid: Good manipulation, can integrate sensors, moderate speed\n- Wheeled: Fast movement, limited manipulation, can mount sensors\n- Drone: Cannot handle ground packages effectively\n- Legged: Redundant with wheeled for this task",
        "capability_matrix": {
          "Humanoid": {"manipulation": 0.9, "sensing": 0.8, "speed": 0.6},
          "Wheeled": {"manipulation": 0.5, "sensing": 0.9, "speed": 0.9}
        }
      },
      {
        "step_num": 3,
        "type": "solution_selection",
        "content": "Optimal Strategy:\n- Wheeled robots with weight/size sensors for initial measurement and transport\n- Humanoid robots for physical sorting and placing\n- Coordination: Wheeled measures → Humanoid sorts → Wheeled transports to bins",
        "workflow": ["measure", "sort", "transport"]
      },
      {
        "step_num": 4,
        "type": "conclusion",
        "content": "Best combination: PRIMARY: Robot with Wheels (measurement & transport), SECONDARY: Humanoid (sorting manipulation)",
        "confidence_factors": ["efficiency", "accuracy", "coordination_feasibility"]
      }
    ],
    "consistency_score": 0.94,
    "completeness_score": 0.96,
    "logical_flow_score": 0.92
  },
  "answer": {
    "primary": "Robot with Wheels",
    "secondary": ["Humanoid"],
    "confidence": 0.93
  },
  "metadata": {
    "source": "teacher_generated",
    "teacher_model": "deepseek-r1",
    "validated": true,
    "complexity": "high"
  }
}
```

**Generation Script**:
```bash
python scripts/generate_reasoning_data.py \
    --teacher_model deepseek-r1 \
    --input_dataset data/stage3/robot_selection.json \
    --output_file data/stage4/reasoning_data.json \
    --num_samples 50000 \
    --validate_consistency \
    --min_quality_score 0.8
```

## 🛠️ Data Preparation

### Complete Setup Script

```bash
#!/bin/bash
# prepare_all_data.sh

echo "Preparing EmberVLM datasets..."

# Create directory structure
mkdir -p data/{coco,flickr30k,cc3m,llava,vqa,okvqa,stage3,stage4}

# Stage 1 data
echo "Downloading Stage 1 datasets..."
bash scripts/download_coco.sh
bash scripts/download_flickr30k.sh
bash scripts/download_cc3m_subset.sh

# Stage 2 data
echo "Downloading Stage 2 datasets..."
bash scripts/download_llava.sh
bash scripts/download_vqa.sh
bash scripts/download_okvqa.sh

# Stage 3 data (included in repo)
echo "Preparing Stage 3 robot selection data..."
python scripts/prepare_robot_data.py
python scripts/augment_robot_data.py --augmentation_factor 10

# Stage 4 data
echo "Generating Stage 4 reasoning data..."
python scripts/generate_reasoning_data.py \
    --teacher_model deepseek-r1 \
    --num_samples 50000 \
    --validate

echo "Data preparation complete!"
echo "Total disk usage:"
du -sh data/
```

### Quick Start (Minimal Dataset)

For quick experimentation, download minimal subsets:

```bash
# Minimal setup (~5 GB vs ~100 GB full)
python scripts/download_minimal_data.py \
    --stage1_samples 10000 \
    --stage2_samples 10000 \
    --stage3_samples 500 \
    --stage4_samples 5000 \
    --output_dir data/minimal/
```

## 📋 Data Format Specifications

### Unified JSON Format

All datasets are converted to a unified format for training:

```json
{
  "id": "unique_identifier",
  "stage": 1,  // 1, 2, 3, or 4
  "image": "path/to/image.jpg",  // Optional for text-only
  "instruction": "The input instruction or question",
  "response": "The expected output",
  "reasoning": "Step-by-step reasoning (Stage 4 only)",
  "metadata": {
    "source_dataset": "coco_captions",
    "task_type": "captioning",
    "difficulty": "medium",
    "modality": "image_text"
  }
}
```

### Image Preprocessing

Standard preprocessing for all images:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),                    # Resize shorter side to 256
    transforms.CenterCrop(224),                # Center crop to 224x224
    transforms.ToTensor(),                     # Convert to tensor [0, 1]
    transforms.Normalize(                      # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## 📥 Download Instructions

### Automatic Download

```bash
# Download all datasets automatically
python scripts/download_all_datasets.py \
    --stages 1,2,3,4 \
    --output_dir data/ \
    --parallel 4
```

### Manual Download

See individual dataset sections above for manual download links.

### Dataset Verification

```bash
# Verify all datasets are correctly downloaded
python scripts/verify_datasets.py \
    --data_dir data/ \
    --stages 1,2,3,4 \
    --check_integrity
```

**Expected Output**:
```
✓ Stage 1 datasets:
  - COCO: 100,000 samples OK
  - Flickr30k: 30,000 samples OK
  - CC3M: 200,000 samples OK

✓ Stage 2 datasets:
  - LLaVA: 150,000 samples OK
  - VQA-v2: 100,000 samples OK
  - OK-VQA: 50,000 samples OK

✓ Stage 3 datasets:
  - Robot Selection: 10,000 samples OK

✓ Stage 4 datasets:
  - Reasoning: 50,000 samples OK

All datasets verified successfully!
```

## 📊 Dataset Statistics Summary

| Dataset | Images | Samples | Size | Purpose |
|---------|--------|---------|------|---------|
| COCO | 100K | 500K | 25 GB | Captioning |
| Flickr30k | 30K | 150K | 8 GB | Captioning |
| CC3M Subset | 200K | 200K | 40 GB | Contrastive |
| LLaVA-Instruct | 100K | 150K | 12 GB | Instructions |
| VQA-v2 | 100K | 100K | - | Question answering |
| OK-VQA | 50K | 50K | - | Knowledge QA |
| Robot Selection | 500 | 10K | 2 GB | Robot selection |
| Reasoning | 500 | 50K | 5 GB | CoT reasoning |
| **Total** | **~500K unique** | **1.16M** | **~92 GB** | **All stages** |

## 🔗 Next Steps

- **[Stage 1 Training Guide](../training/STAGE1_GUIDE.md)**: Start training with these datasets
- **[Data Augmentation Strategies](AUGMENTATION_STRATEGIES.md)**: Learn about data augmentation
- **[Robot Selection Data Format](ROBOT_SELECTION_DATA.md)**: Detailed robot data specification

---

**Questions?** See our [FAQ](../FAQ.md) or open an [issue](https://github.com/yourusername/EmberVLM/issues).

