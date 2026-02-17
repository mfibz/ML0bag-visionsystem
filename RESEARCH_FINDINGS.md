# ML0bag Vision System - Research Findings

## 1. Dataset Validation Report

### 1.1 LFFD Dataset (Innovatiana) - VERIFIED

- **Source**: [Innovatiana/innv-luxury-fashion-dataset-fraud-detection](https://huggingface.co/datasets/Innovatiana/innv-luxury-fashion-dataset-fraud-detection)
- **Status**: Available and downloadable on HuggingFace
- **Size**: 2.09 GB, 12,379 images
- **Format**: JPG images with ClassLabel annotations (357 product categories)
- **Structure**: Folder-based organization by product category (Chanel products)
- **Split**: Single `train` split
- **License**: Research/educational use only; contact Innovatiana for commercial use
- **Limitations**:
  - Only covers Chanel brand products
  - Classification labels (not bounding box annotations) -- will need conversion/re-annotation for object detection
  - No explicit authentic vs. counterfeit binary labels; categories represent product types
  - Would require additional annotation work to create fraud/authenticity labels suitable for YOLO training
- **Usefulness**: Medium. Good source of luxury product imagery but requires significant preprocessing for our detection pipeline. Best used as a classification/pretraining dataset or for generating synthetic authentic/counterfeit pairs.

### 1.2 Kaputt Dataset - VERIFIED

- **Source**: [kaputt-dataset.com](https://www.kaputt-dataset.com/)
- **Status**: Available. Published at ICCV 2025 by Amazon Science.
- **Size**: 238,421 high-resolution images, 48,376 unique items, 29,316 defective instances
- **Defect Categories** (7 types):
  1. Penetration (holes, tears, cuts)
  2. Deformation (dents, crushes)
  3. Actuation (open box/bag/book)
  4. Deconstruction
  5. Spillage (liquid, powder)
  6. Superficial (dirt, scratches)
  7. Missing unit
- **Severity Levels**: Minor and Major
- **Annotation Format**: Categorical labels + segmentation masks (U-Net generated). Includes reference images (1-3 per item) for comparison.
- **Limitations**:
  - Focused on retail logistics (general consumer products), not specifically luxury bags
  - Defect categories are general-purpose; may need filtering for leather/fabric-relevant defects (penetration, deformation, superficial)
  - Large dataset size may require selective downloading
- **Usefulness**: High. The "superficial" (scratches, dirt) and "penetration" (tears, cuts) categories are directly relevant to bag condition assessment. Reference image comparison approach aligns well with authentication workflows.

### 1.3 Roboflow - Bag Authenticity Detection by YING - NOT FOUND

- **Status**: Not found on Roboflow Universe
- **Notes**: No dataset matching "bag authenticity detection" by user "YING" was found. The search returned no results for this specific dataset.

### 1.4 Roboflow - Brand Logo Recognition by DATA6000 - VERIFIED

- **Source**: [Brand Logo recognition - YOLOv8 by DATA6000](https://universe.roboflow.com/data6000/brand-logo-recognition-yolov8)
- **Status**: Available on Roboflow Universe
- **Size**: 503 images
- **Classes**: 5 classes (Coke, Heineken, Mc, Pepsi, Starbucks)
- **License**: CC BY 4.0
- **Limitations**:
  - Only 503 images -- quite small
  - Classes are beverage brands, NOT luxury fashion brands
  - Not directly useful for luxury bag logo authentication
- **Usefulness**: Low. Wrong domain entirely (beverages, not luxury fashion).

### 1.5 Alternative/Supplementary Datasets Found

#### A. Detection Bag Logo by CVandDL (Roboflow) - RECOMMENDED ALTERNATIVE
- **Source**: [Detection bag logo by CVandDL](https://universe.roboflow.com/cvanddl/detection-bag-logo)
- **Size**: 1,920 images with pre-trained model
- **Classes**: 24 luxury brand classes including BALENCIAGA, BURBERRY, CELINE, DIOR, FENDI, GUCCI, LOUIS-VUITTON, PRADA, LOEWE, GOYARD, MCM, MICHAEL-KORS, etc.
- **Use Cases**: Brand counterfeit detection, retail inventory, e-commerce tagging
- **Usefulness**: HIGH. Directly relevant -- luxury bag brand logos with object detection annotations.

#### B. Leather Defect Classification (Kaggle)
- **Source**: [Kaggle - Leather Defect Classification](https://www.kaggle.com/datasets/praveen2084/leather-defect-classification)
- **Categories**: Holes, scratches, dirt, rotten surfaces, folds
- **Usefulness**: Medium. Small but directly relevant to leather condition assessment.

#### C. Fabric Defect Datasets (Multiple Sources)
- Kaggle fabric defect datasets, Mendeley fabric defects, FabricSpotDefect
- **Usefulness**: Low-Medium. More relevant to textile manufacturing than luxury bag inspection.

---

## 2. Recommended Project Directory Structure

```
/workspaces/ML0bag-visionsystem/
|
|-- README.md                          # Project overview and setup instructions
|-- RESEARCH_FINDINGS.md               # This document
|-- requirements.txt                   # Python dependencies
|-- setup.py                           # Package setup (optional)
|-- .env.example                       # Environment variable template
|
|-- configs/                           # Training and pipeline configurations
|   |-- phase1_detection.yaml          # Phase 1: Bag detection + brand logo detection
|   |-- phase2_condition.yaml          # Phase 2: Defect/condition assessment
|   |-- phase3_authentication.yaml     # Phase 3: Authenticity classification
|   |-- data_sources.yaml              # Dataset URLs and credentials
|   |-- inference.yaml                 # Inference pipeline config
|
|-- data/                              # Data directory (gitignored)
|   |-- raw/                           # Raw downloaded datasets
|   |   |-- lffd/                      # Innovatiana LFFD dataset
|   |   |-- kaputt/                    # Kaputt dataset
|   |   |-- roboflow_logos/            # CVandDL bag logo dataset
|   |   |-- leather_defects/           # Kaggle leather defects
|   |-- processed/                     # Preprocessed and unified datasets
|   |   |-- phase1/                    # YOLO-format data for Phase 1
|   |   |   |-- images/
|   |   |   |   |-- train/
|   |   |   |   |-- val/
|   |   |   |   |-- test/
|   |   |   |-- labels/
|   |   |       |-- train/
|   |   |       |-- val/
|   |   |       |-- test/
|   |   |-- phase2/                    # YOLO-format data for Phase 2
|   |   |   |-- images/
|   |   |   |-- labels/
|   |   |-- phase3/                    # YOLO-format data for Phase 3
|   |       |-- images/
|   |       |-- labels/
|   |-- data.yaml                      # Combined YOLO data config
|
|-- src/                               # Source code
|   |-- __init__.py
|   |-- data/                          # Data ingestion and preprocessing
|   |   |-- __init__.py
|   |   |-- download.py                # Dataset download scripts
|   |   |-- preprocess.py              # Format conversion, resizing, augmentation
|   |   |-- split.py                   # Train/val/test splitting
|   |   |-- convert_lffd.py            # LFFD-specific conversion to YOLO format
|   |   |-- convert_kaputt.py          # Kaputt-specific conversion to YOLO format
|   |   |-- convert_roboflow.py        # Roboflow dataset conversion
|   |
|   |-- training/                      # Training pipeline
|   |   |-- __init__.py
|   |   |-- train_phase1.py            # Phase 1: Detection training
|   |   |-- train_phase2.py            # Phase 2: Condition training
|   |   |-- train_phase3.py            # Phase 3: Authentication training
|   |   |-- train_pipeline.py          # End-to-end multi-phase training orchestrator
|   |
|   |-- inference/                     # Inference pipeline
|   |   |-- __init__.py
|   |   |-- detector.py                # Single-image detection
|   |   |-- camera.py                  # Real-time camera/video inference
|   |   |-- pipeline.py                # Multi-phase inference orchestrator
|   |   |-- visualize.py               # Result visualization and overlay
|   |
|   |-- evaluation/                    # Evaluation and metrics
|   |   |-- __init__.py
|   |   |-- metrics.py                 # mAP, precision, recall, confusion matrix
|   |   |-- benchmark.py               # Speed/latency benchmarking
|   |   |-- report.py                  # Generate evaluation reports
|   |
|   |-- utils/                         # Shared utilities
|       |-- __init__.py
|       |-- config.py                  # Configuration loading
|       |-- logger.py                  # Logging setup
|       |-- constants.py               # Class names, paths, thresholds
|
|-- models/                            # Saved model weights (gitignored)
|   |-- phase1/
|   |-- phase2/
|   |-- phase3/
|
|-- runs/                              # Training runs and logs (gitignored)
|
|-- tests/                             # Unit and integration tests
|   |-- test_data_pipeline.py
|   |-- test_inference.py
|   |-- test_training.py
|
|-- notebooks/                         # Jupyter notebooks for exploration
|   |-- 01_data_exploration.ipynb
|   |-- 02_training_analysis.ipynb
|   |-- 03_inference_demo.ipynb
|
|-- scripts/                           # CLI scripts
    |-- download_datasets.sh
    |-- train_all.sh
    |-- run_inference.sh
```

---

## 3. Technical Recommendations

### 3.1 Model Architecture

**YOLO11 by Ultralytics** (latest stable release)

| Phase | Task | Recommended Model Size | Rationale |
|-------|------|----------------------|-----------|
| Phase 1 | Bag Detection + Brand Logo | YOLO11m (medium) | Good balance of accuracy and speed for logo detection; 20M params |
| Phase 2 | Condition/Defect Assessment | YOLO11m or YOLO11l | Defects can be subtle; larger model helps. Consider YOLO11-seg for segmentation masks |
| Phase 3 | Authenticity Classification | YOLO11s-cls | Classification task; smaller model sufficient |

### 3.2 Multi-Phase Training Strategy

1. **Phase 1 - Detection** (Object Detection)
   - Detect bags in frame + identify brand logos
   - Primary datasets: CVandDL bag logo dataset (1,920 images) + LFFD imagery (repurposed)
   - Classes: bag, + 24 luxury brand logo classes
   - Start from YOLO11m pretrained on COCO
   - Training: 100-200 epochs, imgsz=640, batch=16

2. **Phase 2 - Condition Assessment** (Object Detection / Segmentation)
   - Detect defects: scratches, tears, stains, wear, deformation
   - Primary datasets: Kaputt (filtered for relevant defects) + Kaggle leather defects
   - Use YOLO11m-seg if segmentation masks needed; otherwise YOLO11m for bounding boxes
   - Fine-tune from Phase 1 weights or COCO pretrained
   - Training: 150-300 epochs, imgsz=640, batch=16

3. **Phase 3 - Authentication** (Classification)
   - Binary or multi-class: authentic / suspicious / counterfeit
   - Dataset: LFFD (with manual labeling of authentic vs. counterfeit)
   - YOLO11s-cls or fine-tuned classifier head
   - Training: 50-100 epochs, imgsz=224, batch=32

### 3.3 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x NVIDIA T4 (16GB) | 1x NVIDIA A100 (40GB) |
| GPU VRAM | 8 GB | 16-40 GB |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 200 GB (for all datasets) |
| Training Time (per phase) | 4-12 hours (T4) | 1-4 hours (A100) |

### 3.4 Key Dependencies

```
ultralytics>=8.3.0        # YOLO11 support
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
huggingface-hub>=0.20.0
roboflow>=1.1.0
Pillow>=10.0.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
```

---

## 4. Risks and Gaps

### Critical Gaps

1. **No ready-made bag authenticity dataset**: The "bag authenticity detection by YING" dataset was not found on Roboflow. There is no single public dataset with authentic vs. counterfeit luxury bag labels with bounding box annotations. Manual annotation or synthetic data generation will be necessary for Phase 3.

2. **LFFD lacks detection annotations**: The Innovatiana dataset has classification labels (product categories) but no bounding box annotations. Converting it to YOLO object detection format requires manual or semi-automated re-annotation.

3. **Brand logo dataset is small**: The CVandDL bag logo dataset has only 1,920 images. Data augmentation and potentially additional logo data collection will be critical for robust Phase 1 performance.

### Moderate Risks

4. **Kaputt domain mismatch**: The Kaputt dataset covers general retail products, not specifically luxury bags. Defect detection models trained on it may need fine-tuning on bag-specific defect images.

5. **Cross-domain transfer**: Training on mixed datasets from different domains (retail logistics, fashion, leather manufacturing) may introduce distribution shift. Careful validation split design is needed.

6. **Authentication subjectivity**: Distinguishing authentic from counterfeit bags is inherently difficult and subjective. Even expert authenticators disagree on borderline cases. Model confidence calibration will be important.

### Mitigations

- Use the CVandDL bag logo dataset as the primary Phase 1 dataset (best match)
- Filter Kaputt to relevant defect types (penetration, superficial, deformation) for Phase 2
- Plan for a manual annotation sprint for Phase 3 authenticity labels
- Aggressive data augmentation (mosaic, mixup, color jitter, geometric transforms)
- Consider active learning: train initial model, use it to pre-annotate, then manually correct
- Start with YOLO11m pretrained on COCO to leverage transfer learning

---

## 5. Summary of Dataset Decisions

| Dataset | Status | Phase | Action Required |
|---------|--------|-------|-----------------|
| Innovatiana LFFD | Verified | 3 (Auth) | Download, re-annotate for authenticity labels |
| Kaputt | Verified | 2 (Condition) | Download, filter relevant defect categories |
| Bag Auth by YING | NOT FOUND | - | Replace with CVandDL bag logo dataset |
| Brand Logo DATA6000 | Verified (wrong domain) | - | Not useful (beverage brands only) |
| CVandDL Bag Logo | Verified | 1 (Detection) | Primary dataset for brand logo detection |
| Kaggle Leather Defects | Verified | 2 (Condition) | Supplement for leather-specific defects |
