# ML0bag Vision System - Training Strategy Guide

## What Are We Building?

A **3-stage AI inspection system** that looks at a luxury bag and answers three questions in sequence:

1. **"What am I looking at?"** - Detect the bag and identify the brand (Gucci, LV, Chanel, etc.)
2. **"Is it damaged?"** - Find any cosmetic defects (scratches, tears, stains, wear)
3. **"Is it real?"** - Compare it against known authentic patterns to flag potential fakes

These three stages run as a **cascade pipeline**: the camera sees a bag, crops it out, checks for damage, then checks for authenticity - all in real time.

---

## The Three Training Phases Explained

### Phase 1: Detection - "Find the Bag and Read the Logo"

| Detail | Value |
|--------|-------|
| **Goal** | Locate bags in the frame, identify brand logos |
| **Model** | YOLOv11m (medium) - Object Detection |
| **Training data** | CVandDL Bag Logo dataset (1,920 images, 24 luxury brands) |
| **Image size** | 640x640 pixels |
| **What it learns** | Where the bag is in the image, which brand logo is visible |
| **Output** | Bounding boxes around bags + brand name + confidence % |

**How it works**: YOLO (You Only Look Once) divides the image into a grid and predicts bounding boxes and class probabilities simultaneously. We start from a model pre-trained on millions of general images (COCO dataset), then fine-tune it on luxury bag logos. This is called **transfer learning** - the model already knows what edges, shapes, and textures look like, we just teach it "this specific pattern = Gucci logo."

**24 brands it recognizes**: Balenciaga, Burberry, Celine, Chanel, Dior, Fendi, Gucci, Hermes, Louis Vuitton, Loewe, Goyard, MCM, Michael Kors, Prada, Saint Laurent, Valentino, Versace, Bottega Veneta, Chloe, Coach, Givenchy, Miu Miu, Mulberry, Salvatore Ferragamo

---

### Phase 2: Condition Assessment - "Find the Damage"

| Detail | Value |
|--------|-------|
| **Goal** | Detect and classify cosmetic defects |
| **Model** | YOLOv11m (medium) - Object Detection |
| **Training data** | Kaputt (238K images, filtered) + Kaggle Leather Defects |
| **Image size** | 640x640 pixels |
| **What it learns** | What scratches, tears, stains, wear, and holes look like |
| **Output** | Bounding boxes around each defect + defect type + severity confidence |

**7 defect classes:**

| Class | What It Detects | Examples |
|-------|----------------|----------|
| scratch | Surface scratches on leather/hardware | Key marks, nail scratches, abrasion lines |
| tear | Rips or cuts in material | Torn stitching, cut leather, split seams |
| stain | Discoloration from liquids/dirt | Water marks, ink stains, grease spots |
| wear | General aging and use damage | Faded corners, worn edges, patina loss |
| deformation | Shape distortion | Crushed structure, bent hardware, warping |
| hole | Punctures or perforations | Moth damage, punctures, burn holes |
| discoloration | Color changes or fading | Sun fading, dye transfer, oxidation |

**How it works**: Phase 1 detects the bag and crops it from the image. This cropped region is then fed to Phase 2, which looks specifically for defects within the bag area. This two-stage approach is more accurate than trying to find tiny scratches in a full-frame image because the model gets a close-up view of just the bag.

**Data sources**:
- **Kaputt** (Amazon Science, ICCV 2025): 238,000 images of damaged products. We filter for categories relevant to bags: penetration (tears/holes), superficial (scratches/dirt), and deformation. The model learns what "damage" looks like in general.
- **Kaggle Leather Defects**: Specifically leather damage - holes, scratches, dirt, rotten surfaces, folds. Supplements Kaputt with leather-specific knowledge.

---

### Phase 3: Authentication - "Real or Fake?"

| Detail | Value |
|--------|-------|
| **Goal** | Classify bag as authentic or counterfeit |
| **Model** | YOLOv11s-cls (small, classification) |
| **Training data** | LFFD dataset (12,379 Chanel images) + your own curated photos |
| **Image size** | 224x224 pixels |
| **What it learns** | Subtle differences between real and fake: stitching patterns, material texture, hardware finish, logo precision |
| **Output** | "Authentic" or "Counterfeit" + confidence % |

**How it works**: Unlike Phases 1 and 2 (which find objects), Phase 3 is a **classifier** - it looks at the whole cropped bag image and assigns a probability. The model learns a "fingerprint" of what authentic bags look like: the regularity of stitching, the exact shade of hardware, the texture of leather grain. Fakes, even good ones, have subtle pattern differences that the AI can pick up.

**Important limitation**: This is the hardest phase. The LFFD dataset has 12,379 Chanel images but they're labeled by product category, NOT as real/fake. You will need to manually sort some as "counterfeit" before this phase works well. The more real photos of YOUR actual inventory you add, the better this phase gets.

---

## What Model Are We Using?

**YOLOv11 by Ultralytics** (released 2024)

| Feature | Detail |
|---------|--------|
| Full name | You Only Look Once, version 11 |
| Developer | Ultralytics |
| Architecture | CSPDarknet backbone + C3k2 neck + detection head |
| Pre-trained on | COCO dataset (330K images, 80 object classes) |
| Why this model | Fastest accurate detector available, runs in real-time, proven in production |

**Model sizes we use:**

| Phase | Model | Parameters | Speed (RTX 5090) | Why This Size |
|-------|-------|-----------|------------------|---------------|
| Phase 1 | yolo11m.pt | ~20M | ~5ms/image | Medium - good for logo detail |
| Phase 2 | yolo11m.pt | ~20M | ~5ms/image | Medium - defects need detail |
| Phase 3 | yolo11s-cls.pt | ~9M | ~2ms/image | Small classifier is enough for binary real/fake |

---

## How the Cascade Works in Real-Time

```
Camera Frame (1280x720)
       |
       v
 [Phase 1: Detection]
 "I see a Gucci bag at position (x,y)"
       |
       v
 [Crop the bag region + 10% padding]
       |
       +--------+---------+
       |                   |
       v                   v
 [Phase 2: Condition]  [Phase 3: Auth]
 "Scratch found on      "92% authentic"
  bottom left corner"
       |                   |
       +--------+---------+
                |
                v
      [Display results on screen]
      Brand: GUCCI (97%)
      Condition: scratch (84%)
      Auth: Authentic (92%)
```

---

## Adding Your Own Bag Photos

This is the most important part for your business. The public datasets give the model general knowledge, but **YOUR photos of YOUR inventory** are what make it accurate for your specific use case.

### Photo Requirements

| Requirement | Specification | Why |
|------------|--------------|-----|
| **Resolution** | Minimum 1280x720 (HD), recommended 1920x1080 or higher | The model resizes to 640x640 internally, but starting with higher resolution preserves detail |
| **Format** | JPG or PNG | JPG for photos (smaller files), PNG for screenshots or edited images |
| **File size** | No limit, but under 10MB per image is practical | Larger files slow down loading but don't improve training |
| **Color** | Full color (RGB) | The model uses color information for leather texture and hardware analysis |
| **Aspect ratio** | Any - the model handles it | Images get resized with padding (letterboxing) to maintain proportions |

### How to Photograph Your Bags

For the best results, photograph each bag with this **shot list**:

| Shot # | What to Capture | Camera Distance | Purpose |
|--------|----------------|-----------------|---------|
| 1 | **Full front view** | 1-2 feet away | Overall detection, brand ID |
| 2 | **Full back view** | 1-2 feet away | Check back panel condition |
| 3 | **Logo close-up** | 6-12 inches | Authentication detail |
| 4 | **Hardware close-up** (zipper, clasp, chain) | 6-12 inches | Authentication + defect detection |
| 5 | **Interior shot** | 6-12 inches | Serial number, lining condition |
| 6 | **Bottom/base** | 6-12 inches | Wear detection (most common damage area) |
| 7 | **Corners close-up** | 6-12 inches | Wear detection (second most common) |
| 8 | **Stitching close-up** | 3-6 inches | Authentication (stitch regularity) |
| 9 | **Any visible defects** | As close as needed | Condition training data |

**Lighting tips:**
- Use natural daylight or a ring light (even, diffused light)
- Avoid harsh shadows or flash glare on hardware
- Place the bag on a clean, plain background (white or light gray)
- Avoid busy backgrounds - they confuse the detector

### Where to Put Your Photos

Your photos go into different folders depending on what they're training:

```
data/
  custom/                          <-- CREATE THIS FOLDER
    phase1_detection/              <-- Full bag photos with logos visible
      images/
        gucci_bag_001.jpg
        lv_neverfull_002.jpg
        ...
      labels/                      <-- You'll create these (see annotation below)
        gucci_bag_001.txt
        lv_neverfull_002.txt

    phase2_defects/                <-- Close-up photos of defects you find
      images/
        scratch_on_corner_001.jpg
        stain_interior_002.jpg
        ...
      labels/
        scratch_on_corner_001.txt
        stain_interior_002.txt

    phase3_authentication/         <-- Full bag photos sorted by authenticity
      authentic/                   <-- Bags YOU have verified as real
        chanel_classic_real_001.jpg
        gucci_marmont_real_002.jpg
      counterfeit/                 <-- Bags YOU have identified as fake
        fake_chanel_001.jpg
        fake_lv_002.jpg
```

### How to Annotate Your Photos (Phases 1 & 2)

For Phases 1 and 2, the model needs to know WHERE in the image the bag/defect is. This is done with **bounding box annotations** in YOLO format.

**Use a free annotation tool:**
- **[Roboflow](https://roboflow.com)** (recommended - free tier, web-based, exports YOLO format directly)
- **[CVAT](https://cvat.ai)** (free, open source, more powerful)
- **[Label Studio](https://labelstud.io)** (free, open source)

**YOLO label format** (one .txt file per image):
```
# Each line: class_index center_x center_y width height
# All values are normalized (0.0 to 1.0, relative to image size)
# Example: a Gucci logo at the center of the image, covering 30% width and 20% height
7 0.5 0.5 0.3 0.2
```

**For Phase 3 (authentication)**: No annotation needed - just sort photos into `authentic/` and `counterfeit/` folders. The folder name IS the label.

### How to Retrain After Adding Your Photos

```bash
# 1. Preprocess your new data (merges with existing training data)
python -m src.data.preprocess all --force

# 2. Retrain the phase you added data for
python -m src.models.trainer train phase1          # If you added logo photos
python -m src.models.trainer train phase2          # If you added defect photos
python -m src.models.trainer train phase3          # If you added auth photos

# 3. Or retrain everything
python -m src.models.trainer train-all
```

### Image Size - What Happens Under the Hood

You do NOT need to resize your images manually. Here's what happens automatically:

```
Your photo (4032x3024 from iPhone)
       |
       v
  [Ultralytics auto-resize]
  Scales down to fit 640x640 (Phase 1 & 2) or 224x224 (Phase 3)
  Maintains aspect ratio using letterboxing (black padding)
       |
       v
  [Data augmentation during training]
  Random flips, rotations, color shifts, mosaic (combining 4 images)
  This artificially creates more training variety from your photos
       |
       v
  [Model training]
  The model learns from the 640x640 version
```

**Rule of thumb**: Take the highest resolution photos you can. The system handles the rest. Minimum 1280x720, but 4K phone photos are great.

### How Many Photos Do You Need?

| Phase | Minimum to Start | Good | Excellent |
|-------|-----------------|------|-----------|
| Phase 1 (detection) | 50 per brand you sell | 200 per brand | 500+ per brand |
| Phase 2 (defects) | 100 per defect type | 500 per defect type | 1000+ per defect type |
| Phase 3 (auth) | 200 authentic + 200 fake | 500 + 500 | 1000+ each |

**Start small, improve over time.** Even 50 photos of your own bags mixed with the public dataset will noticeably improve accuracy for your specific inventory.

### Continuous Improvement Cycle

This is how you keep the model getting better over time:

```
1. Receive new bag inventory
          |
          v
2. Photograph each bag (shot list above)
          |
          v
3. Run inference to get AI prediction
          |
          v
4. Expert reviews the AI prediction
          |
    +-----+-----+
    |             |
    v             v
  CORRECT       WRONG
    |             |
    v             v
  Save photo    Save photo + correct label
  as verified   (this becomes new training data)
    |             |
    +-----+-----+
          |
          v
5. After collecting 50+ new corrections,
   retrain the relevant phase
          |
          v
6. Model improves, fewer errors next time
```

This is called **active learning** - the model's mistakes become its best training data. Every bag your team processes makes the AI smarter.

---

## Data You're Working With - Summary

| Dataset | Size | Source | Used For | Phase |
|---------|------|--------|----------|-------|
| CVandDL Bag Logos | 1,920 images | Roboflow | Brand detection | 1 |
| Kaputt | 238,000 images | Amazon/ICCV | Defect detection | 2 |
| Leather Defects | ~1,000 images | Kaggle | Leather-specific damage | 2 |
| LFFD (Innovatiana) | 12,379 images | HuggingFace | Authentication | 3 |
| **Your own photos** | Grows over time | Your inventory | All phases | 1, 2, 3 |

---

## Glossary

| Term | Meaning |
|------|---------|
| **YOLO** | You Only Look Once - the detection model architecture. Fast and accurate. |
| **Bounding box** | Rectangle drawn around an object in an image (x, y, width, height) |
| **Confidence score** | How sure the model is (0% = no idea, 100% = certain) |
| **Epoch** | One complete pass through all training images |
| **Fine-tuning** | Starting from a pre-trained model and training further on your data |
| **Transfer learning** | Using knowledge from one task (general objects) to help with another (luxury bags) |
| **mAP** | Mean Average Precision - the standard accuracy metric for object detection |
| **Augmentation** | Artificially creating more training images by flipping, rotating, changing colors |
| **Cascade** | Running models in sequence, each one refining the previous result |
| **ONNX** | Open Neural Network Exchange - format for deploying models in production |
| **Inference** | Running the trained model on new images to get predictions |
| **Letterboxing** | Adding black padding to maintain aspect ratio when resizing |
