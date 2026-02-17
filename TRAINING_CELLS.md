# Training Cells - Copy/Paste into Notebook

These cells fix the CUDA pin memory error and include Phase 1 optimizations. Copy each one directly into the corresponding notebook cell.

## Phase 1 Training Cell (Optimized)

```python
# Train Phase 1 - Optimized for small dataset
import torch
torch.cuda.empty_cache()

# Reload config to pick up latest changes
config = load_config('phase1_detection')
model = YOLO(config['model']['architecture'])
data_path = str(PROJECT_ROOT / config['data']['config'])

results = model.train(
    data=data_path,
    epochs=config['training']['epochs'],
    imgsz=config['data']['imgsz'],
    batch=8,
    device=device,
    project=str(PROJECT_ROOT / config['output']['project']),
    name=config['output']['name'],
    patience=config['training']['patience'],
    optimizer=config['training']['optimizer'],
    lr0=config['training']['lr0'],
    cos_lr=config['training']['cos_lr'],
    save_period=config['output']['save_period'],
    weight_decay=config['training'].get('weight_decay', 0.0005),
    warmup_epochs=config['training'].get('warmup_epochs', 5),
    freeze=config['training'].get('freeze', 0),
    plots=True,
    verbose=True,
    workers=0,
    **config.get('augmentation', {}),
)
```

## Phase 2 Training Cell

```python
# Load phase 2 config
import torch
torch.cuda.empty_cache()

config2 = load_config('phase2_condition')

model2 = YOLO(config2['model']['architecture'])
data_path2 = str(PROJECT_ROOT / config2['data']['config'])

results2 = model2.train(
    data=data_path2,
    epochs=config2['training']['epochs'],
    imgsz=config2['data']['imgsz'],
    batch=8,
    device=device,
    project=str(PROJECT_ROOT / config2['output']['project']),
    name=config2['output']['name'],
    patience=config2['training']['patience'],
    optimizer=config2['training']['optimizer'],
    lr0=config2['training']['lr0'],
    cos_lr=config2['training']['cos_lr'],
    save_period=config2['output']['save_period'],
    plots=True,
    verbose=True,
    workers=0,
    **config2.get('augmentation', {}),
)
```

## Phase 3 Training Cell

```python
# Train Phase 3 (only if data is ready)
import torch
torch.cuda.empty_cache()

if PHASE3_READY:
    config3 = load_config('phase3_authentication')

    model3 = YOLO(config3['model']['architecture'])
    data_path3 = str(PROJECT_ROOT / config3['data']['config'])

    results3 = model3.train(
        data=data_path3,
        epochs=config3['training']['epochs'],
        imgsz=config3['data']['imgsz'],
        batch=8,
        device=device,
        project=str(PROJECT_ROOT / config3['output']['project']),
        name=config3['output']['name'],
        patience=config3['training']['patience'],
        optimizer=config3['training']['optimizer'],
        lr0=config3['training']['lr0'],
        cos_lr=config3['training']['cos_lr'],
        save_period=config3['output']['save_period'],
        plots=True,
        verbose=True,
        workers=0,
        **config3.get('augmentation', {}),
    )

    # Save weights
    best_src3 = Path(results3.save_dir) / 'weights' / 'best.pt'
    best_dst3 = MODELS_DIR / 'phase3' / 'best.pt'
    best_dst3.parent.mkdir(parents=True, exist_ok=True)
    if best_src3.exists():
        shutil.copy2(best_src3, best_dst3)
        print(f"Phase 3 best model saved to: {best_dst3}")
else:
    print("Skipping Phase 3 training - not enough counterfeit samples.")
    print("Complete the annotation step above first.")
```

## What changed from original

- `batch=8` (was 16) - reduces GPU memory usage
- `workers=0` (was 8) - prevents CUDA pin memory thread crash
- `torch.cuda.empty_cache()` - clears leftover GPU memory before training

## Phase 1 optimizations (v2)

- `yolo11n.pt` (was yolo11m.pt) - 2.6M params vs 20M, better for 55-image dataset
- `lr0=0.0001` (was 0.001) - lower learning rate for transfer learning
- `freeze=10` - freezes backbone layers, trains only detection head
- `weight_decay=0.001` (was 0.0005) - stronger regularization
- `scale=0.9, shear=5.0` - more aggressive augmentation
- `copy_paste=0.2, erasing=0.3, perspective=0.001` - additional augmentation
- `epochs=100` (was 150), `patience=20` (was 30) - faster convergence expected
