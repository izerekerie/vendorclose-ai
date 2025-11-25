# Speed Optimization Guide - Reduce Training Time

## Current Problem
- **18,901 training samples** with batch size 32 = **591 steps per epoch**
- **3 seconds per step** = **~30 minutes per epoch**
- With 20 epochs = **~10 hours total!**

## Quick Fixes (Apply These Changes)

### Option 1: Increase Batch Size (FASTEST FIX)
**In Cell 25 (Data Preprocessing), change:**
```python
# OLD:
preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=32)

# NEW (2x faster):
preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=64)

# OR EVEN FASTER (4x faster):
preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=128)
```

**Result:**
- Batch 64: 296 steps/epoch = **~15 min/epoch** (2x faster)
- Batch 128: 148 steps/epoch = **~7.5 min/epoch** (4x faster)

### Option 2: Reduce Image Size (GOOD BALANCE)
**In Cell 25, change:**
```python
# OLD:
preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=32)

# NEW:
preprocessor = ImagePreprocessor(img_size=(160, 160), batch_size=64)
```

**Result:** ~3-4x faster overall

### Option 3: Use Data Subset (FOR TESTING ONLY)
**Add this before training (Cell 28):**
```python
# Use only 50% of data for faster training (testing only!)
USE_SUBSET = True
if USE_SUBSET:
    # Limit steps per epoch
    steps_per_epoch = train_gen.samples // (batch_size * 2)  # Half the steps
    validation_steps = val_gen.samples // (batch_size * 2)
else:
    steps_per_epoch = None
    validation_steps = None
```

**Then in training, add:**
```python
history = classifier.train(
    train_generator=train_gen,
    val_generator=val_gen,
    epochs=recommended_epochs,
    batch_size=32,
    steps_per_epoch=steps_per_epoch,  # Add this
    validation_steps=validation_steps  # Add this
)
```

## Recommended Settings

### For Fast Training (Testing):
```python
img_size=(160, 160)
batch_size=128
MAX_EPOCHS=10
```

### For Balanced (Recommended):
```python
img_size=(192, 192)
batch_size=64
MAX_EPOCHS=15
```

### For Best Quality (Slower):
```python
img_size=(224, 224)
batch_size=32
MAX_EPOCHS=20
```

## Time Estimates

| Config | Steps/Epoch | Time/Epoch | Total (15 epochs) |
|--------|-------------|------------|-------------------|
| Current (224, 32) | 591 | ~30 min | ~7.5 hours |
| Fast (160, 128) | 148 | ~4 min | ~1 hour |
| Balanced (192, 64) | 296 | ~8 min | ~2 hours |

## What to Change Right Now

**Stop current training (Ctrl+C) and modify Cell 25:**

```python
# OPTIMIZED FOR SPEED
preprocessor = ImagePreprocessor(img_size=(160, 160), batch_size=128)
```

**And Cell 28 (training), change:**
```python
recommended_epochs = 10  # Reduced from 25
```

This will make training **~8x faster**!

