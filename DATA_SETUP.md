# 📥 Data Setup Guide

## ⚠️ Important: Manual Download Required

**The notebook does NOT automatically download data.** You need to download it manually first.

---

## 🎯 Two Ways to Get Data

### Option 1: Manual Download (Easiest - No API needed)

1. **Go to Kaggle:**
   - https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

2. **Download:**
   - Click "Download" button
   - Sign in if needed (free account)
   - Download the ZIP file

3. **Extract:**
   - Extract the ZIP file
   - You'll see folders like `train`, `test`, etc.

4. **Organize:**
   ```
   Your Project/
   └── data/
       ├── train/
       │   ├── fresh/     ← Put fresh fruit images here
       │   └── rotten/    ← Put rotten fruit images here
       └── test/
           ├── fresh/
           └── rotten/
   ```

5. **That's it!** The notebook will automatically:
   - Create the "medium" class
   - Preprocess the data
   - Train the model

---

### Option 2: Automatic Download (Requires Kaggle API)

1. **Install Kaggle:**
   ```bash
   pip install kaggle
   ```

2. **Get API Token:**
   - Go to: https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Download `kaggle.json`

3. **Setup Token:**
   
   **Windows:**
   ```bash
   # Create folder
   mkdir C:\Users\%USERNAME%\.kaggle
   
   # Move kaggle.json there
   move kaggle.json C:\Users\%USERNAME%\.kaggle\
   ```
   
   **Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Run Download Script:**
   ```bash
   python download_dataset.py
   ```

---

## ✅ Verify Data is Ready

After downloading, check:

```bash
python download_dataset.py
```

Or manually check:
```bash
# Windows
dir data\train\fresh
dir data\train\rotten

# Linux/Mac
ls data/train/fresh
ls data/train/rotten
```

You should see image files (.jpg or .png)

---

## 🚀 Then Run Notebook

Once data is in place:

```bash
jupyter notebook notebook/vendorclose_ai.ipynb
```

The notebook will:
1. ✅ Check if data exists
2. ✅ Create "medium" class automatically
3. ✅ Preprocess data
4. ✅ Train model

---

## 📋 Quick Checklist

- [ ] Downloaded dataset from Kaggle
- [ ] Extracted ZIP file
- [ ] Organized into `data/train/fresh/` and `data/train/rotten/`
- [ ] Verified images are in folders
- [ ] Ready to run notebook!

---

## 🆘 Troubleshooting

**"Training data directory not found"**
- Make sure you created `data/train/` folder
- Put images in `data/train/fresh/` and `data/train/rotten/`

**"No images found"**
- Check file extensions (.jpg, .png)
- Make sure images are directly in class folders, not subfolders

**"Cannot create data generators"**
- Verify folder structure matches:
  ```
  data/
  └── train/
      ├── fresh/
      └── rotten/
  ```

---

**Remember:** Download data FIRST, then run notebook! 📥




