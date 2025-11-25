# 🎯 Exact Run Order - What to Run and When

## Complete Sequence (Copy-Paste Ready)

### ⚙️ STEP 1: Initial Setup (One Time)
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
python setup.py
```
**✅ Check:** No errors = Good to go!

---

### 📥 STEP 2: Download Data (One Time - MANUAL!)
**⚠️ You MUST download manually - notebook won't do it!**

1. Go to: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
2. Click "Download" button
3. Extract ZIP file
4. Organize into:
   ```
   data/
   ├── train/
   │   ├── fresh/     ← Put images here
   │   └── rotten/    ← Put images here
   └── test/
       ├── fresh/
       └── rotten/
   ```

**OR use:** `python download_dataset.py` (if Kaggle API setup)

**✅ Check:** Run `python download_dataset.py` to verify = Ready!

---

### 🏋️ STEP 3: Train Model (One Time, Takes Time!)
```bash
jupyter notebook notebook/vendorclose_ai.ipynb
```
**In notebook:** Run ALL cells in order (Cell 1 → Cell 2 → ... → Cell 8)

**✅ Check:** File `models/fruit_classifier.h5` exists = Model ready!

---

### 🌐 STEP 4: Start API (Every Time You Want to Test)
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
**Keep this terminal open!**

**✅ Check:** See "Uvicorn running on http://0.0.0.0:8000" = API running!

---

### 🧪 STEP 5: Test API (In NEW Terminal)
```bash
# Basic test
python quick_test.py

# Or test prediction with image
python test_prediction.py data/test/fresh/apple_1.jpg
```
**✅ Check:** All tests pass = API working!

---

### 🖥️ STEP 6: Start UI (In NEW Terminal)
```bash
streamlit run app.py
```
**Keep API running in Terminal 1!**

**✅ Check:** Browser opens at http://localhost:8501 = UI working!

---

### 🧪 STEP 7: Load Testing (Optional, In NEW Terminal)
```bash
locust -f locustfile.py --host=http://localhost:8000
```
**✅ Check:** Locust UI opens = Load testing ready!

---

## 📋 Quick Command Reference

### Windows Users:
```bash
# Step 1: Setup
pip install -r requirements.txt
python setup.py

# Step 3: Train (in Jupyter)
jupyter notebook notebook/vendorclose_ai.ipynb

# Step 4: Start API
run_api.bat

# Step 5: Test
python quick_test.py

# Step 6: Start UI (new terminal)
run_ui.bat
```

### Linux/Mac Users:
```bash
# Step 1: Setup
pip install -r requirements.txt
python setup.py

# Step 3: Train (in Jupyter)
jupyter notebook notebook/vendorclose_ai.ipynb

# Step 4: Start API
chmod +x run_api.sh
./run_api.sh

# Step 5: Test
python quick_test.py

# Step 6: Start UI (new terminal)
streamlit run app.py
```

---

## 🎯 Daily Testing Workflow

**Every time you want to test:**

1. **Start API:**
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Start UI (new terminal):**
   ```bash
   streamlit run app.py
   ```

3. **Test in browser:** http://localhost:8501

**That's it!** Model only needs training once (unless you retrain).

---

## 🔄 What Runs When

| Component | When to Run | How Often |
|-----------|-------------|-----------|
| `setup.py` | First time only | Once |
| `notebook/vendorclose_ai.ipynb` | First time + when retraining | As needed |
| `api/main.py` | Every testing session | Every time |
| `app.py` | Every testing session | Every time |
| `locustfile.py` | Load testing | Optional |
| `quick_test.py` | After starting API | Every time |
| `test_prediction.py` | After starting API | Every time |

---

## ✅ Success Indicators

**After Step 1:**
- ✅ No errors in terminal

**After Step 3:**
- ✅ File exists: `models/fruit_classifier.h5`
- ✅ Training metrics displayed

**After Step 4:**
- ✅ Terminal shows: "Uvicorn running on http://0.0.0.0:8000"

**After Step 5:**
- ✅ All tests show ✅ (green checkmarks)

**After Step 6:**
- ✅ Browser opens with UI
- ✅ Can upload and predict images

---

## 🚨 Troubleshooting Order

**If something fails:**

1. **Check Step 1:** Dependencies installed?
2. **Check Step 3:** Model file exists?
3. **Check Step 4:** API actually running?
4. **Check ports:** 8000 (API) and 8501 (UI) free?
5. **Check paths:** Are you in project root directory?

---

**Follow this order exactly and everything will work! 🎉**

