# Garbage Classification System (Custom Deep CNN + Flask Web UI)

An end-to-end, production-grade Deep Learning project using a custom Convolutional Neural Network (CNN) in TensorFlow 2.x for multi-class garbage classification across 6 waste categories (Cardboard, Glass, Metal, Paper, Plastic, Trash). Integrated with a modern glassmorphic Flask web interface for real-time inference and recycling guidance.

---

## 📌 Target Waste Categories

1. **Cardboard** 📦 — Flatten boxes; place in paperboard recycling bin.
2. **Glass** 🍾 — Rinse containers; place in glass recycling bin.
3. **Metal** 🥫 — Clean food residues; place in metal/aluminum recycling bin.
4. **Paper** 📄 — Keep clean and dry; place in paper recycling bin.
5. **Plastic** 🥤 — Check resin codes (#1 PET, #2 HDPE); rinse and recycle.
6. **Trash** 🗑️ — Non-recyclable composite waste; place in landfill bin.

---

## 🛠️ Technology Stack

- **Core & Logic**: Python 3.10+
- **Deep Learning Framework**: TensorFlow 2.x / Keras
- **Data Pipeline**: `tf.data.Dataset` (Data Augmentation, Stratified Splitting, Class Weight Balancing)
- **Computer Vision & Processing**: Pillow, OpenCV, NumPy, Pandas
- **Evaluation & Visualizations**: scikit-learn, Matplotlib, Seaborn
- **Web Application Server**: Flask REST API
- **Frontend Dashboard**: HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JavaScript

---

## 📂 Project Directory Structure

```text
DL/
├── dataset/                    # Dataset directory containing subdirectories per class
│   └── dataset-resized/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
├── models/                     # Trained model artifacts (.keras format)
│   └── best_garbage_cnn.keras
├── results/                    # Quantitative evaluation metrics & visualizations
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── classification_report.json
│   └── training_history.csv
├── src/                        # Modular Python source packages
│   ├── __init__.py              # Package initializer
│   ├── download_dataset.py      # Automated dataset downloader helper
│   ├── verify_dataset.py        # Dataset validation & corruption check
│   ├── data_preprocessing.py    # Stratified split, augmentation, tf.data pipeline
│   ├── model.py                 # Custom deep CNN architecture builder
│   ├── train.py                 # Model training loop with Keras callbacks
│   ├── evaluate.py              # Test set evaluation & plot generators
│   └── predict.py               # Single/batch inference & guidance mapping
├── static/                     # Web UI static assets
│   ├── css/
│   │   └── style.css            # Modern glassmorphism UI styling
│   ├── js/
│   │   └── main.js              # Drag-and-drop, dynamic classification, interactive charts
│   └── uploads/                 # Temporary user upload buffer
├── templates/
│   └── index.html               # Main Web UI dashboard
├── app.py                      # Flask Web Application & REST API
├── requirements.txt            # Python dependencies
└── README.md                   # Comprehensive documentation
```

---

## ⚡ Quick Start Guide

### 1. Run Dataset Verification
```bash
python -m src.verify_dataset
```

### 2. Train Custom CNN Model
```bash
python -m src.train
```

### 3. Evaluate Model Performance on Test Data
```bash
python -m src.evaluate
```

### 4. Launch Web Application UI
```bash
python app.py
```
Open **`http://localhost:5000`** in your web browser.

---

## 📄 License & Credits
Developed as a complete deep learning garbage classification project utilizing TensorFlow 2.x and Kaggle TrashNet dataset.
