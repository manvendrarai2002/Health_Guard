# HealthGuard — ML Inference Demo 🧠

An end-to-end machine-learning demo that trains a Random Forest classifier on **synthetic health data**, exposes predictions through Flask, and provides a simple browser UI.

> **Important:** this is a software/ML portfolio project. The dataset is synthetic and the predictions are not medical advice or a clinical diagnostic system.

## 🔍 What the project demonstrates

- Synthetic dataset generation (5,000 samples)
- Random Forest classification
- Class-imbalance handling with SMOTE
- Model training and evaluation pipeline
- Flask inference API
- Browser-based prediction form
- Basic inference latency and load testing

## 🧱 Architecture

```text
Synthetic Data
      │
      ▼
Training + SMOTE
      │
      ▼
Random Forest Model
      │
      ▼
Flask API ───► HTML UI
```

## 📊 Reported results

Results documented for the current project run:

- Accuracy: approximately **87%**
- Recall: approximately **62% → 79%** after applying SMOTE
- Single prediction latency: a few milliseconds on the author's local machine

These figures are project-specific measurements on synthetic data and should not be interpreted as clinical performance.

## 🛠️ Tech Stack

**Language:** Python  
**ML:** Scikit-learn, Random Forest, imbalanced-learn / SMOTE  
**API:** Flask  
**UI:** HTML/CSS  
**Testing:** Python load-testing script

## 📁 Project Structure

```text
Health_Guard/
├── data/                  # Synthetic dataset
├── models/                # Trained model + metrics
├── src/
│   ├── generate_data.py
│   └── train_model.py
├── templates/
│   └── index.html
├── app.py
├── test_load.py
├── requirements.txt
└── interview_guide.md
```

## 🚀 Run locally

### Install

```bash
pip install -r requirements.txt
```

### Generate data and train

```bash
python src/generate_data.py
python src/train_model.py
```

### Start the API/UI

```bash
python app.py
```

Then open the local address printed by Flask.

### Optional load test

```bash
python test_load.py
```

## 📌 Engineering improvements to pursue

For a production-oriented version, the next steps would be automated tests, request validation, model/version tracking, structured logging, containerization and CI for training/inference checks.
