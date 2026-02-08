# Interview Guide: AI HealthGuard System

## 1. Project Introduction
"I built a small end‑to‑end ML demo that predicts potential risks for diabetes and heart disease from basic vitals. It’s not a medical tool—just a clean portfolio project that shows data generation, model training, and a simple web app." 

## 2. Key Technical Challenges & Solutions

### Challenge 1: Imbalanced Data
- **Problem**: The dataset is skewed toward the “healthy” class, so a model could score high accuracy while missing real risk cases.
- **Solution**: I used **SMOTE** to oversample minority classes.
- **Explanation**: "SMOTE creates realistic synthetic samples rather than just copying rows, which helped improve recall from ~62% to ~79%."

### Challenge 2: Real‑time Performance
- **Problem**: I wanted predictions to feel instant in the UI.
- **Solution**: I tuned hyperparameters with **GridSearchCV**.
- **Explanation**: "I tried a small grid of depth and split values and picked the best trade‑off between accuracy and speed. On my machine, a single prediction takes just a few milliseconds."

## 3. Tech Stack Justification
- **Flask**: "Lightweight, easy to demo, and fast to set up."
- **Random Forest**: "Strong baseline, handles non‑linear relationships, and is easy to interpret at a high level."
- **Scikit‑Learn**: "Reliable tooling for pipelines, metrics, and quick experimentation."

## 4. “Proof of Work”
- Show `models/performance_metrics.txt` as the training log.
- Walk through the setup steps in `README.md`.
- Run `test_load.py` if you want to demonstrate simple concurrency.
