# AI HealthGuard - Multi-Disease Prediction System

A small, end-to-end ML demo that predicts potential health risks (Diabetes, Heart Disease) from basic patient vitals. It’s built to show the full workflow—data generation, model training, and a simple web UI—not to replace clinical judgment.

## Project Overview
I used a Random Forest classifier and handled class imbalance with SMOTE. The app exposes a lightweight Flask endpoint and a simple HTML form so you can try predictions quickly.

### Highlights
- **Solid baseline performance**: ~87% accuracy on held‑out data.
- **Imbalance handling**: SMOTE helped improve recall for minority classes.
- **Practical latency**: Inference runs in a few milliseconds on a local machine.
- **Simple API + UI**: A clean Flask endpoint and a minimal form for testing.

## Project Structure
```
AI_HealthGuard_System/
├── data/
│   └── medical_data.csv       # Generated synthetic dataset (5000 samples)
├── models/
│   └── model.pkl              # Trained Random Forest model
├── src/
│   ├── generate_data.py       # Data simulation script
│   └── train_model.py         # Training, SMOTE, and Evaluation pipeline
├── templates/
│   └── index.html             # Web Interface
├── app.py                     # Flask API Application
├── test_load.py               # Load testing script
└── requirements.txt           # Dependencies
```

## Setup & Installation

1. **Clone the repository** (or extract files).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Generate Data & Train Model
If you want to regenerate data or retrain the model:
```bash
python src/generate_data.py
python src/train_model.py
```
Output prints metrics and a quick latency check.

### 2. Start the Application
```bash
python app.py
```
The application will start at `http://127.0.0.1:5000`.

### 3. Usage
- Open your browser to `http://127.0.0.1:5000`.
- Enter patient details (Age, BMI, BP, etc.).
- Click **Analyze Health Risk**.

### 4. Load Testing (Optional)
If you want a quick concurrency check (10 simulated users):
```bash
python test_load.py
```

## Results (from this run)
- **Accuracy**: ~87% (5‑fold cross‑validation).
- **Recall**: Improved from baseline ~62% to ~79% with SMOTE.
- **Latency**: Single prediction around a few milliseconds on my machine.

### Notes & References
- [Training & Performance metrics](models/performance_metrics.txt)
- [Interview Preparation Guide](interview_guide.md)

---
*Portfolio demo project. The dataset is synthetic and outputs are not medical advice.*
