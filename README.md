# 🛡️ UPI Fraud Detection

A machine learning–based system to detect fraudulent Unified Payments Interface (UPI) transactions.  
This project applies data preprocessing, feature extraction, model training, and web deployment using **FastAPI** to build a lightweight and accurate fraud detection solution.

---

## 🚀 Features
- **Data Preprocessing** — Cleans and transforms raw UPI transaction data (`preprocess.py`)
- **Model Training** — Trains ML models for fraud detection (`train_model.py`)
- **Model Evaluation** — Tests and evaluates model accuracy (`evaluate_model.py`)
- **FastAPI Backend** — Serves real-time fraud prediction via REST API (`fastapi_backend.py`)
- **Frontend Interface** — Simple web UI built with HTML/CSS (`templates/index.html`, `static/style.css`)
- **SQLite Database** — Stores user submissions and predictions (`database.py`)

---

## 🧠 Tech Stack
| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming | Python 3.9+ |
| ML / Data | pandas, scikit-learn |
| Web | FastAPI, Jinja2, Uvicorn |
| Database | SQLite3 |
| Others | Git, VS Code |

---

## 🧩 Project Structure
UPI FRAUD/
├── app.py # Main app entry point (optional wrapper)
├── fastapi_backend.py # FastAPI server for model prediction
├── train_model.py # Model training pipeline
├── evaluate_model.py # Evaluation and metrics
├── preprocess.py # Data preprocessing and feature engineering
├── database.py # SQLite database handler
├── requirements.txt # Python dependencies
├── static/style.css # Frontend styling
├── templates/index.html # Web interface
└── LICENSE


---

## ⚙️ Installation & Setup

1) **Clone the repo**
```bash
git clone https://github.com/<your-username>/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection/"UPI FRAUD"
