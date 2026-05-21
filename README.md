# Aircraft Engine RUL Prediction System

## Project Overview

This project predicts the Remaining Useful Life (RUL) of aircraft engines using:

- Machine Learning (Random Forest)
- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)
- Streamlit Web Application

The project uses the NASA Turbofan Engine Degradation Simulation Dataset (FD001).

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

# Project Structure

```bash
rul_prediction_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── models/
│   ├── ml_model.pkl
│   ├── ann_model.h5
│   ├── lstm_model.h5
│   └── scaler.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_ml.py
│   ├── train_ann.py
│   └── train_lstm.py
```

---

# Dataset

Dataset Files:

- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

The dataset contains aircraft engine sensor readings collected over multiple operational cycles.

---

# Features

- Predict Remaining Useful Life (RUL)
- Compare ML and Deep Learning models
- Interactive Streamlit Interface
- Upload test dataset and get predictions

---

# Installation

## Step 1 — Clone Project

```bash
git clone <repository-link>
```

OR manually create the project folder.

---

## Step 2 — Create Virtual Environment

```bash
python -m venv tfenv
```

Activate environment:

### Windows

```bash
tfenv\Scripts\activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Models

Go to project directory:

```bash
cd rul_prediction_project
```

---

## Train Machine Learning Model

```bash
python -m src.train_ml
```

---

## Train ANN Model

```bash
python -m src.train_ann
```

---

## Train LSTM Model

```bash
python -m src.train_lstm
```

---

# Run Streamlit Application

```bash
streamlit run app.py
```

---

# Models Used

## 1. Random Forest Regressor

Used as the Machine Learning baseline model.

---

## 2. Artificial Neural Network (ANN)

Used for Deep Learning prediction.

---

## 3. Long Short-Term Memory (LSTM)

Used for sequence-based prediction and time-series learning.

---

# Streamlit Application

The Streamlit application allows users to:

- Select prediction model
- Upload test dataset
- Predict engine RUL
- View prediction results

---

# Future Improvements

- Add model comparison graphs
- Add accuracy visualization
- Deploy on Streamlit Cloud
- Improve LSTM performance
- Add real-time sensor prediction

---

# Author

Nilmoni pangas
