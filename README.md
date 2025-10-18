# Multiple Disease Prediction System

## Overview

The **Multiple Disease Prediction System** is a machine learning–based web application that predicts the likelihood of three major diseases — **Heart Disease**, **Diabetes**, and **Parkinson’s Disease** — from patient input data.
It aims to assist in **early detection** and **preventive healthcare**, offering doctors and patients an accessible, data-driven decision-support tool.

---

## Features

* Predicts **three diseases** using trained ML models.
* Built with an intuitive **Streamlit web interface**.
* Real-time, user-friendly input forms.
* Displays results instantly with model confidence.
* Clean, modular, and easy to extend with new models.

---

## Tech Stack

| Component            | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python 3.x          |
| Web Framework        | Streamlit           |
| Machine Learning     | scikit-learn        |
| Data Handling        | pandas, NumPy       |
| Visualization        | Matplotlib, Seaborn |
| Model Storage        | Pickle (.pkl files) |

---

## Machine Learning Models

| Disease                 | Model Used                   | Dataset Source               |
| ----------------------- | ---------------------------- | ---------------------------- |
| **Heart Disease**       | Logistic Regression          | UCI Heart Disease Dataset    |
| **Diabetes**            | Support Vector Machine (SVM) | PIMA Indian Diabetes Dataset |
| **Parkinson’s Disease** | Random Forest Classifier     | UCI Parkinson’s Dataset      |

---

## Project Workflow

1. **Data Collection & Cleaning**

   * Datasets imported from trusted medical repositories (UCI / Kaggle).
   * Missing values handled and features normalized.
2. **Model Training**

   * Each disease trained with its best-performing ML algorithm.
   * Models serialized with `pickle` for deployment.
3. **Model Evaluation**

   * Accuracy, Precision, Recall, and Confusion Matrix used for performance analysis.
4. **Deployment**

   * Deployed as a **Streamlit web app** for real-time predictions.

---

## 💻 How to Run Locally

1. Clone the repository

   ```bash
   git clone https://github.com/<your-username>/Multiple-Disease-Prediction-System.git
   cd Multiple-Disease-Prediction-System
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in your terminal 

---

## Output Preview

| Disease     | Input                             | Prediction                               |
| ----------- | --------------------------------- | ---------------------------------------- |
| Heart       | Blood Pressure, Cholesterol, etc. | “You are likely safe from Heart Disease” |
| Diabetes    | Glucose, BMI, Age, etc.           | “You have a high risk of Diabetes”       |
| Parkinson’s | Voice parameters                  | “Low probability of Parkinson’s”         |

---

---

## License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## Acknowledgments

* UCI Machine Learning Repository
* Streamlit open-source community
* Scikit-learn documentation

---

> *“Early detection saves lives — bringing machine learning closer to healthcare.”*
