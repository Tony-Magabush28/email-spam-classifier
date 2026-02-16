# 📧 Email Spam Classifier

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green)](#license)

---

## 👨‍💻 Author
**Anthony Sergo**  
Machine Learning & Data Enthusiast  
[LinkedIn](https://www.linkedin.com/in/anthony-sergo1) | [GitHub](https://github.com/Tony-Magabush28) | [Portfolio](https://my-flask-portfolio.onrender.com)  

---

## 🚀 Project Overview
The Email Spam Classifier App is an interactive **Streamlit** web application that predicts whether an email or SMS message is spam.  

It demonstrates a complete **machine learning workflow**, including:

- Data preprocessing & cleaning  
- TF-IDF feature extraction  
- Model training & comparison (Naive Bayes vs Logistic Regression)  
- Model evaluation (accuracy, confusion matrix, ROC curve)  
- Interactive predictions with probability visualization  
- Downloadable trained models and vectorizer for reuse  

This project is ideal for showcasing **ML, NLP, and full-stack data science skills** to recruiters.

---

## 🧠 Problem Statement
Spam messages can cause fraud, privacy breaches, and a poor user experience.  
This project aims to **automatically detect spam messages** using supervised machine learning.

---

## 🔥 Features
- Interactive dashboard with tabs: **Home / Predict / About**  
- Multiple algorithm comparison: **Naive Bayes vs Logistic Regression**  
- Model evaluation: accuracy, confusion matrix, and ROC curves  
- Interactive prediction with confidence scores  
- Downloadable pre-trained models and TF-IDF vectorizer  
- Clean, professional UI using **Streamlit**

---

## 🛠 Technologies Used
- **Python** – Core programming  
- **Pandas & NumPy** – Data manipulation  
- **Scikit-learn** – ML algorithms (Naive Bayes, Logistic Regression)  
- **Streamlit** – Interactive dashboard and UI  
- **Matplotlib** – Visualizations (confusion matrix, ROC, probability charts)  
- **Joblib** – Save and load trained models  

---

## 📂 Project Structure
spam-classifier/
│
├── Spam_SMS.csv
├── app.py
├── spam_classifier_model.joblib
├── tfidf_vectorizer.joblib
├── requirements.txt
└── README.md


---

## 📊 Screenshots
- **Home Tab:** Model comparison & ROC curves  
- **Predict Tab:** Interactive message classification  

---

## 🧠 Usage
1. Go to the **Predict** tab  
2. Select a model (**Naive Bayes** or **Logistic Regression**)  
3. Enter a message and click **Predict**  
4. View the prediction result, confidence score, and probability chart  
5. Optionally, download the trained model and TF-IDF vectorizer  

---

## 📈 Model Performance
| Model               | Accuracy | AUC  |
|--------------------|---------|------|
| Naive Bayes         | 97.48%  | 0.98 |
| Logistic Regression | 94.86%  | 0.98 |

*Metrics may vary depending on the dataset and preprocessing.*

---

## ✨ Future Enhancements
- Add more ML algorithms: **Random Forest, XGBoost**  
- Improve preprocessing and feature engineering  

---

## 📥 Download Models
- **Naive Bayes Model:** `nb_model.joblib`  
- **Logistic Regression Model:** `lr_model.joblib`  
- **TF-IDF Vectorizer:** `vectorizer.joblib`  

---

## 📜 License
This project is for **educational and portfolio purposes**.

---

## [![Live Demo]
(https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge)](https://email-spam-classifier-jb9adewm5isnfjisgrwogd.streamlit.app/)
