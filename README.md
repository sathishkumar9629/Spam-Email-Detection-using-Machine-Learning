# 📧 Spam Email Classifier

A Machine Learning project that classifies messages as Spam or Ham (Not Spam) using the Naive Bayes algorithm.

## 🚀 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning techniques to detect spam messages with high accuracy.

The model is trained on the SMS Spam Collection dataset and achieves 98% accuracy.

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer (Text Feature Extraction)
- Multinomial Naive Bayes

---

## 📂 Dataset

SMS Spam Collection Dataset  
Contains labeled messages as:
- Ham (Not Spam)
- Spam

---

## ⚙️ How It Works

1. Load dataset
2. Clean and preprocess data
3. Convert text into numerical features using CountVectorizer
4. Split data into training and testing sets
5. Train using Multinomial Naive Bayes
6. Evaluate model accuracy

---

## 📊 Model Performance

Accuracy: **98%**

---

## ▶️ How to Run

```bash
pip install pandas scikit-learn
python spam_classifier.py
