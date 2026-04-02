# 🎯 AI-Powered Interview Feedback System

## 🔗 Live Demo
👉 https://ai-interview-feedback-system-jegze3odhsam94wnjr8p48.streamlit.app

---

## 📌 Overview

This project is an AI-based interview preparation system that uses:

- Natural Language Processing (NLP)
- Large Language Models (LLM)

to evaluate answers and provide feedback.

---

## ✨ Features

- Answer evaluation using NLP
- Feedback generation using LLM
- Score calculation (semantic, keyword, structure)
- Performance metrics (Accuracy, Precision, Recall, F1 Score)
- Streamlit web interface

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- NLTK  
- Scikit-learn  
- HuggingFace / Gemini API  

---

## 📊 Results

- Accuracy: ~0.83
- Precision: ~1.00
- Recall: ~0.74
- F1 Score: ~0.84

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/sivakumari982/ai-interview-feedback-system.git
cd ai-interview-feedback-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py