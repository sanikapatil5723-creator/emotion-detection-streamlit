# 😊 Emotion Detection from Text (Streamlit + NLP)

## 📌 Project Overview
This is a Streamlit-based mini project that detects emotions from user-entered text using a pre-trained NLP model from HuggingFace Transformers.

The application predicts the most likely emotion and displays:
- Detected Emotion
- Confidence Score
- Emotion score table
- Bar chart visualization
- Prediction history (with CSV download)

---

## 🎯 Features
- User enters any sentence or paragraph
- AI model predicts emotion (joy, sadness, anger, fear, love, surprise, neutral)
- Displays confidence percentage
- Shows all emotion probabilities in table format
- Visualizes emotion scores using a bar chart
- Stores history of predictions
- Download prediction history as CSV file

---

## 🧠 Technologies Used
- Python 3.10 / 3.11
- Streamlit (Web UI)
- HuggingFace Transformers (NLP model)
- PyTorch (Model backend)
- Pandas (Data table)
- Matplotlib (Chart)

---

## 🏗️ Architecture / Working
1. User inputs text in Streamlit UI.
2. The input is passed to a pre-trained Transformer model.
3. The model returns probabilities for multiple emotions.
4. The highest score emotion is selected as the final prediction.
5. The app displays the results, chart, and stores history.

---

## 📂 Project Structure
Emotion_app/
│── app.py
│── requirements.txt
│── README.md
│── venv/ (optional, not uploaded to GitHub)

---

## ⚙️ Installation & Run (Step-by-Step)

### 
1️⃣ Create and activate virtual environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate

2️⃣ Install requirements
python -m pip install -r requirements.txt

3️⃣ Run the Streamlit app
python -m streamlit run app.py

4️⃣ Open in browser
http://localhost:8501

# Sample Inputs
I am very happy today because I got selected.

I feel sad and lonely.

I am nervous about tomorrow’s interview.

I am angry about this situation.

# Notes

First-time run may take 2–5 minutes because the AI model downloads and loads.

After first load, the app runs fast.