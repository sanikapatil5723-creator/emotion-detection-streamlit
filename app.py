import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection from Text",
    page_icon="😊",
    layout="centered"
)

# -----------------------------
# SIDEBAR (UI/UX)
# -----------------------------
st.sidebar.title("⚙️ Settings")
show_table = st.sidebar.checkbox("Show emotion score table", value=True)
show_chart = st.sidebar.checkbox("Show emotion chart", value=True)
save_history = st.sidebar.checkbox("Save results in history", value=True)

st.sidebar.markdown("---")
st.sidebar.write("📌 **Project:** Emotion Detection from Text")
st.sidebar.write("🧠 **Model:** DistilRoBERTa (HuggingFace)")

# -----------------------------
# MAIN UI
# -----------------------------
st.title("😊 Emotion Detection from Text")
st.write("Type a sentence and this app will detect the emotion using an AI model.")

# -----------------------------
# EMOJI MAPPING (Innovation)
# -----------------------------
EMOJI_MAP = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐",
}

# -----------------------------
# SUGGESTIONS (Innovation)
# -----------------------------
SUGGESTIONS = {
    "joy": "Keep going! Stay consistent and enjoy the moment ✨",
    "sadness": "Try talking to a friend or taking a short break 🌿",
    "anger": "Take a deep breath and relax for a minute 🧘",
    "fear": "It’s okay to feel nervous. Prepare step-by-step 💪",
    "surprise": "That’s interesting! Take a moment to process it 😄",
    "love": "Great! Positive emotions are powerful ❤️",
    "neutral": "Looks calm and balanced. Stay focused 🎯",
}

# -----------------------------
# LOAD MODEL (Core Logic)
# -----------------------------
@st.cache_resource
def load_emotion_model():
    """
    Loads the pre-trained emotion detection model only once.
    This improves performance and avoids re-downloading.
    """
    return pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

with st.spinner("Loading AI model (first time may take 2–5 minutes)..."):
    model = load_emotion_model()

# -----------------------------
# SESSION STATE FOR HISTORY
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# USER INPUT
# -----------------------------
text = st.text_area(
    "Enter your text:",
    height=150,
    placeholder="Example: I am feeling nervous about my interview tomorrow..."
)

col1, col2 = st.columns([1, 1])

with col1:
    detect_btn = st.button("🔍 Detect Emotion")

with col2:
    clear_btn = st.button("🧹 Clear History")

if clear_btn:
    st.session_state.history = []
    st.success("History cleared!")

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
if detect_btn:
    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    # Optional: limit very long text (stability improvement)
    if len(text) > 1000:
        st.warning("Text is too long. Please keep it under 1000 characters.")
        st.stop()

    try:
        results = model(text)[0]  # list of dicts with label + score
    except Exception:
        st.error("Model failed to analyze the text. Please try again.")
        st.stop()

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Top emotion
    top_emotion = df.loc[0, "label"]
    confidence = df.loc[0, "score"] * 100

    emoji = EMOJI_MAP.get(top_emotion, "🙂")
    suggestion = SUGGESTIONS.get(top_emotion, "Stay positive and keep improving!")

    # Display result
    st.success(f"Detected Emotion: **{top_emotion.upper()}** {emoji}")
    st.info(f"Confidence: **{confidence:.2f}%**")
    st.write(f"💡 **Suggestion:** {suggestion}")

    # Save history
    if save_history:
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Text": text,
            "Emotion": top_emotion,
            "Confidence(%)": round(confidence, 2)
        })

    # Show table
    if show_table:
        st.subheader("📌 Emotion Score Table")
        df_display = df.copy()
        df_display["score"] = (df_display["score"] * 100).round(2)
        df_display.rename(columns={"label": "Emotion", "score": "Score (%)"}, inplace=True)
        st.dataframe(df_display, use_container_width=True)

    # Show chart
    if show_chart:
        st.subheader("📊 Emotion Probability Chart")
        fig, ax = plt.subplots()
        ax.bar(df["label"], df["score"])
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# -----------------------------
# HISTORY SECTION (Innovation + Viva)
# -----------------------------
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.subheader("🕒 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    # Download history as CSV
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv,
        file_name="emotion_history.csv",
        mime="text/csv"
    )
