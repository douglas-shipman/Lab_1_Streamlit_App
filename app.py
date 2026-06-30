import streamlit as st
import joblib


@st.cache_data
def load_model():
    return joblib.load("sentiment_model.pkl")


model = load_model()

st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review to predict whether the sentiment is positive or negative.")

user_text = st.text_area("Enter a movie review to analyze:")

if st.button("Analyze"):
    if user_text:
        prediction = model.predict([user_text])[0]
        probabilities = model.predict_proba([user_text])[0]
        confidence = max(probabilities) * 100

        if prediction == "positive":
            st.success("Predicted Sentiment: Positive")
        else:
            st.error("Predicted Sentiment: Negative")

        st.write(f"Prediction Probability: {confidence:.2f}%")
    else:
        st.write("Please enter a movie review.")