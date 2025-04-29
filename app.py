import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Detection")
st.markdown("Enter a message below to check if it's **Spam** or **Not Spam**.")

# User input
input_sms = st.text_input("✉️ Message Text")

# Preprocess, vectorize, predict, and display
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform input and predict
        transformed_input = tfidf.transform([input_sms])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.error("🚨 This is a SPAM message.")
        else:
            st.success("✅ This is NOT a spam message.")
