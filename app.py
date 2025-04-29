import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (Make sure these are downloaded at runtime)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize text into words
    text = word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Perform stemming on each word
    text = [ps.stem(i) for i in text]

    # Join the list of words back into a string
    return " ".join(text)

# Load the trained model and TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the processed input
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict using the model
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
