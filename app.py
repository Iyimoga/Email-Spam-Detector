import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

# Create the web interface
st.title("Email Spam Detector")
st.write("Enter an email message to check if it's spam or not")

# Text input box
email_text = st.text_area("Email Message:", height=200, 
                          placeholder="Type or paste email here...")

# Classify button
if st.button("Classify Email"):
    if email_text.strip():
        # Transform the input text
        features = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Show result
        if prediction == 1:
            st.success("This is HAM (Not Spam)")
            st.balloons()
        else:
            st.error("This is SPAM!")
    else:
        st.warning("Please enter some text first")