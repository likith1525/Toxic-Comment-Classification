
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import exp # Import exp for sigmoid function

# Ensure NLTK data is downloaded for the Streamlit app context if run separately
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK components for the app
stop_words_app = set(stopwords.words('english'))
lemmatizer_app = WordNetLemmatizer()

# Load saved model and vectorizer
# Assuming the saved model is 'toxicity_svm_model.pkl'
model_app = joblib.load('toxicity_svm_model.pkl') 
vectorizer_app = joblib.load('tfidf_vectorizer.pkl')
# Define target classes directly in the app for self-containment
target_classes_app = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_text_app(text):
    text = str(text).lower() # Ensure text is a string
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces

    # Remove stopwords and lemmatize
    tokens = [lemmatizer_app.lemmatize(word) for word in text.split() if word not in stop_words_app]
    return " ".join(tokens)

# Streamlit UI
st.title("🛡️Toxicity Comment Classifier")
st.write("Enter a comment below to check for various toxicity types.")

user_input = st.text_area("Enter Text:", height=150)

if st.button("Analyze Toxicity"):
    if user_input:
        with st.spinner("Analyzing text..."):
            # Clean and vectorize input text
            cleaned_input = clean_text_app(user_input)
            vectorized_input = vectorizer_app.transform([cleaned_input])

            # Predict toxicity labels
            predictions = model_app.predict(vectorized_input)[0]
            
            # Retrieve probabilities or confidence scores
            probabilities = None
            if hasattr(model_app, 'predict_proba') and callable(getattr(model_app, 'predict_proba')):
                try:
                    # For OneVsRestClassifier, predict_proba exists if the base estimator supports it
                    probabilities = model_app.predict_proba(vectorized_input)[0]
                except AttributeError: 
                    pass # Base estimator does not have predict_proba
            
            if probabilities is None:
                # Fallback for models like LinearSVC that don't directly support predict_proba.
                # Using decision_function as a proxy for confidence, then scaling it.
                # For true probabilities, CalibratedClassifierCV should be used with LinearSVC.
                decision_scores = model_app.decision_function(vectorized_input)[0]
                # Simple scaling to roughly [0,1], for display purposes. Not true probabilities.
                # This is a heuristic: positive score -> toxic, negative -> not toxic.
                probabilities = [1 / (1 + exp(-score)) for score in decision_scores] # Sigmoid scaling

            st.subheader("Analysis Results:")

            # Display detected toxicity types
            is_toxic_overall = False
            for i, col in enumerate(target_classes_app):
                if predictions[i] == 1:
                    is_toxic_overall = True
                    # Display a warning with a scaled confidence score
                    st.error(f"⚠️ **{col.replace('_', ' ').capitalize()}** detected! (Confidence: {probabilities[i]*100:.2f}%) ")

            if not is_toxic_overall:
                st.success("✅ This text appears to be clean and safe!")

    else:
        st.warning("Please enter some text to analyze.")
