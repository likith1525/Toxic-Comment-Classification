# 🛡️ Toxicity and Sentiment Analysis of Comments

## Overview
This project is an end-to-end Natural Language Processing (NLP) pipeline designed to classify Wikipedia comments into various categories of toxicity. It also includes a sentiment analysis component to gauge the overall tone of the text. The project culminates in a Streamlit web application that allows users to test the model in real-time.

## Features
* **Multi-Label Toxicity Classification:** Detects six distinct types of toxicity: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
* **Sentiment Analysis:** Utilizes NLTK's VADER to categorize comments as Positive, Negative, or Neutral.
* **Interactive Web App:** A clean, user-friendly Streamlit interface for instant text analysis.

## Exploratory Data Analysis (EDA) Observations
During the data exploration phase, several key insights were uncovered:
* **Class Imbalance:** The vast majority of comments are clean, requiring careful thresholding and evaluation metrics during model training.
* **Comment Length:** Toxic comments generally exhibit a different word-count distribution compared to clean comments, often skewing slightly shorter but denser in specific vocabulary.
* **Vocabulary Differences:** Word clouds and N-gram analysis (Bigrams) revealed stark contrasts in vocabulary. Clean comments utilize constructive and neutral language, while toxic comments are heavily dominated by explicit, aggressive, and targeting terminology.

## Model Architecture & Evaluation
* **Text Preprocessing:** Comments were cleaned by removing URLs, mentions, hashtags, special characters, and stopwords, followed by lemmatization.
* **Vectorization:** Text was converted to numerical features using `TfidfVectorizer` (Top 5000 features, 1-2 N-grams).
* **Classifier:** A `LinearSVC` (Support Vector Machine) wrapped in a `OneVsRestClassifier` was utilized to handle the multi-label nature of the dataset.
* **Performance:** The LinearSVC outperformed Logistic Regression, showcasing strong accuracy and reliable F1-scores across the individual toxicity classes.

## How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/toxicity-comment-classifier.git](https://github.com/your-username/toxicity-comment-classifier.git)
cd toxicity-comment-classifier

```

**2. Run in your command prompt**
* Download all the pkl & app files provided
* Save them in a separte folder
* Open command prompt in your windows
* enter "cd (folder path)"
* Pass "pip install streamlit"
* Pass "pip install joblib scikit-learn nltk"
* pass python -m streamlit run "app (3)"
## There you go enjoy your interface of Toxic Comment Classification.
