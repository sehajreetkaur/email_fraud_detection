# Email Fraud Detector

A machine learning web application that detects fraudulent emails using Natural Language Processing and a Random Forest classifier. Built with Python, Scikit-learn, and Streamlit.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Pipeline](#project-pipeline)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## Project Overview

Email fraud (phishing, scams, spam) is one of the most common forms of cybercrime. This project builds an end-to-end machine learning pipeline that:

1. Trains a Random Forest classifier on 11,929 real labelled emails
2. Predicts whether any given email is fraudulent or legitimate
3. Serves the model through a clean, professional Streamlit web app
4. Includes a Sender Reputation Checker that analyzes email domains using live DNS lookups

---

## Live Demo

Run locally with:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## Features

### Tab 1 — Email Content Analyzer
- Paste any email content and get an instant fraud prediction
- Enter the sender email address for full context
- Three-state result: **FRAUD DETECTED** / **UNCERTAIN** / **LEGITIMATE**
- Fraud probability and legitimate probability scores with progress bars
- Word cloud visualization of the email content
- Top words bar chart highlighting suspicious language
- Email statistics including word count, character count, uppercase percentage, and unique word count
- Example fraud and legit emails to test with one click

### Tab 2 — Sender Reputation Checker
- Enter any sender email address to analyze the domain
- Checks SPF record (mail sending policy)
- Checks DMARC record (email authentication)
- Checks MX record (mail server existence)
- Detects brand new or suspicious domains via WHOIS lookup
- Flags display name mismatches (e.g. "Apple Support" sent from a non-Apple domain)
- Overall risk score out of 100 with Low / Medium / High rating

---

## Project Structure

```
email_fraud_detection/
│
├── eda.py                  # Phase 1 — Exploratory Data Analysis
├── preprocess.py           # Phase 2 — Text cleaning and TF-IDF vectorization
├── train.py                # Phase 3 — Model training and evaluation
├── app.py                  # Phase 4 — Streamlit web application
├── reputation.py           # Sender domain reputation checker
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── X_train.pkl         # TF-IDF training matrix
│   ├── X_test.pkl          # TF-IDF test matrix
│   ├── y_train.pkl         # Training labels
│   ├── y_test.pkl          # Test labels
│   ├── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
│   └── best_model.pkl      # Trained Random Forest model
│
└── plots/
    ├── class_distribution.png
    ├── email_length.png
    ├── wordcloud_fraud.png
    ├── wordcloud_legit.png
    ├── top_words_fraud.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── model_comparison.png
```

---

## Dataset

- **Source:** [Fraud Email Dataset on Kaggle](https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset)
- **Size:** 11,929 emails
- **Columns:** `Text` (email body), `Class` (0 = Legitimate, 1 = Fraud)
- **Class Distribution:** 6,742 legitimate emails, 5,187 fraud emails
- **Downloaded via:** `kagglehub`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/email_fraud_detection.git
cd email_fraud_detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
# Phase 1 — EDA
python eda.py

# Phase 2 — Preprocessing
python preprocess.py

# Phase 3 — Training
python train.py

# Phase 4 — Launch the app
streamlit run app.py
```

---

## Usage

### Analyzing an Email

1. Open the app at `http://localhost:8501`
2. In the **Email Content Analyzer** tab, enter the sender's email address (optional)
3. Paste the email body into the text area
4. Click **Analyze Email**
5. View the prediction, confidence scores, word cloud, and statistics

### Checking Sender Reputation

1. Click the **Sender Reputation Checker** tab
2. Enter the sender's email address (e.g. `support@apple.com`)
3. Optionally enter the display name (e.g. `Apple Support`)
4. Click **Check Sender Reputation**
5. View the risk score, security checks, and any warning flags

---

## Project Pipeline

### Phase 1 — Exploratory Data Analysis (`eda.py`)
- Loads the dataset via kagglehub
- Plots class distribution (fraud vs legitimate)
- Analyzes email length distributions
- Generates word clouds for fraud and legitimate emails
- Identifies the top 15 most common words in fraud emails

### Phase 2 — Preprocessing (`preprocess.py`)
- Drops rows with missing text
- Cleans text: lowercasing, removing URLs, emails, punctuation, and extra whitespace
- Splits data into 80% training and 20% test sets with stratification
- Applies TF-IDF vectorization with 5,000 features
- Saves all processed data and the fitted vectorizer as pickle files

### Phase 3 — Model Training (`train.py`)
- Trains and evaluates four models: Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting
- Uses class weights `{0:1, 1:3}` to penalize missed fraud detections
- Evaluates using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Performs 5-fold cross-validation on recall
- Generates confusion matrices, ROC curves, precision-recall curves, and a model comparison chart
- Saves the best performing model by F1 Score

### Phase 4 — Web App (`app.py`)
- Loads the trained model and TF-IDF vectorizer
- Applies a 0.70 fraud probability threshold for confident fraud detection
- Shows UNCERTAIN state for probabilities between 0.45 and 0.70
- Renders word clouds and bar charts for each analyzed email

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.98 | 0.97 | 0.98 | 0.97 | 0.99 |
| Naive Bayes | ~0.97 | ~0.96 | ~0.97 | ~0.96 | ~0.99 |
| Random Forest | **0.98** | **0.97** | **0.98** | **0.97** | **0.99** |
| Gradient Boosting | ~0.98 | ~0.97 | ~0.97 | ~0.97 | ~0.99 |

The **Random Forest** classifier was selected as the final model due to its strong recall score — in fraud detection, recall is the most critical metric as it measures how many actual fraud emails are correctly caught.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9 |
| Data Processing | Pandas, NumPy |
| NLP | Scikit-learn (TF-IDF), RegEx |
| Machine Learning | Scikit-learn (Random Forest, Logistic Regression, Naive Bayes, Gradient Boosting) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |
| Domain Analysis | dnspython, python-whois |
| Dataset | kagglehub |

---

## Known Limitations

- The model was trained on a general spam dataset and may flag legitimate security emails from companies like Apple, Google, or banks as suspicious or uncertain. This is because real security notification emails often contain urgent language that closely resembles fraud patterns.
- The sender reputation checker relies on public DNS records which may occasionally time out or return incomplete results.
- The model does not analyze email headers, attachments, or embedded links — only the plain text body.

---

## Future Improvements

- Integrate Gmail API to automatically scan and flag incoming emails in a real inbox
- Add support for analyzing email headers (Return-Path, X-Originating-IP)
- Train on a larger and more diverse dataset including modern phishing emails
- Add a user feedback mechanism so users can report incorrect predictions to improve the model over time
- Deploy to Streamlit Cloud for public access without local setup

