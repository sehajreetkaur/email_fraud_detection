# preprocess.py

import pandas as pd
import os
import re
import pickle
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 1. LOAD DATA ──────────────────────────────────────────────
path = kagglehub.dataset_download("llabhishekll/fraud-email-dataset")
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

# ── 2. DROP MISSING VALUES ─────────────────────────────────────
print(f"Before drop: {df.shape}")
df = df.dropna(subset=['Text'])
print(f"After drop: {df.shape}")

# ── 3. CLEAN TEXT ──────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()                        # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)      # remove URLs
    text = re.sub(r'\S+@\S+', '', text)             # remove emails
    text = re.sub(r'[^a-z\s]', '', text)            # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()        # remove extra spaces
    return text

df['clean_text'] = df['Text'].apply(clean_text)

print("\n✅ Sample cleaned text:")
print(df[['Text', 'clean_text']].head(3))

# ── 4. TRAIN / TEST SPLIT ──────────────────────────────────────
X = df['clean_text']
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train size: {X_train.shape[0]}")
print(f"✅ Test size:  {X_test.shape[0]}")

# ── 5. TF-IDF VECTORIZATION ────────────────────────────────────
# Converts text into numbers the ML model can understand
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n✅ TF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"✅ TF-IDF matrix shape (test):  {X_test_tfidf.shape}")

# ── 6. SAVE EVERYTHING ─────────────────────────────────────────
os.makedirs('data', exist_ok=True)

# Save processed data
pickle.dump(X_train_tfidf, open('data/X_train.pkl', 'wb'))
pickle.dump(X_test_tfidf,  open('data/X_test.pkl', 'wb'))
pickle.dump(y_train,       open('data/y_train.pkl', 'wb'))
pickle.dump(y_test,        open('data/y_test.pkl', 'wb'))

# Save the vectorizer (needed later for the app!)
pickle.dump(tfidf, open('data/tfidf_vectorizer.pkl', 'wb'))

print("\n🎉 Phase 2 Complete! All data saved in /data folder")
print("Files saved:")
print("  data/X_train.pkl")
print("  data/X_test.pkl")
print("  data/y_train.pkl")
print("  data/y_test.pkl")
print("  data/tfidf_vectorizer.pkl")
