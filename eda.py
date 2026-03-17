# eda.py

import pandas as pd
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
os.makedirs('plots', exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────
path = kagglehub.dataset_download("llabhishekll/fraud-email-dataset")
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print("✅ Dataset loaded!")
print("Shape:", df.shape)
print(df.head())

# ── 2. BASIC INFO ──────────────────────────────────────────────
print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Class Distribution ---")
print(df['Class'].value_counts())

# ── 3. CLASS DISTRIBUTION BAR CHART ───────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Fraud vs Legit Emails', fontsize=14, fontweight='bold')
plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/class_distribution.png', dpi=150)
plt.show()
print("✅ Saved: plots/class_distribution.png")

# ── 4. EMAIL LENGTH ANALYSIS ───────────────────────────────────
df['email_length'] = df['Text'].astype(str).apply(len)

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='email_length', hue='Class', bins=50,
             palette=['#2ecc71', '#e74c3c'], alpha=0.7)
plt.title('Email Length Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Character Count')
plt.ylabel('Frequency')
plt.legend(labels=['Fraud', 'Legit'])
plt.tight_layout()
plt.savefig('plots/email_length.png', dpi=150)
plt.show()
print("✅ Saved: plots/email_length.png")

# ── 5. WORD CLOUD — FRAUD EMAILS ──────────────────────────────
fraud_text = " ".join(df[df['Class'] == 1]['Text'].astype(str).tolist())
wc_fraud = WordCloud(width=800, height=400,
                     background_color='black',
                     colormap='Reds',
                     max_words=150).generate(fraud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc_fraud, interpolation='bilinear')
plt.axis('off')
plt.title('🚨 Most Common Words in FRAUD Emails', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/wordcloud_fraud.png', dpi=150)
plt.show()
print("✅ Saved: plots/wordcloud_fraud.png")

# ── 6. WORD CLOUD — LEGIT EMAILS ──────────────────────────────
legit_text = " ".join(df[df['Class'] == 0]['Text'].astype(str).tolist())
wc_legit = WordCloud(width=800, height=400,
                     background_color='white',
                     colormap='Greens',
                     max_words=150).generate(legit_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc_legit, interpolation='bilinear')
plt.axis('off')
plt.title('✅ Most Common Words in LEGIT Emails', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/wordcloud_legit.png', dpi=150)
plt.show()
print("✅ Saved: plots/wordcloud_legit.png")

# ── 7. TOP 15 MOST COMMON WORDS (FRAUD) ───────────────────────
def get_top_words(text, n=15):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {'the','and','to','of','a','in','is','it','you','that',
                 'was','for','on','are','with','as','this','be','have','or'}
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(n)

fraud_words = get_top_words(fraud_text)
words, counts = zip(*fraud_words)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(counts), y=list(words), palette='Reds_r')
plt.title('Top 15 Words in Fraud Emails', fontsize=14, fontweight='bold')
plt.xlabel('Frequency')
plt.tight_layout()
plt.savefig('plots/top_words_fraud.png', dpi=150)
plt.show()
print("✅ Saved: plots/top_words_fraud.png")

print("\n🎉 Phase 1 Complete! All plots saved in /plots folder")