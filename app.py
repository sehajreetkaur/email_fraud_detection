# app.py

import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from reputation import analyze_sender

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Email Fraud Detector",
    page_icon="https://cdn-icons-png.flaticon.com/512/5087/5087579.png",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }

    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, label {
        color: #1a1a2e !important;
        font-family: 'Segoe UI', sans-serif !important;
    }

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        color: #1565c0 !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #5c6bc0 !important;
        margin-bottom: 2rem;
    }

    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e0e6f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .fraud-box {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border: 2px solid #e53935;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .legit-box {
        background: linear-gradient(135deg, #f0fff4, #d0f5e0);
        border: 2px solid #2e7d32;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        color: #1a1a2e !important;
    }
    .result-sub {
        font-size: 0.9rem;
        margin-top: 0.3rem;
        color: #555 !important;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e6f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1565c0 !important;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #607d8b !important;
        margin-top: 0.2rem;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1.5px solid #c5cae9 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 2px rgba(21,101,192,0.15) !important;
    }

    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1.5px solid #c5cae9 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 2px rgba(21,101,192,0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #1976d2);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: 0.2s;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2, #2196f3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(21,101,192,0.3);
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e6f0;
    }

    .stProgress > div > div {
        background-color: #1565c0 !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    h2 { color: #1565c0 !important; font-weight: 700 !important; }
    h3, h4 { color: #1a237e !important; font-weight: 600 !important; }

    hr { border-color: #e0e6f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & VECTORIZER ───────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open('data/best_model.pkl', 'rb'))
    tfidf = pickle.load(open('data/tfidf_vectorizer.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()

# ── TEXT CLEANER ──────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
    <div style='color:#455a64; font-size:0.9rem; line-height:1.8'>
    This app uses a <b style='color:#1565c0'>Random Forest</b> model
    trained on 11,929 real emails to detect fraud.<br><br>
    <b style='color:#1565c0'>Dataset:</b> Fraud Email Dataset (Kaggle)<br>
    <b style='color:#1565c0'>Features:</b> TF-IDF (5000 features)<br>
    <b style='color:#1565c0'>Model:</b> Random Forest Classifier<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Performance")

    metrics = {
        "Accuracy":  0.98,
        "Precision": 0.97,
        "Recall":    0.98,
        "F1 Score":  0.97,
        "ROC-AUC":   0.99
    }
    for metric, val in metrics.items():
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    padding:0.4rem 0; border-bottom:1px solid #e0e6f0;
                    color:#37474f; font-size:0.9rem'>
            <span>{metric}</span>
            <span style='color:#1565c0; font-weight:700'>{val:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Try These Examples")

    fraud_example = """DEAR FRIEND, I AM MR. JAMES BROWN FROM BANK OF AFRICA.
I HAVE A BUSINESS PROPOSAL OF $10,000,000 USD FOR YOU.
PLEASE CONTACT ME URGENTLY FOR MORE DETAILS.
YOUR COOPERATION IS NEEDED. GOD BLESS YOU."""

    legit_example = """Hi Sarah, just a reminder that our team meeting
is scheduled for Thursday at 2pm in Conference Room B.
Please bring your Q3 reports. Let me know if you have questions."""

    if st.button("Load Fraud Example"):
        st.session_state['email_input'] = fraud_example
    if st.button("Load Legit Example"):
        st.session_state['email_input'] = legit_example

# ── MAIN CONTENT ──────────────────────────────────────────────
st.markdown('<p class="main-title">Email Fraud Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Paste any email below to instantly detect if it is fraudulent</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Email Content Analyzer", "Sender Reputation Checker"])

# ── TAB 1: EMAIL CONTENT ANALYZER ────────────────────────────
with tab1:

    col1, col2 = st.columns([2, 1])

    with col1:
        sender_input = st.text_input(
            "Sender Email Address",
            placeholder="Enter the sender's email address here e.g. support@apple.com"
        )
        email_input = st.text_area(
            "Paste Email Content Here",
            value=st.session_state.get('email_input', ''),
            height=200,
            placeholder="Paste the email text you want to analyze..."
        )
        analyze_btn = st.button("Analyze Email", use_container_width=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h4 style='color:#1565c0; margin-top:0'>How It Works</h4>
            <ol style='color:#455a64; font-size:0.88rem; line-height:1.9; padding-left:1.2rem'>
                <li>Enter the sender email</li>
                <li>Paste the email text</li>
                <li>Click Analyze Email</li>
                <li>Get instant prediction</li>
                <li>See confidence score</li>
            </ol>
        </div>
        <div class='card'>
            <h4 style='color:#1565c0; margin-top:0'>Common Fraud Signs</h4>
            <ul style='color:#455a64; font-size:0.88rem; line-height:1.9; padding-left:1.2rem'>
                <li>Urgent money requests</li>
                <li>Lottery winnings</li>
                <li>Bank transfer offers</li>
                <li>ALL CAPS writing</li>
                <li>Suspicious links</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── ANALYSIS ──────────────────────────────────────────────
    if analyze_btn and email_input.strip():
        cleaned    = clean_text(email_input)
        vec        = tfidf.transform([cleaned])
        probs      = model.predict_proba(vec)[0]
        fraud_prob = probs[1]
        legit_prob = probs[0]
        pred       = int(fraud_prob >= 0.70)

        st.markdown("---")
        st.markdown("## Analysis Results")

        if sender_input.strip():
            st.markdown(f"""
            <div style='background:#f0f4ff; border-left:4px solid #1565c0;
                        border-radius:6px; padding:0.8rem 1.2rem;
                        color:#37474f; font-size:0.9rem; margin-bottom:1rem'>
                <b style='color:#1565c0'>Sender:</b> {sender_input}
            </div>
            """, unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            if pred == 1:
                st.markdown("""
                <div class='fraud-box'>
                    <p class='result-label' style='color:#c62828 !important'>FRAUD DETECTED</p>
                    <p class='result-sub'>This email is likely fraudulent</p>
                </div>
                """, unsafe_allow_html=True)
            elif fraud_prob >= 0.45:
                st.markdown("""
                <div style='background:linear-gradient(135deg,#fffde7,#fff9c4);
                            border:2px solid #f9a825; border-radius:12px;
                            padding:1.5rem; text-align:center'>
                    <p class='result-label' style='color:#f57f17 !important'>UNCERTAIN</p>
                    <p class='result-sub' style='color:#555'>
                        This email has some suspicious patterns but may be legitimate.
                        Security emails from Apple, Google or banks often trigger this.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='legit-box'>
                    <p class='result-label' style='color:#2e7d32 !important'>LEGITIMATE</p>
                    <p class='result-sub'>This email appears to be safe</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#c62828 !important'>{fraud_prob*100:.1f}%</div>
                <div class='metric-label'>Fraud Probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(fraud_prob))

        with res_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#2e7d32 !important'>{legit_prob*100:.1f}%</div>
                <div class='metric-label'>Legitimate Probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(legit_prob))

        # Word analysis
        st.markdown("---")
        st.markdown("## Word Analysis")

        wc_col1, wc_col2 = st.columns(2)

        with wc_col1:
            st.markdown("#### Word Cloud")
            if cleaned.strip():
                wc = WordCloud(
                    width=600, height=300,
                    background_color='#ffffff',
                    colormap='Blues',
                    max_words=80
                ).generate(cleaned)
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor('#ffffff')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()

        with wc_col2:
            st.markdown("#### Top Words in This Email")
            words = cleaned.split()
            stopwords = {'the','and','to','of','a','in','is','it','you','that',
                         'was','for','on','are','with','as','this','be','have','or','i'}
            words = [w for w in words if w not in stopwords and len(w) > 2]
            word_freq = Counter(words).most_common(8)

            if word_freq:
                wds, cnts = zip(*word_freq)
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f5f7fa')
                bar_color = '#e53935' if pred == 1 else '#2e7d32'
                ax.barh(list(wds), list(cnts), color=bar_color, alpha=0.85)
                ax.tick_params(colors='#37474f')
                for spine in ax.spines.values():
                    spine.set_color('#e0e6f0')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Email stats
        st.markdown("---")
        st.markdown("## Email Statistics")

        s1, s2, s3, s4 = st.columns(4)
        word_count  = len(email_input.split())
        char_count  = len(email_input)
        upper_count = sum(1 for c in email_input if c.isupper())
        upper_pct   = (upper_count / max(char_count, 1)) * 100

        for col, label, val in zip(
            [s1, s2, s3, s4],
            ["Word Count", "Characters", "Uppercase %", "Unique Words"],
            [word_count, char_count, f"{upper_pct:.1f}%", len(set(email_input.lower().split()))]
        ):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='background:#f0f4ff; border-left:4px solid #1565c0;
                    border-radius:6px; padding:1rem 1.2rem; color:#37474f;
                    font-size:0.88rem; line-height:1.7'>
            <b style='color:#1565c0'>Note on Accuracy:</b> This model was trained on a general
            spam dataset and may flag legitimate security emails from Apple, Google, or banks
            as suspicious — these emails intentionally use urgent language similar to fraud.
            Always use your own judgment alongside this tool.
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please paste some email content first.")

# ── TAB 2: SENDER REPUTATION CHECKER ─────────────────────────
with tab2:
    st.markdown("## Sender Reputation Checker")
    st.markdown("Analyze how trustworthy an email sender is based on their domain.")

    rep_col1, rep_col2 = st.columns([2, 1])

    with rep_col1:
        sender_email = st.text_input("Sender Email Address",
                                      placeholder="e.g. support@apple.com")
        display_name = st.text_input("Display Name (optional)",
                                      placeholder="e.g. Apple Support")
        check_btn    = st.button("Check Sender Reputation", use_container_width=True)

    with rep_col2:
        st.markdown("""
        <div class='card'>
            <h4 style='color:#1565c0; margin-top:0'>What We Check</h4>
            <ul style='color:#455a64; font-size:0.88rem; line-height:1.9; padding-left:1.2rem'>
                <li>Email format validity</li>
                <li>SPF record (mail policy)</li>
                <li>DMARC record (verification)</li>
                <li>Domain age</li>
                <li>Display name mismatch</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if check_btn and sender_email.strip():
        with st.spinner("Analyzing sender reputation..."):
            rep = analyze_sender(sender_email.strip(), display_name.strip() or None)

        st.markdown("---")
        st.markdown("## Reputation Results")

        r1, r2, r3 = st.columns(3)

        risk_colors = {
            'Low':     ('#f0fff4', '#2e7d32', '#d0f5e0'),
            'Medium':  ('#fffde7', '#f57f17', '#fff9c4'),
            'High':    ('#fff5f5', '#c62828', '#ffe0e0'),
            'Unknown': ('#f5f7fa', '#607d8b', '#e0e6f0')
        }
        bg, text, border_bg = risk_colors[rep['risk_level']]

        with r1:
            st.markdown(f"""
            <div style='background:{bg}; border:2px solid {text};
                        border-radius:12px; padding:1.5rem; text-align:center'>
                <div style='font-size:1.8rem; font-weight:800; color:{text}'>{rep['risk_level']} Risk</div>
                <div style='color:#555; font-size:0.9rem; margin-top:0.3rem'>Overall Assessment</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#1565c0'>{rep['risk_score']}/100</div>
                <div class='metric-label'>Risk Score</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(rep['risk_score'] / 100)

        with r3:
            age_text = f"{rep['domain_age']} days" if rep['domain_age'] else "Unknown"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#1565c0; font-size:1.3rem'>{age_text}</div>
                <div class='metric-label'>Domain Age</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Security Checks")

        checks = {
            "Valid Email Format": rep['valid_format'],
            "SPF Record":         rep['has_spf'],
            "DMARC Record":       rep['has_dmarc'],
            "MX Record":          rep['has_mx'],
            "Trusted Domain":     rep['is_trusted'],
            "No Name Mismatch":   not rep['name_mismatch'],
        }

        c1, c2, c3 = st.columns(3)
        for i, (check, passed) in enumerate(checks.items()):
            col = [c1, c2, c3][i % 3]
            color  = '#2e7d32' if passed else '#c62828'
            symbol = 'Pass' if passed else 'Fail'
            col.markdown(f"""
            <div class='metric-card' style='margin-bottom:0.5rem'>
                <div style='font-size:1rem; font-weight:700; color:{color}'>{symbol}</div>
                <div class='metric-label'>{check}</div>
            </div>
            """, unsafe_allow_html=True)

        if rep['flags']:
            st.markdown("---")
            st.markdown("### Warning Flags")
            for flag in rep['flags']:
                st.markdown(f"""
                <div style='background:#fff5f5; border-left:4px solid #e53935;
                            border-radius:6px; padding:0.8rem 1rem;
                            color:#c62828; font-size:0.9rem; margin-bottom:0.5rem'>
                    {flag}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No warning flags found. This sender looks clean!")

    elif check_btn:
        st.warning("Please enter a sender email address.")