import string
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# TRAIN MODELS (CACHED)

@st.cache_resource
def train_models():
    df = pd.read_csv("Spam_SMS.csv", encoding='latin-1')
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']

    df = df.dropna().drop_duplicates()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    def clean_text(text):
        text = text.lower()
        text = "".join([c for c in text if c not in string.punctuation])
        return text

    df['message'] = df['message'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train models
    nb_model = MultinomialNB(alpha=0.5)
    lr_model = LogisticRegression(max_iter=1000)

    nb_model.fit(X_train_vec, y_train)
    lr_model.fit(X_train_vec, y_train)

    nb_pred = nb_model.predict(X_test_vec)
    lr_pred = lr_model.predict(X_test_vec)

    nb_acc = accuracy_score(y_test, nb_pred)
    lr_acc = accuracy_score(y_test, lr_pred)

    # Save models
    joblib.dump(nb_model, "nb_model.joblib")
    joblib.dump(lr_model, "lr_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    return {
        "vectorizer": vectorizer,
        "nb_model": nb_model,
        "lr_model": lr_model,
        "X_test": X_test_vec,
        "y_test": y_test,
        "nb_pred": nb_pred,
        "lr_pred": lr_pred,
        "nb_acc": nb_acc,
        "lr_acc": lr_acc
    }

data = train_models()

# SIDEBAR NAVIGATION

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

# HOME PAGE

if page == "Home":
    st.title("📧 Email Spam Classifier")
    st.markdown("Compare **Naive Bayes** and **Logistic Regression** models.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Naive Bayes Accuracy", f"{data['nb_acc']:.2%}")
    with col2:
        st.metric("Logistic Regression Accuracy", f"{data['lr_acc']:.2%}")

    st.subheader("📊 Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(["Naive Bayes", "Logistic Regression"], [data['nb_acc'], data['lr_acc']])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    st.subheader("Confusion Matrix — Naive Bayes")
    cm = confusion_matrix(data['y_test'], data['nb_pred'])
    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
    disp.plot(ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 ROC Curve Comparison")

    nb_probs = data['nb_model'].predict_proba(data['X_test'])[:, 1]
    lr_probs = data['lr_model'].predict_proba(data['X_test'])[:, 1]

    nb_fpr, nb_tpr, _ = roc_curve(data['y_test'], nb_probs)
    lr_fpr, lr_tpr, _ = roc_curve(data['y_test'], lr_probs)

    nb_auc = auc(nb_fpr, nb_tpr)
    lr_auc = auc(lr_fpr, lr_tpr)

    fig3, ax3 = plt.subplots()
    ax3.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_auc:.2f})")
    ax3.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.2f})")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("⬇️ Download Models & Vectorizer")
    col3, col4, col5 = st.columns(3)
    with col3:
        with open("nb_model.joblib", "rb") as f:
            st.download_button("Download Naive Bayes Model", f, file_name="nb_model.joblib")
    with col4:
        with open("lr_model.joblib", "rb") as f:
            st.download_button("Download Logistic Regression Model", f, file_name="lr_model.joblib")
    with col5:
        with open("vectorizer.joblib", "rb") as f:
            st.download_button("Download TF-IDF Vectorizer", f, file_name="vectorizer.joblib")

# PREDICT PAGE

elif page == "Predict":
    st.title("✉️ Predict Message")
    model_choice = st.selectbox("Choose Model", ["Naive Bayes", "Logistic Regression"])
    user_input = st.text_area("Enter a message", height=150)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            vectorized = data["vectorizer"].transform([user_input])
            model = data["nb_model"] if model_choice == "Naive Bayes" else data["lr_model"]

            prediction = model.predict(vectorized)[0]
            probs = model.predict_proba(vectorized)[0]
            ham_prob, spam_prob = probs[0], probs[1]

            if prediction == 1:
                st.error(f"🚨 SPAM detected — Confidence: {spam_prob:.2f}")
            else:
                st.success(f"✅ NOT SPAM — Confidence: {ham_prob:.2f}")

            st.subheader("📈 Probability")
            fig3, ax3 = plt.subplots()
            ax3.bar(["Ham", "Spam"], [ham_prob, spam_prob])
            ax3.set_ylim(0, 1)
            st.pyplot(fig3)

    st.subheader("⬇️ Download Selected Model & Vectorizer")
    col6, col7 = st.columns(2)
    model_file = "nb_model.joblib" if model_choice == "Naive Bayes" else "lr_model.joblib"
    with col6:
        with open(model_file, "rb") as f:
            st.download_button(f"Download {model_choice} Model", f, file_name=model_file)
    with col7:
        with open("vectorizer.joblib", "rb") as f:
            st.download_button("Download TF-IDF Vectorizer", f, file_name="vectorizer.joblib")

# ABOUT PAGE

elif page == "About":
    st.title("📖 About This Project")
    st.markdown("""
    ### Email Spam Classifier

    Complete ML workflow:

    ✅ Data preprocessing  
    ✅ TF-IDF feature extraction  
    ✅ Model training & comparison  
    ✅ ROC Curve analysis  
    ✅ Interactive prediction & downloads  

    **Models**: Naive Bayes, Logistic Regression  
    **Tech**: Python, Scikit-learn, Streamlit, Pandas, Matplotlib  
    """)

st.sidebar.info("Built with ❤️ using Streamlit and Scikit-learn by Anthony Sergo")
