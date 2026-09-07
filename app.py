import re
import joblib
import streamlit as st
from ollama import Client


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CrisisLens",
    page_icon="🚨",
    layout="wide"
)


# ==================================================
# LOAD ML MODEL
# ==================================================

@st.cache_resource
def load_models():

    tfidf = joblib.load(
        "crisislens_tfidf.pkl"
    )

    model = joblib.load(
        "crisislens_model.pkl"
    )

    return tfidf, model


tfidf, model = load_models()


# ==================================================
# OLLAMA CLOUD CLIENT
# ==================================================

@st.cache_resource
def get_ollama_client():

    client = Client(
        host="https://ollama.com",
        headers={
            "Authorization":
            "Bearer " + st.secrets["OLLAMA_API_KEY"]
        }
    )

    return client


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag symbol but keep the word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Keep letters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==================================================
# ML PREDICTION
# ==================================================

def predict_message(message):

    cleaned_message = clean_text(message)

    vector = tfidf.transform(
        [cleaned_message]
    )

    prediction = model.predict(
        vector
    )[0]

    probabilities = model.predict_proba(
        vector
    )[0]

    disaster_probability = probabilities[1]

    return prediction, disaster_probability


# ==================================================
# AI REPORT GENERATION
# ==================================================

def generate_ai_report(results):

    client = get_ollama_client()

    disaster_messages = [
        item
        for item in results
        if item["prediction"] == "Disaster"
    ]

    results_text = ""

    for i, item in enumerate(
        results,
        start=1
    ):

        results_text += (
            f"\nMessage {i}: {item['message']}\n"
            f"ML Prediction: {item['prediction']}\n"
            f"Disaster Probability: "
            f"{item['probability']:.2f}%\n"
        )

    prompt = f"""
You are assisting with crisis-information curation.

The messages below were first classified by a machine-learning
model. You are NOT performing the classification yourself.

The ML classifier is a TF-IDF + Logistic Regression model.

Total messages analysed: {len(results)}
Messages classified as potential disasters: {len(disaster_messages)}

Machine-learning results:
{results_text}

Create a concise, professional incident-screening report.

Use these sections:

1. Executive Summary
2. Potential Incident Signals
3. Highest-Priority Messages
4. Common Themes
5. Recommended Human Review
6. Limitations

Important rules:
- Treat all social-media messages as unverified reports.
- Do not claim that any disaster definitely occurred.
- Do not invent locations, casualties, events, or facts.
- Base the report only on the messages and ML results provided.
- Clearly distinguish ML predictions from verified information.
- Prioritize higher disaster-probability messages for human review.
- State that the system may produce false positives and false negatives.
"""

    response = client.chat(
        model="gpt-oss:20b-cloud",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("CrisisLens")

    st.write(
        "AI-assisted disaster message classification"
    )

    mode = st.radio(
        "Choose Mode",
        [
            "Dashboard",
            "Classify Message",
            "AI Crisis Report"
        ]
    )

    st.divider()

    st.caption(
        "Final Model: Logistic Regression"
    )

    st.caption(
        "Best C: 5.0"
    )


# ==================================================
# DASHBOARD
# ==================================================

if mode == "Dashboard":

    st.title("🚨 CrisisLens")

    st.subheader(
        "Disaster Message Classification System"
    )

    st.write(
        """
        CrisisLens is a machine-learning system designed to
        identify whether a social-media message potentially
        refers to a real disaster.

        The system uses TF-IDF text features and a tuned
        Logistic Regression classifier.
        """
    )

    st.divider()

    st.subheader(
        "Final Model Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "78.82%"
        )

    with col2:
        st.metric(
            "Precision",
            "80.23%"
        )

    with col3:
        st.metric(
            "Recall",
            "66.77%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "72.88%"
        )

    st.divider()

    st.subheader(
        "Model Configuration"
    )

    st.write(
        "**Algorithm:** Logistic Regression"
    )

    st.write(
        "**Feature Extraction:** TF-IDF"
    )

    st.write(
        "**N-grams:** Unigrams + Bigrams `(1, 2)`"
    )

    st.write(
        "**Maximum Features:** 10,000"
    )

    st.write(
        "**Minimum Document Frequency:** 2"
    )

    st.write(
        "**Best Logistic Regression C:** 5.0"
    )

    st.write(
        "**Hyperparameter Selection:** GridSearchCV"
    )

    st.write(
        "**Cross-Validation:** 5 folds"
    )

    st.write(
        "**Best Cross-Validation F1:** 0.7554"
    )

    st.divider()

    st.subheader(
        "Dataset"
    )

    st.write(
        """
        The model was trained using the Kaggle
        **Natural Language Processing with Disaster Tweets**
        dataset.

        After removing conflicting labels and duplicate
        tweet texts, approximately **7,485 samples**
        remained for modelling.
        """
    )

    st.info(
        """
        CrisisLens predictions are machine-learning estimates.
        Messages identified as potential disasters should be
        verified by a human before taking action.
        """
    )


# ==================================================
# CLASSIFY MESSAGE
# ==================================================

elif mode == "Classify Message":

    st.title(
        "🔍 Classify Message"
    )

    st.write(
        """
        Enter a social-media message below.
        CrisisLens will estimate whether it potentially
        refers to a real disaster.
        """
    )

    message = st.text_area(
        "Message",
        placeholder=(
            "Example: Massive wildfire spreading near "
            "homes, residents are evacuating."
        ),
        height=150
    )

    if st.button(
        "Analyze Message",
        type="primary"
    ):

        if not message.strip():

            st.warning(
                "Please enter a message first."
            )

        else:

            prediction, disaster_probability = (
                predict_message(message)
            )

            disaster_percent = (
                disaster_probability * 100
            )

            non_disaster_percent = (
                (1 - disaster_probability) * 100
            )

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            if prediction == 1:

                st.error(
                    "⚠️ Potential Disaster"
                )

            else:

                st.success(
                    "✅ Not Classified as Disaster"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Disaster Probability",
                    f"{disaster_percent:.2f}%"
                )

            with col2:

                st.metric(
                    "Non-Disaster Probability",
                    f"{non_disaster_percent:.2f}%"
                )

            st.progress(
                float(disaster_probability)
            )

            with st.expander(
                "View Processing Details"
            ):

                st.write(
                    "**Original Message:**"
                )

                st.write(
                    message
                )

                st.write(
                    "**Cleaned Message:**"
                )

                st.write(
                    clean_text(message)
                )

            st.warning(
                """
                This prediction is not verification of a
                real-world emergency. Potential disaster
                messages should be reviewed by a human.
                """
            )


# ==================================================
# AI CRISIS REPORT
# ==================================================

elif mode == "AI Crisis Report":

    st.title(
        "🤖 AI Crisis Report"
    )

    st.write(
        """
        Paste multiple social-media messages below,
        with one message on each line.

        CrisisLens first classifies every message using
        the trained machine-learning model. The results
        are then curated into a structured incident report
        using generative AI.
        """
    )

    batch_text = st.text_area(
        "Messages",
        height=250,
        placeholder=(
            "Earthquake felt near downtown.\n"
            "That concert was absolutely fire.\n"
            "Several homes are flooding near the river.\n"
            "Traffic is terrible this morning.\n"
            "Residents evacuating due to wildfire."
        )
    )

    if st.button(
        "Generate Crisis Report",
        type="primary"
    ):

        messages = [
            line.strip()
            for line in batch_text.splitlines()
            if line.strip()
        ]

        if not messages:

            st.warning(
                "Please enter at least one message."
            )

        else:

            results = []

            with st.spinner(
                "Classifying messages..."
            ):

                for message in messages:

                    prediction, probability = (
                        predict_message(message)
                    )

                    label = (
                        "Disaster"
                        if prediction == 1
                        else "Not Disaster"
                    )

                    results.append({
                        "message": message,
                        "prediction": label,
                        "probability": probability * 100
                    })

            st.subheader(
                "Machine Learning Classification Results"
            )

            for i, item in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**Message {i}:** "
                    f"{item['message']}"
                )

                st.write(
                    f"Prediction: "
                    f"**{item['prediction']}**"
                )

                st.write(
                    f"Disaster Probability: "
                    f"**{item['probability']:.2f}%**"
                )

                st.divider()

            disaster_count = sum(
                item["prediction"] == "Disaster"
                for item in results
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Messages Analysed",
                    len(results)
                )

            with col2:

                st.metric(
                    "Potential Disasters",
                    disaster_count
                )

            with col3:

                st.metric(
                    "Not Classified as Disaster",
                    len(results) - disaster_count
                )

            st.divider()

            st.subheader(
                "AI-Curated Incident Report"
            )

            try:

                with st.spinner(
                    "Generating AI report..."
                ):

                    report = generate_ai_report(
                        results
                    )

                st.markdown(
                    report
                )

            except Exception as e:

                st.error(
                    "The AI report could not be generated."
                )

                st.exception(
                    e
                )

            st.warning(
                """
                CrisisLens classifications and AI-generated
                reports are decision-support outputs only.
                Social-media reports are unverified and should
                be reviewed by a human before any action is taken.
                """
            )
