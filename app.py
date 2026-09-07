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
# LOAD TRAINED ML MODEL
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

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove social-media mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag symbol but keep the hashtag word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Keep only English letters and spaces
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
# ML PREDICTION FUNCTION
# ==================================================

def predict_message(message):

    # Apply same cleaning used during training
    cleaned_message = clean_text(message)

    # Convert text into TF-IDF numerical features
    vector = tfidf.transform(
        [cleaned_message]
    )

    # Predict class
    prediction = model.predict(
        vector
    )[0]

    # Get probability estimates
    probabilities = model.predict_proba(
        vector
    )[0]

    # Probability for class 1 = Disaster
    disaster_probability = probabilities[1]

    return prediction, disaster_probability


# ==================================================
# AI REPORT GENERATION
# ==================================================

def generate_ai_report(results):

    client = get_ollama_client()

    # Keep track of messages classified as Disaster
    disaster_messages = [
        item
        for item in results
        if item["prediction"] == "Disaster"
    ]

    # Convert ML outputs into text for the LLM
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

    # Prompt used only for report curation
    prompt = f"""
You are an AI report-curation assistant for CrisisLens.

The messages below have ALREADY been classified by a trained
machine-learning model.

You must NOT perform your own disaster classification and must
NOT override the ML model's labels.

Classifier:
TF-IDF + Logistic Regression

Total messages analysed: {len(results)}
Messages classified as Disaster: {len(disaster_messages)}

Machine-learning results:
{results_text}

Create a concise professional incident-screening report using
ONLY the information provided above.

Use exactly these sections:

1. Executive Summary
2. Potential Incident Signals
3. Highest-Priority Messages
4. Non-Disaster Messages
5. Common Themes
6. Recommended Human Review
7. Limitations

Rules:

- Only messages labelled "Disaster" by the ML model should appear
  under Potential Incident Signals and Highest-Priority Messages.

- Messages labelled "Not Disaster" must remain under
  Non-Disaster Messages. Do not reclassify them.

- Rank Disaster messages by their supplied disaster probability.

- Treat every message as an unverified social-media report.

- Do not state that any disaster definitely occurred.

- Do not invent dates, timestamps, locations, casualties,
  organizations, teams, signatories, infrastructure damage,
  authorities, or other facts.

- Do not assume the messages were posted at the same time or
  originated from the same geographic area.

- Do not add a report date unless one was explicitly provided.

- Do not invent relationships between separate messages.

- Do not call the probability a certainty.

- Recommended Human Review should give only general verification
  guidance such as checking authoritative sources and reviewing
  high-probability messages.

- Clearly state that ML models can produce both false positives
  and false negatives.

- Keep the report concise and factual.
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

    st.caption(
        "Feature Extraction: TF-IDF"
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

        The classification layer uses TF-IDF text features
        and a tuned Logistic Regression classifier.

        Generative AI is used downstream to curate multiple
        classification results into a structured incident
        screening report.
        """
    )

    st.divider()

    # ----------------------------------------------
    # FINAL MODEL PERFORMANCE
    # ----------------------------------------------

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

    # ----------------------------------------------
    # MODEL CONFIGURATION
    # ----------------------------------------------

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

    # ----------------------------------------------
    # DATASET INFORMATION
    # ----------------------------------------------

    st.subheader(
        "Dataset"
    )

    st.write(
        """
        The model was trained using the Kaggle
        **Natural Language Processing with Disaster Tweets**
        dataset.

        The original training dataset contained 7,613 rows.

        After removing conflicting labels and duplicate
        tweet texts, approximately **7,485 samples**
        remained for modelling.
        """
    )

    st.divider()

    # ----------------------------------------------
    # SYSTEM ARCHITECTURE
    # ----------------------------------------------

    st.subheader(
        "System Workflow"
    )

    st.code(
        """
Social-Media Message
        ↓
Text Cleaning
        ↓
TF-IDF Feature Extraction
        ↓
Logistic Regression
        ↓
Disaster / Not Disaster
        ↓
Probability Score
        ↓
Multiple Classification Results
        ↓
Ollama Cloud LLM
        ↓
AI-Curated Incident Report
        """,
        language=None
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

        CrisisLens will estimate whether the message
        potentially refers to a real disaster.
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
                real-world emergency.

                Potential disaster messages should be
                reviewed by a human.
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
        with **one message on each line**.

        CrisisLens will first classify every message
        using the trained machine-learning model.

        The classification results and probability scores
        will then be passed to generative AI to create a
        structured incident-screening report.
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

        # ------------------------------------------
        # SPLIT INPUT INTO SEPARATE MESSAGES
        # ------------------------------------------

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

            # --------------------------------------
            # ML CLASSIFICATION
            # --------------------------------------

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

            # --------------------------------------
            # DISPLAY ML RESULTS
            # --------------------------------------

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

            # --------------------------------------
            # SUMMARY COUNTS
            # --------------------------------------

            disaster_count = sum(
                item["prediction"] == "Disaster"
                for item in results
            )

            non_disaster_count = (
                len(results) - disaster_count
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
                    non_disaster_count
                )

            st.divider()

            # --------------------------------------
            # AI REPORT
            # --------------------------------------

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
