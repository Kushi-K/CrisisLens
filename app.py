import re
import joblib
import streamlit as st
from ollama import Client


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CrisisLens",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main app background */
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(239, 68, 68, 0.08),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                #090e18 0%,
                #0b1220 100%
            );
    }

    /* Main content width */
    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* Transparent top header */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #070c14;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] p {
        color: #94a3b8;
    }

    /* Headings */
    h1 {
        color: #f8fafc !important;
        letter-spacing: -0.04em;
    }

    h2, h3, h4 {
        color: #f1f5f9 !important;
    }

    /* Normal text */
    p, li {
        color: #cbd5e1;
    }

    /* Bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid #263244 !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.14);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 1rem;
        min-height: 125px;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* Text areas */
    .stTextArea textarea {
        background: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    .stTextArea textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 1px #64748b !important;
    }

    /* Buttons */
    .stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
    }

    .stDownloadButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        border-radius: 999px;
    }

    /* Divider */
    hr {
        border-color: #1e293b !important;
    }

    /* Captions */
    .stCaption {
        color: #64748b !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid #263244;
        border-radius: 12px;
    }

        /* ==========================
       AI REPORT TABLES
       ========================== */

    [data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        background: #0f172a !important;
        border: 1px solid #263244 !important;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stMarkdownContainer"] thead tr {
        background: #111827 !important;
    }

    [data-testid="stMarkdownContainer"] th {
        color: #f8fafc !important;
        background: #111827 !important;
        border: 1px solid #334155 !important;
        padding: 0.75rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMarkdownContainer"] td {
        color: #cbd5e1 !important;
        background: #0f172a !important;
        border: 1px solid #263244 !important;
        padding: 0.75rem !important;
    }

    [data-testid="stMarkdownContainer"] tbody tr:nth-child(even) td {
        background: #111827 !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #f8fafc !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

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


# ============================================================
# OLLAMA CLOUD CLIENT
# ============================================================

@st.cache_resource
def get_ollama_client():

    return Client(
        host="https://ollama.com",
        headers={
            "Authorization":
            "Bearer " + st.secrets["OLLAMA_API_KEY"]
        }
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove # but keep hashtag word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Keep English letters and spaces
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


# ============================================================
# ML PREDICTION
# ============================================================

def predict_message(message):

    cleaned_message = clean_text(
        message
    )

    vector = tfidf.transform(
        [cleaned_message]
    )

    prediction = model.predict(
        vector
    )[0]

    probabilities = model.predict_proba(
        vector
    )[0]

    disaster_probability = (
        probabilities[1]
    )

    return (
        prediction,
        disaster_probability
    )


# ============================================================
# AI REPORT GENERATION
# ============================================================

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
You are an AI report-curation assistant for CrisisLens.

The following messages have ALREADY been classified by a
trained machine-learning model.

You must NOT perform your own disaster classification.
You must NOT override the ML model's labels.

Classifier:
TF-IDF + Logistic Regression

Total messages analysed: {len(results)}
Messages classified as Disaster: {len(disaster_messages)}

Machine-learning results:
{results_text}

Create a concise professional incident-screening report
using ONLY the supplied messages and ML outputs.

Use exactly these sections:

1. Executive Summary
2. Potential Incident Signals
3. Highest-Priority Messages
4. Non-Disaster Messages
5. Common Themes
6. Recommended Human Review
7. Limitations

Rules:

- Only messages labelled "Disaster" by the ML classifier
  may appear under Potential Incident Signals and
  Highest-Priority Messages.

- Messages labelled "Not Disaster" must remain under
  Non-Disaster Messages.

- Do not reclassify any message.

- Rank Disaster messages using their supplied disaster
  probabilities.

- Treat every message as an unverified social-media report.

- Do not claim that any event definitely occurred.

- Do not invent dates, timestamps, locations, casualties,
  authorities, organizations, infrastructure damage,
  signatories, or other facts.

- Do not assume that separate messages were posted at the
  same time or originate from the same location.

- Do not invent relationships between messages.

- Probability is not certainty.

- Avoid unnecessary scientific categorization unless
  directly supported by the message itself.

- Recommended Human Review should contain only general
  verification guidance.

- State clearly that the classifier may produce false
  positives and false negatives.

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("CRISISLENS")

    st.caption(
        "Disaster intelligence and incident-screening platform"
    )

    st.divider()

    mode = st.radio(
        "Navigation",
        [
            "Overview",
            "Message Analysis",
            "Incident Intelligence"
        ]
    )

    st.divider()

    st.caption("SYSTEM STATUS")

    st.success(
        "● Operational"
    )

    st.caption("CLASSIFIER")

    st.write(
        "**Logistic Regression**"
    )

    st.caption(
        "Best C = 5.0"
    )

    st.caption("FEATURE LAYER")

    st.write(
        "**TF-IDF**"
    )

    st.caption(
        "Unigrams + Bigrams"
    )

    st.caption("AI CURATION")

    st.write(
        "**Ollama Cloud**"
    )


# ============================================================
# OVERVIEW
# ============================================================

if mode == "Overview":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.caption(
            "DISASTER INTELLIGENCE PLATFORM"
        )

        st.title(
            "CrisisLens"
        )

        st.markdown(
            """
            ### Machine-learning disaster screening with
            AI-assisted incident curation

            CrisisLens transforms noisy social-media messages
            into structured, reviewable intelligence.

            The classification layer is powered by a trained
            **TF-IDF + Logistic Regression** pipeline, while
            generative AI is used only downstream to curate
            multiple ML results into a structured report.
            """
        )

        st.write(
            "`Supervised ML`  "
            "`TF-IDF`  "
            "`Logistic Regression`  "
            "`Ollama Cloud`  "
            "`Human-in-the-loop`"
        )

    st.write("")

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.caption(
        "MODEL PERFORMANCE"
    )

    st.subheader(
        "Holdout Evaluation"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Accuracy",
            "78.82%",
            help=(
                "Percentage of all test messages "
                "classified correctly."
            )
        )

    with col2:

        st.metric(
            "Precision",
            "80.23%",
            help=(
                "When CrisisLens predicts Disaster, "
                "how often that prediction is correct."
            )
        )

    with col3:

        st.metric(
            "Recall",
            "66.77%",
            help=(
                "Percentage of real disaster messages "
                "detected by the classifier."
            )
        )

    with col4:

        st.metric(
            "F1 Score",
            "72.88%",
            help=(
                "Balanced measure of precision "
                "and recall."
            )
        )

    st.write("")

    # --------------------------------------------------------
    # SYSTEM PROFILE
    # --------------------------------------------------------

    st.caption(
        "SYSTEM PROFILE"
    )

    st.subheader(
        "Model & Training Configuration"
    )

    left, right = st.columns(2)

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Model Configuration"
            )

            st.write(
                "**Algorithm**"
            )
            st.caption(
                "Logistic Regression"
            )

            st.write(
                "**Feature Extraction**"
            )
            st.caption(
                "TF-IDF"
            )

            st.write(
                "**N-gram Range**"
            )
            st.caption(
                "(1, 2) — Unigrams + Bigrams"
            )

            st.write(
                "**Maximum Features**"
            )
            st.caption(
                "10,000"
            )

            st.write(
                "**Minimum Document Frequency**"
            )
            st.caption(
                "2"
            )

            st.write(
                "**Best Regularization C**"
            )
            st.caption(
                "5.0"
            )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Training & Validation"
            )

            st.write(
                "**Original Dataset**"
            )
            st.caption(
                "7,613 labelled tweets"
            )

            st.write(
                "**Clean Samples**"
            )
            st.caption(
                "7,485 samples"
            )

            st.write(
                "**Training / Holdout Split**"
            )
            st.caption(
                "80% / 20%"
            )

            st.write(
                "**Model Selection**"
            )
            st.caption(
                "Logistic Regression vs Naive Bayes vs Linear SVM"
            )

            st.write(
                "**Hyperparameter Search**"
            )
            st.caption(
                "GridSearchCV · 5-fold cross-validation"
            )

            st.write(
                "**Best CV F1**"
            )
            st.caption(
                "75.54%"
            )

    st.write("")

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.caption(
        "ARCHITECTURE"
    )

    st.subheader(
        "End-to-End Workflow"
    )

    with st.container(
        border=True
    ):

        st.code(
            """
Incoming Social-Media Message
            ↓
       Text Cleaning
            ↓
   TF-IDF Feature Extraction
            ↓
 Tuned Logistic Regression
            ↓
 Disaster / Not Disaster
            ↓
  Probability Estimation
            ↓
 Batch Classification Results
            ↓
     Ollama Cloud LLM
            ↓
 AI-Curated Incident Report
            ↓
      Human Verification
            """,
            language=None
        )

    st.info(
        """
        **Decision-support prototype:** CrisisLens predictions
        and social-media reports are unverified. Important
        signals must be independently reviewed before any
        real-world action is taken.
        """
    )


# ============================================================
# MESSAGE ANALYSIS
# ============================================================

elif mode == "Message Analysis":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.caption(
            "SINGLE MESSAGE SCREENING"
        )

        st.title(
            "Message Analysis"
        )

        st.markdown(
            """
            Analyse an individual social-media message with the
            trained CrisisLens classifier and inspect the model's
            estimated disaster probability.
            """
        )

    st.write("")

    st.caption(
        "MESSAGE INPUT"
    )

    st.subheader(
        "Analyse a Social-Media Message"
    )

    message = st.text_area(
        "Message",
        placeholder=(
            "Example: Massive wildfire spreading near "
            "homes, residents are evacuating."
        ),
        height=180,
        label_visibility="collapsed"
    )

    if st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True
    ):

        if not message.strip():

            st.warning(
                "Enter a message before running the analysis."
            )

        else:

            prediction, disaster_probability = (
                predict_message(message)
            )

            disaster_percent = (
                disaster_probability * 100
            )

            non_disaster_percent = (
                100 - disaster_percent
            )

            st.write("")

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            with st.container(
                border=True
            ):

                st.caption(
                    "CLASSIFICATION RESULT"
                )

                if prediction == 1:

                    st.error(
                        "### Potential Disaster"
                    )

                    st.write(
                        "The trained classifier identified "
                        "this message as a potential "
                        "disaster signal."
                    )

                else:

                    st.success(
                        "### Not Classified as Disaster"
                    )

                    st.write(
                        "The trained classifier did not "
                        "identify this message as a "
                        "disaster signal."
                    )

                st.markdown(
                    f"**Disaster Probability: "
                    f"{disaster_percent:.2f}%**"
                )

                st.progress(
                    float(disaster_probability)
                )

            st.write("")

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Disaster Probability",
                    f"{disaster_percent:.2f}%"
                )

            with c2:

                st.metric(
                    "Non-Disaster Probability",
                    f"{non_disaster_percent:.2f}%"
                )

            with st.expander(
                "View Processing Details"
            ):

                st.markdown(
                    "**Original Message**"
                )

                st.write(
                    message
                )

                st.markdown(
                    "**Cleaned Representation**"
                )

                st.code(
                    clean_text(message)
                )

            st.warning(
                """
                The displayed probability is a model estimate,
                not certainty. Potential disaster signals
                should be independently verified by a human.
                """
            )


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

elif mode == "Incident Intelligence":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.caption(
            "BATCH SCREENING + GENERATIVE AI"
        )

        st.title(
            "Incident Intelligence"
        )

        st.markdown(
            """
            Screen multiple incoming messages with the trained
            ML classifier, prioritise potential disaster signals,
            and curate those structured outputs into an
            AI-assisted incident-screening report.
            """
        )

        st.write(
            "`ML classification first`  "
            "`GenAI curation second`  "
            "`Human verification required`"
        )

    st.write("")

    st.caption(
        "BATCH INPUT"
    )

    st.subheader(
        "Enter One Message Per Line"
    )

    batch_text = st.text_area(
        "Messages",
        height=270,
        placeholder=(
            "Massive wildfire spreading near homes, residents are evacuating.\n"
            "That concert was absolutely fire last night.\n"
            "Several homes are flooding after heavy rainfall.\n"
            "I am going shopping tomorrow.\n"
            "Earthquake shaking buildings near downtown."
        ),
        label_visibility="collapsed"
    )

    if st.button(
        "Generate Intelligence Report",
        type="primary",
        use_container_width=True
    ):

        messages = [
            line.strip()
            for line in batch_text.splitlines()
            if line.strip()
        ]

        if not messages:

            st.warning(
                "Enter at least one message."
            )

        else:

            results = []

            # ------------------------------------------------
            # ML CLASSIFICATION
            # ------------------------------------------------

            with st.spinner(
                "Running machine-learning classification..."
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

            disaster_count = sum(
                item["prediction"] == "Disaster"
                for item in results
            )

            non_disaster_count = (
                len(results)
                - disaster_count
            )

            st.write("")

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.caption(
                "CLASSIFICATION SUMMARY"
            )

            st.subheader(
                "Machine-Learning Screening Results"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Messages Analysed",
                    len(results)
                )

            with c2:

                st.metric(
                    "Potential Disasters",
                    disaster_count
                )

            with c3:

                st.metric(
                    "Non-Disaster",
                    non_disaster_count
                )

            st.write("")

            # ------------------------------------------------
            # INDIVIDUAL RESULTS
            # ------------------------------------------------

            for i, item in enumerate(
                results,
                start=1
            ):

                probability = (
                    item["probability"]
                )

                with st.container(
                    border=True
                ):

                    top_left, top_right = (
                        st.columns(
                            [4, 1]
                        )
                    )

                    with top_left:

                        st.markdown(
                            f"#### Message {i}"
                        )

                        st.write(
                            item["message"]
                        )

                    with top_right:

                        st.metric(
                            "Probability",
                            f"{probability:.1f}%"
                        )

                    if (
                        item["prediction"]
                        == "Disaster"
                    ):

                        st.error(
                            "Potential Disaster"
                        )

                        if probability >= 80:

                            st.caption(
                                "Priority: HIGH — presentation "
                                "priority only"
                            )

                        else:

                            st.caption(
                                "Priority: REVIEW — presentation "
                                "priority only"
                            )

                    else:

                        st.success(
                            "Not Classified as Disaster"
                        )

                        st.caption(
                            "Priority: LOWER SIGNAL"
                        )

            st.caption(
                """
                Priority labels are a presentation layer only.
                They do not alter the Logistic Regression
                classifier or its decision threshold.
                """
            )

            st.write("")

            # ------------------------------------------------
            # AI REPORT
            # ------------------------------------------------

            st.caption(
                "AI CURATION"
            )

            st.subheader(
                "Incident-Screening Report"
            )

            try:

                with st.spinner(
                    "Curating ML results with Ollama Cloud..."
                ):

                    report = generate_ai_report(
                        results
                    )

                with st.container(
                    border=True
                ):

                    st.markdown(
                        report
                    )

                st.download_button(
                    label="Download Incident Report",
                    data=report,
                    file_name=(
                        "crisislens_incident_report.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "The AI-curated report could not be generated."
                )

                st.exception(
                    e
                )

            st.warning(
                """
                CrisisLens classifications and AI-generated
                reports are decision-support outputs only.
                Social-media reports remain unverified and
                important signals require human validation.
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CrisisLens · ML Classification + AI-Assisted Incident Curation"
)
