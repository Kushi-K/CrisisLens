import re
import html
import textwrap
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ==========================
       GLOBAL APP
       ========================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(239, 68, 68, 0.07),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                #090e18 0%,
                #0b1220 100%
            );
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    p {
        color: #cbd5e1;
    }


    /* ==========================
       SIDEBAR
       ========================== */

    [data-testid="stSidebar"] {
        background: #070c14;
        border-right: 1px solid #1e293b;
    }

    .sidebar-brand {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.55;
        margin-bottom: 1.4rem;
    }

    .sidebar-panel {
        margin-top: 1.5rem;
        padding: 1rem;
        border-radius: 14px;
        background: #0f172a;
        border: 1px solid #1e293b;
    }

    .sidebar-label {
        color: #64748b;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sidebar-value {
        color: #e2e8f0;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 7px;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.55);
    }


    /* ==========================
       HERO
       ========================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.3rem 2.5rem;
        border-radius: 20px;
        border: 1px solid #263244;
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.98),
                rgba(30, 41, 59, 0.88)
            );
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
        margin-bottom: 2rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -90px;
        top: -120px;
        border-radius: 50%;
        background: rgba(239, 68, 68, 0.08);
    }

    .hero-kicker {
        color: #f87171;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.8rem;
    }

    .hero-copy {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 780px;
    }

    .hero-tags {
        margin-top: 1.25rem;
    }

    .hero-tag {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        border-radius: 999px;
        border: 1px solid #334155;
        background: rgba(15, 23, 42, 0.65);
        color: #cbd5e1;
        font-size: 0.72rem;
    }


    /* ==========================
       SECTION TITLES
       ========================== */

    .section-kicker {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 1rem;
    }


    /* ==========================
       METRIC CARDS
       ========================== */

    .metric-card {
        min-height: 135px;
        padding: 1.2rem;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid #263244;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.16);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.45rem;
    }

    .metric-note {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 0.25rem;
        line-height: 1.4;
    }


    /* ==========================
       INFORMATION CARDS
       ========================== */

    .info-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 1.4rem;
        min-height: 330px;
    }

    .info-card-title {
        color: #f1f5f9;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid #1e293b;
        color: #94a3b8;
        font-size: 0.84rem;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-value {
        color: #e2e8f0;
        font-weight: 600;
        text-align: right;
    }


    /* ==========================
       INPUT BOXES
       ========================== */

    .stTextArea textarea {
        background: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 13px !important;
    }

    .stTextArea textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 1px #64748b !important;
    }


    /* ==========================
       BUTTONS
       ========================== */

    div[data-testid="stButton"] button {
        border-radius: 11px;
        min-height: 46px;
        font-weight: 700;
    }


    /* ==========================
       SINGLE RESULT
       ========================== */

    .result-card {
        padding: 1.4rem 1.5rem;
        border-radius: 16px;
        margin-top: 1.2rem;
        margin-bottom: 1rem;
        background: #0f172a;
    }

    .result-danger {
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: inset 4px 0 #ef4444;
    }

    .result-safe {
        border: 1px solid rgba(34, 197, 94, 0.35);
        box-shadow: inset 4px 0 #22c55e;
    }

    .result-label {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .result-title {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .result-copy {
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .probability-bar {
        background: #1e293b;
        height: 8px;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 0.7rem;
    }

    .probability-danger {
        height: 100%;
        background: #ef4444;
        border-radius: 999px;
    }

    .probability-safe {
        height: 100%;
        background: #22c55e;
        border-radius: 999px;
    }


    /* ==========================
       BATCH RESULT CARDS
       ========================== */

    .batch-card {
        padding: 1rem 1.1rem;
        border-radius: 13px;
        background: #0f172a;
        border: 1px solid #263244;
        margin-bottom: 0.75rem;
    }

    .batch-message {
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
        line-height: 1.5;
    }

    .badge-danger,
    .badge-safe,
    .badge-review {
        display: inline-block;
        padding: 0.25rem 0.58rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-danger {
        color: #f87171;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.28);
    }

    .badge-safe {
        color: #4ade80;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.24);
    }

    .badge-review {
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .prob-text {
        color: #94a3b8;
        font-size: 0.75rem;
    }


    /* ==========================
       DISCLAIMER + FOOTER
       ========================== */

    .disclaimer {
        margin-top: 1.5rem;
        padding: 1rem 1.15rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid #263244;
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.6;
    }

    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #1e293b;
        color: #475569;
        font-size: 0.7rem;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAFE HTML RENDERING
# ============================================================

def render_html(content):

    cleaned_html = textwrap.dedent(
        content
    ).strip()

    st.markdown(
        cleaned_html,
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
# OLLAMA CLOUD
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

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREDICTION
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
# AI INCIDENT REPORT
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
  Non-Disaster Messages.

- Do not reclassify any message.

- Rank Disaster messages by their supplied disaster probability.

- Treat every message as an unverified social-media report.

- Do not state that any disaster definitely occurred.

- Do not invent dates, timestamps, locations, casualties,
  organizations, teams, signatories, infrastructure damage,
  authorities, or other facts.

- Do not assume the messages came from the same time period
  or geographic area.

- Do not invent relationships between separate messages.

- Do not call probability a certainty.

- Avoid adding hazard categories unless directly supported
  by the message.

- Recommended Human Review should contain only general
  verification guidance.

- Clearly state that ML systems can produce false positives
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


# ============================================================
# UI HELPERS
# ============================================================

def section_header(kicker, title):

    render_html(
        f"""
        <div class="section-kicker">
            {kicker}
        </div>

        <div class="section-title">
            {title}
        </div>
        """
    )


def metric_card(label, value, note):

    render_html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-note">
                {note}
            </div>

        </div>
        """
    )


def show_footer():

    render_html(
        """
        <div class="footer">
            CrisisLens · ML Classification + AI-Assisted Incident Curation
        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            CRISISLENS
        </div>

        <div class="sidebar-subtitle">
            Disaster intelligence and incident-screening platform
        </div>
        """
    )

    mode = st.radio(
        "Navigation",
        [
            "Overview",
            "Message Analysis",
            "Incident Intelligence"
        ],
        label_visibility="collapsed"
    )

    render_html(
        """
        <div class="sidebar-panel">

            <div class="sidebar-label">
                System
            </div>

            <div class="sidebar-value">
                <span class="status-dot"></span>
                Operational
            </div>

            <div class="sidebar-label">
                Classifier
            </div>

            <div class="sidebar-value">
                Logistic Regression · C = 5.0
            </div>

            <div class="sidebar-label">
                Feature Layer
            </div>

            <div class="sidebar-value">
                TF-IDF · Unigrams + Bigrams
            </div>

            <div class="sidebar-label">
                AI Curation
            </div>

            <div class="sidebar-value">
                Ollama Cloud
            </div>

        </div>
        """
    )


# ============================================================
# OVERVIEW
# ============================================================

if mode == "Overview":

    render_html(
        """
        <div class="hero">

            <div class="hero-kicker">
                Disaster Intelligence Platform
            </div>

            <div class="hero-title">
                CrisisLens
            </div>

            <div class="hero-copy">
                Machine-learning driven disaster-message screening
                with downstream AI-assisted incident curation.
                CrisisLens transforms noisy social-media signals
                into structured, reviewable intelligence for
                human verification.
            </div>

            <div class="hero-tags">

                <span class="hero-tag">
                    Supervised ML
                </span>

                <span class="hero-tag">
                    TF-IDF
                </span>

                <span class="hero-tag">
                    Logistic Regression
                </span>

                <span class="hero-tag">
                    Ollama Cloud
                </span>

                <span class="hero-tag">
                    Human-in-the-loop
                </span>

            </div>

        </div>
        """
    )

    section_header(
        "Model Performance",
        "Holdout evaluation"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        metric_card(
            "Accuracy",
            "78.82%",
            "Overall correct classifications"
        )

    with col2:

        metric_card(
            "Precision",
            "80.23%",
            "Reliability of disaster predictions"
        )

    with col3:

        metric_card(
            "Recall",
            "66.77%",
            "Actual disaster messages detected"
        )

    with col4:

        metric_card(
            "F1 Score",
            "72.88%",
            "Precision-recall balance"
        )

    section_header(
        "System Profile",
        "Model and training configuration"
    )

    left, right = st.columns(2)

    with left:

        render_html(
            """
            <div class="info-card">

                <div class="info-card-title">
                    Model Configuration
                </div>

                <div class="info-row">
                    <span>Algorithm</span>
                    <span class="info-value">
                        Logistic Regression
                    </span>
                </div>

                <div class="info-row">
                    <span>Feature extraction</span>
                    <span class="info-value">
                        TF-IDF
                    </span>
                </div>

                <div class="info-row">
                    <span>N-gram range</span>
                    <span class="info-value">
                        (1, 2)
                    </span>
                </div>

                <div class="info-row">
                    <span>Maximum features</span>
                    <span class="info-value">
                        10,000
                    </span>
                </div>

                <div class="info-row">
                    <span>Minimum document frequency</span>
                    <span class="info-value">
                        2
                    </span>
                </div>

                <div class="info-row">
                    <span>Best C</span>
                    <span class="info-value">
                        5.0
                    </span>
                </div>

            </div>
            """
        )

    with right:

        render_html(
            """
            <div class="info-card">

                <div class="info-card-title">
                    Training & Validation
                </div>

                <div class="info-row">
                    <span>Original rows</span>
                    <span class="info-value">
                        7,613
                    </span>
                </div>

                <div class="info-row">
                    <span>Clean samples</span>
                    <span class="info-value">
                        7,485
                    </span>
                </div>

                <div class="info-row">
                    <span>Training split</span>
                    <span class="info-value">
                        80%
                    </span>
                </div>

                <div class="info-row">
                    <span>Holdout split</span>
                    <span class="info-value">
                        20%
                    </span>
                </div>

                <div class="info-row">
                    <span>Hyperparameter search</span>
                    <span class="info-value">
                        GridSearchCV
                    </span>
                </div>

                <div class="info-row">
                    <span>Cross-validation</span>
                    <span class="info-value">
                        5 folds
                    </span>
                </div>

                <div class="info-row">
                    <span>Best CV F1</span>
                    <span class="info-value">
                        75.54%
                    </span>
                </div>

            </div>
            """
        )

    section_header(
        "Architecture",
        "From message to actionable review"
    )

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

    render_html(
        """
        <div class="disclaimer">
            CrisisLens is a decision-support prototype.
            Model predictions and social-media reports are
            unverified and require human review before any
            real-world action is taken.
        </div>
        """
    )

    show_footer()


# ============================================================
# MESSAGE ANALYSIS
# ============================================================

elif mode == "Message Analysis":

    render_html(
        """
        <div class="hero">

            <div class="hero-kicker">
                Single Message Screening
            </div>

            <div class="hero-title">
                Message Analysis
            </div>

            <div class="hero-copy">
                Analyse an individual social-media message using
                the trained CrisisLens classifier and inspect the
                model's estimated disaster probability.
            </div>

        </div>
        """
    )

    section_header(
        "Input",
        "Analyse a social-media message"
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

            if prediction == 1:

                card_class = "result-danger"
                bar_class = "probability-danger"
                title = "Potential Disaster"

                description = (
                    "The trained classifier identified this "
                    "message as a potential disaster signal."
                )

            else:

                card_class = "result-safe"
                bar_class = "probability-safe"
                title = "Not Classified as Disaster"

                description = (
                    "The trained classifier did not identify "
                    "this message as a disaster signal."
                )

            render_html(
                f"""
                <div class="result-card {card_class}">

                    <div class="result-label">
                        Classification Result
                    </div>

                    <div class="result-title">
                        {title}
                    </div>

                    <div class="result-copy">
                        {description}
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-top:1.1rem;
                        color:#cbd5e1;
                        font-size:0.82rem;
                    ">
                        <span>
                            Disaster probability
                        </span>

                        <strong>
                            {disaster_percent:.2f}%
                        </strong>
                    </div>

                    <div class="probability-bar">

                        <div
                            class="{bar_class}"
                            style="width:{disaster_percent:.2f}%;">
                        </div>

                    </div>

                </div>
                """
            )

            c1, c2 = st.columns(2)

            with c1:

                metric_card(
                    "Disaster Probability",
                    f"{disaster_percent:.2f}%",
                    "Estimated probability for class 1"
                )

            with c2:

                metric_card(
                    "Non-Disaster Probability",
                    f"{non_disaster_percent:.2f}%",
                    "Estimated probability for class 0"
                )

            with st.expander(
                "Processing details"
            ):

                st.write(
                    "**Original message**"
                )

                st.write(
                    message
                )

                st.write(
                    "**Cleaned representation**"
                )

                st.code(
                    clean_text(message)
                )

            render_html(
                """
                <div class="disclaimer">
                    Classification probability is a model estimate,
                    not a certainty. Potential disaster signals
                    should be independently verified.
                </div>
                """
            )

    show_footer()


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

elif mode == "Incident Intelligence":

    render_html(
        """
        <div class="hero">

            <div class="hero-kicker">
                Batch Screening + Generative AI
            </div>

            <div class="hero-title">
                Incident Intelligence
            </div>

            <div class="hero-copy">
                Screen multiple incoming messages using the
                trained ML classifier, prioritise potential
                disaster signals, and curate the structured
                outputs into an AI-assisted incident report.
            </div>

            <div class="hero-tags">

                <span class="hero-tag">
                    ML classification first
                </span>

                <span class="hero-tag">
                    GenAI curation second
                </span>

                <span class="hero-tag">
                    Human verification required
                </span>

            </div>

        </div>
        """
    )

    section_header(
        "Batch Input",
        "Enter one message per line"
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

            section_header(
                "Classification Summary",
                "Machine-learning screening results"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                metric_card(
                    "Messages Analysed",
                    str(len(results)),
                    "Total submitted messages"
                )

            with c2:

                metric_card(
                    "Potential Disasters",
                    str(disaster_count),
                    "Messages classified as Disaster"
                )

            with c3:

                metric_card(
                    "Non-Disaster",
                    str(non_disaster_count),
                    "Messages outside the disaster class"
                )

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            for i, item in enumerate(
                results,
                start=1
            ):

                safe_message = html.escape(
                    item["message"]
                )

                probability = (
                    item["probability"]
                )

                if (
                    item["prediction"]
                    == "Disaster"
                ):

                    classification_class = (
                        "badge-danger"
                    )

                    if probability >= 80:

                        priority_label = (
                            "High Priority"
                        )

                        priority_class = (
                            "badge-danger"
                        )

                    else:

                        priority_label = (
                            "Review"
                        )

                        priority_class = (
                            "badge-review"
                        )

                else:

                    classification_class = (
                        "badge-safe"
                    )

                    priority_label = (
                        "Lower Signal"
                    )

                    priority_class = (
                        "badge-safe"
                    )

                render_html(
                    f"""
                    <div class="batch-card">

                        <div class="batch-message">
                            <strong>#{i}</strong>
                            &nbsp;
                            {safe_message}
                        </div>

                        <span class="{classification_class}">
                            {item["prediction"]}
                        </span>

                        <span class="{priority_class}">
                            {priority_label}
                        </span>

                        <span class="prob-text">
                            Disaster probability:
                            {probability:.2f}%
                        </span>

                    </div>
                    """
                )

            st.caption(
                "Priority badges are only a presentation layer. "
                "They do not alter the ML classifier's prediction."
            )

            section_header(
                "AI Curation",
                "Incident-screening report"
            )

            try:

                with st.spinner(
                    "Curating classification results with Ollama Cloud..."
                ):

                    report = generate_ai_report(
                        results
                    )

                st.markdown(
                    report
                )

                st.download_button(
                    label="Download Incident Report",
                    data=report,
                    file_name="crisislens_incident_report.txt",
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

            render_html(
                """
                <div class="disclaimer">
                    CrisisLens classifications and AI-generated
                    reports are decision-support outputs only.
                    Social-media reports remain unverified.
                    Important signals should be validated by a
                    human using authoritative sources.
                </div>
                """
            )

    show_footer()
