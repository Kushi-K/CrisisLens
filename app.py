import re
import html
import joblib
import streamlit as st
from ollama import Client


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CrisisLens",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(239,68,68,0.08), transparent 30%),
            linear-gradient(180deg, #0a0f1a 0%, #0d1422 100%);
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
        letter-spacing: -0.02em;
    }

    p, label {
        color: #cbd5e1;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background: #080d16;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-brand {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-bottom: 1.4rem;
    }

    .sidebar-status {
        margin-top: 1.4rem;
        padding: 0.9rem 1rem;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 7px;
        box-shadow: 0 0 8px rgba(34,197,94,0.6);
    }


    /* ---------- HERO ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        border-radius: 20px;
        border: 1px solid #263244;
        background:
            linear-gradient(
                135deg,
                rgba(15,23,42,0.98),
                rgba(30,41,59,0.82)
            );
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        margin-bottom: 1.8rem;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(239,68,68,0.08);
        right: -80px;
        top: -110px;
    }

    .hero-kicker {
        color: #ef4444;
        font-size: 0.76rem;
        letter-spacing: 0.15em;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.6rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
    }

    .hero-copy {
        max-width: 760px;
        margin-top: 0.8rem;
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.65;
    }

    .hero-tags {
        margin-top: 1.2rem;
    }

    .hero-tag {
        display: inline-block;
        padding: 0.38rem 0.7rem;
        border: 1px solid #334155;
        border-radius: 999px;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
        font-size: 0.75rem;
        color: #cbd5e1;
        background: rgba(15,23,42,0.65);
    }


    /* ---------- SECTION HEADERS ---------- */

    .section-kicker {
        color: #64748b;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        min-height: 140px;
        background: rgba(15,23,42,0.9);
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 1.25rem 1.2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.45rem;
    }

    .metric-note {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }


    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: rgba(15,23,42,0.82);
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 1.4rem;
        min-height: 210px;
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
        padding: 0.5rem 0;
        border-bottom: 1px solid #1e293b;
        color: #94a3b8;
        font-size: 0.86rem;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-value {
        color: #e2e8f0;
        font-weight: 600;
        text-align: right;
    }


    /* ---------- INPUTS ---------- */

    .stTextArea textarea {
        background: #0f172a !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    .stTextArea textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 1px #64748b !important;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border: none;
        border-radius: 10px;
        min-height: 45px;
        padding: 0 1.3rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: #ef4444;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #dc2626;
        transform: translateY(-1px);
    }


    /* ---------- CLASSIFICATION RESULT ---------- */

    .result-card {
        padding: 1.4rem 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        background: #0f172a;
    }

    .result-danger {
        border: 1px solid rgba(239,68,68,0.5);
        box-shadow: inset 4px 0 #ef4444;
    }

    .result-safe {
        border: 1px solid rgba(34,197,94,0.35);
        box-shadow: inset 4px 0 #22c55e;
    }

    .result-status {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 800;
        color: #64748b;
    }

    .result-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.35rem;
        color: #f8fafc;
    }

    .result-description {
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .probability-bar {
        height: 8px;
        border-radius: 999px;
        overflow: hidden;
        background: #1e293b;
        margin-top: 0.8rem;
    }

    .probability-fill-danger {
        height: 100%;
        background: #ef4444;
        border-radius: 999px;
    }

    .probability-fill-safe {
        height: 100%;
        background: #22c55e;
        border-radius: 999px;
    }


    /* ---------- BATCH RESULT ---------- */

    .batch-card {
        padding: 1rem 1.1rem;
        margin-bottom: 0.7rem;
        background: #0f172a;
        border: 1px solid #263244;
        border-radius: 13px;
    }

    .batch-message {
        color: #e2e8f0;
        font-size: 0.92rem;
        margin-bottom: 0.55rem;
    }

    .badge-danger,
    .badge-safe,
    .badge-review {
        display: inline-block;
        padding: 0.27rem 0.6rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 0.4rem;
    }

    .badge-danger {
        background: rgba(239,68,68,0.12);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.28);
    }

    .badge-safe {
        background: rgba(34,197,94,0.10);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.25);
    }

    .badge-review {
        background: rgba(245,158,11,0.10);
        color: #fbbf24;
        border: 1px solid rgba(245,158,11,0.25);
    }

    .prob-text {
        color: #94a3b8;
        font-size: 0.78rem;
    }


    /* ---------- DISCLAIMER ---------- */

    .disclaimer {
        margin-top: 1.5rem;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #263244;
        background: rgba(15,23,42,0.7);
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.55;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        margin-top: 3rem;
        border-top: 1px solid #1e293b;
        padding-top: 1rem;
        color: #475569;
        font-size: 0.72rem;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED ML COMPONENTS
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
# TEXT PREPROCESSING
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
# ML INFERENCE
# ============================================================

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

- Do not assume messages were posted at the same time or came
  from the same geographic area.

- Do not invent relationships between messages.

- Do not treat probability as certainty.

- Avoid adding scientific hazard categories unless they are
  explicitly stated in the message.

- Recommended Human Review should provide only general
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
# REUSABLE UI FUNCTIONS
# ============================================================

def metric_card(label, value, note):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(kicker, title):

    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True
    )


def footer():

    st.markdown(
        """
        <div class="footer">
            CrisisLens · Machine Learning Classification +
            AI-Assisted Incident Curation
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">CRISISLENS</div>
        <div class="sidebar-subtitle">
            Disaster intelligence and incident-screening platform
        </div>
        """,
        unsafe_allow_html=True
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

    st.markdown(
        """
        <div class="sidebar-status">
            <div style="font-size:0.72rem;color:#64748b;
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                        margin-bottom:0.5rem;">
                System
            </div>

            <div style="font-size:0.83rem;color:#cbd5e1;">
                <span class="status-dot"></span>
                Operational
            </div>

            <div style="margin-top:0.9rem;
                        font-size:0.72rem;
                        color:#64748b;">
                CLASSIFIER
            </div>

            <div style="font-size:0.83rem;
                        color:#e2e8f0;
                        margin-top:0.2rem;">
                Logistic Regression · C=5.0
            </div>

            <div style="margin-top:0.8rem;
                        font-size:0.72rem;
                        color:#64748b;">
                FEATURE LAYER
            </div>

            <div style="font-size:0.83rem;
                        color:#e2e8f0;
                        margin-top:0.2rem;">
                TF-IDF · 1–2 grams
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OVERVIEW
# ============================================================

if mode == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">DISASTER INTELLIGENCE PLATFORM</div>

            <div class="hero-title">
                CrisisLens
            </div>

            <div class="hero-copy">
                Machine-learning driven disaster-message screening
                with downstream AI-assisted incident curation.
                Designed to transform noisy social-media signals into
                structured, reviewable intelligence.
            </div>

            <div class="hero-tags">
                <span class="hero-tag">Supervised ML</span>
                <span class="hero-tag">TF-IDF</span>
                <span class="hero-tag">Logistic Regression</span>
                <span class="hero-tag">Ollama Cloud</span>
                <span class="hero-tag">Human-in-the-loop</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_header(
        "MODEL PERFORMANCE",
        "Holdout evaluation"
    )

    col1, col2, col3, col4 = st.columns(4)

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
            "Balance of precision and recall"
        )

    section_header(
        "SYSTEM PROFILE",
        "Model and dataset configuration"
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
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
                    <span class="info-value">TF-IDF</span>
                </div>

                <div class="info-row">
                    <span>N-gram range</span>
                    <span class="info-value">(1, 2)</span>
                </div>

                <div class="info-row">
                    <span>Maximum features</span>
                    <span class="info-value">10,000</span>
                </div>

                <div class="info-row">
                    <span>Minimum document frequency</span>
                    <span class="info-value">2</span>
                </div>

                <div class="info-row">
                    <span>Best C</span>
                    <span class="info-value">5.0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    Training & Validation
                </div>

                <div class="info-row">
                    <span>Original samples</span>
                    <span class="info-value">7,613</span>
                </div>

                <div class="info-row">
                    <span>Cleaned samples</span>
                    <span class="info-value">7,485</span>
                </div>

                <div class="info-row">
                    <span>Training split</span>
                    <span class="info-value">80%</span>
                </div>

                <div class="info-row">
                    <span>Holdout split</span>
                    <span class="info-value">20%</span>
                </div>

                <div class="info-row">
                    <span>Model selection</span>
                    <span class="info-value">GridSearchCV</span>
                </div>

                <div class="info-row">
                    <span>Best CV F1</span>
                    <span class="info-value">75.54%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    section_header(
        "PIPELINE",
        "From raw message to curated intelligence"
    )

    st.code(
        """
Incoming Message
      ↓
Text Preprocessing
      ↓
TF-IDF Feature Vector
      ↓
Tuned Logistic Regression
      ↓
Disaster / Not Disaster
      ↓
Probability Estimate
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

    st.markdown(
        """
        <div class="disclaimer">
            CrisisLens is a decision-support prototype.
            Model predictions and social-media reports are not
            verified emergency information and require human review.
        </div>
        """,
        unsafe_allow_html=True
    )

    footer()


# ============================================================
# MESSAGE ANALYSIS
# ============================================================

elif mode == "Message Analysis":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">SINGLE MESSAGE SCREENING</div>

            <div class="hero-title">
                Message Analysis
            </div>

            <div class="hero-copy">
                Screen an individual social-media message using the
                trained CrisisLens classifier and inspect the model's
                estimated disaster probability.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_header(
        "INPUT",
        "Analyse a social-media message"
    )

    message = st.text_area(
        "Message",
        placeholder=(
            "Example: Massive wildfire spreading near "
            "homes, residents are evacuating."
        ),
        height=170,
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

                result_class = "result-danger"
                fill_class = "probability-fill-danger"
                title = "Potential Disaster"
                description = (
                    "The model classified this message as a "
                    "potential disaster signal."
                )

            else:

                result_class = "result-safe"
                fill_class = "probability-fill-safe"
                title = "Not Classified as Disaster"
                description = (
                    "The model did not classify this message "
                    "as a disaster signal."
                )

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-status">
                        Classification Result
                    </div>

                    <div class="result-title">
                        {title}
                    </div>

                    <div class="result-description">
                        {description}
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-top:1.1rem;
                        color:#cbd5e1;
                        font-size:0.83rem;
                    ">
                        <span>Disaster probability</span>
                        <strong>{disaster_percent:.2f}%</strong>
                    </div>

                    <div class="probability-bar">
                        <div
                            class="{fill_class}"
                            style="width:{disaster_percent:.2f}%;">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                metric_card(
                    "Disaster Probability",
                    f"{disaster_percent:.2f}%",
                    "Estimated probability for class 1"
                )

            with col2:

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

            st.markdown(
                """
                <div class="disclaimer">
                    Classification probability is a model estimate,
                    not certainty. Potential disaster signals should
                    be independently verified before action.
                </div>
                """,
                unsafe_allow_html=True
            )

    footer()


# ============================================================
# INCIDENT INTELLIGENCE
# ============================================================

elif mode == "Incident Intelligence":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">BATCH SCREENING + GENAI</div>

            <div class="hero-title">
                Incident Intelligence
            </div>

            <div class="hero-copy">
                Analyse multiple messages with the trained ML
                classifier, prioritise potential disaster signals,
                and curate the structured results into an
                AI-assisted incident-screening report.
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
        """,
        unsafe_allow_html=True
    )

    section_header(
        "BATCH INPUT",
        "Paste one social-media message per line"
    )

    batch_text = st.text_area(
        "Messages",
        height=260,
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
                "Running ML classification..."
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
                len(results) - disaster_count
            )

            section_header(
                "CLASSIFICATION SUMMARY",
                "Machine-learning screening results"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                metric_card(
                    "Messages Analysed",
                    str(len(results)),
                    "Total incoming messages"
                )

            with c2:

                metric_card(
                    "Potential Disasters",
                    str(disaster_count),
                    "ML-classified disaster signals"
                )

            with c3:

                metric_card(
                    "Non-Disaster",
                    str(non_disaster_count),
                    "Messages outside disaster class"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            for i, item in enumerate(
                results,
                start=1
            ):

                safe_message = html.escape(
                    item["message"]
                )

                probability = item["probability"]

                if item["prediction"] == "Disaster":

                    if probability >= 80:
                        priority = "High Priority"
                        badge = "badge-danger"
                    else:
                        priority = "Review"
                        badge = "badge-review"

                else:
                    priority = "Lower Signal"
                    badge = "badge-safe"

                classification_badge = (
                    "badge-danger"
                    if item["prediction"] == "Disaster"
                    else "badge-safe"
                )

                st.markdown(
                    f"""
                    <div class="batch-card">

                        <div class="batch-message">
                            <strong>#{i}</strong>
                            &nbsp; {safe_message}
                        </div>

                        <span class="{classification_badge}">
                            {item['prediction']}
                        </span>

                        <span class="{badge}">
                            {priority}
                        </span>

                        <span class="prob-text">
                            Disaster probability:
                            {probability:.2f}%
                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption(
                "Priority badges are a presentation layer only "
                "and do not change the classifier's decision."
            )

            section_header(
                "AI CURATION",
                "Incident-screening report"
            )

            try:

                with st.spinner(
                    "Curating the ML results with Ollama Cloud..."
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

            st.markdown(
                """
                <div class="disclaimer">
                    CrisisLens classifications and AI-generated
                    reports are decision-support outputs only.
                    Social-media reports remain unverified.
                    A human reviewer should validate important
                    signals using authoritative information.
                </div>
                """,
                unsafe_allow_html=True
            )

    footer()
