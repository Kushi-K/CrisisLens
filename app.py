import re
import joblib
import streamlit as st

st.set_page_config(
    page_title="CrisisLens",
    page_icon="🚨",
    layout="wide"
)

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

    return (
        prediction,
        disaster_probability
    )


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
            "Classify Message"
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


    # ----------------------------------------------
    # Model metrics
    # ----------------------------------------------

    st.subheader("Final Model Performance")

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
    # Model information
    # ----------------------------------------------

    st.subheader("Model Configuration")

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
    # Dataset information
    # ----------------------------------------------

    st.subheader("Dataset")

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

    st.title("🔍 Classify Message")

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

            st.subheader("Prediction Result")


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

                st.write(message)

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
