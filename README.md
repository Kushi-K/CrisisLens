# CrisisLens

### Disaster Message Classification & AI-Assisted Incident Intelligence

CrisisLens is an end-to-end AIML application designed to screen social-media messages for potential disaster-related content.

The system combines a **traditional supervised machine-learning classifier** with a downstream **Generative AI reporting layer**:

- **TF-IDF + Logistic Regression** performs disaster classification.
- **Ollama Cloud (`gpt-oss:20b-cloud`)** curates multiple ML classification results into a structured incident-screening report.
- All predictions and generated reports are designed for **human review**, not automatic emergency decision-making.

## Live Demo

**CrisisLens Web App:**  
https://crisislens-7zeotwfigvpxkuootzzh4b.streamlit.app/

---

## Problem Statement

During disasters and emergencies, large volumes of social-media posts may appear within a short period of time. Some messages describe genuine incidents, while others contain unrelated, figurative, or ambiguous language.

Manually reviewing every message can be inefficient.

CrisisLens addresses this problem by:

1. Classifying social-media messages as **Disaster** or **Not Disaster**.
2. Estimating the model's disaster probability.
3. Screening multiple messages in batches.
4. Using Generative AI to curate ML outputs into a structured incident report.
5. Keeping a human reviewer in the decision loop.

---

## System Architecture

```text
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
```

A key design principle of CrisisLens is the separation between **classification** and **report generation**.

The LLM does **not** perform the disaster classification.

The trained ML classifier first produces the prediction and probability. Generative AI is then used only to organize and summarize those structured outputs.

---

## Dataset

The model was trained using Kaggle's:

**Natural Language Processing with Disaster Tweets**

The original labelled training dataset contains:

- **7,613 rows**
- Binary target:
  - `0` — Not Disaster
  - `1` — Disaster

### Initial class distribution

| Class | Samples |
|---|---:|
| Not Disaster | 4,342 |
| Disaster | 3,271 |

The dataset is reasonably balanced, with approximately:

- **57% Not Disaster**
- **43% Disaster**

### Data Quality Analysis

The dataset contained:

- 110 duplicate tweet texts
- 18 unique tweet texts with conflicting labels
- Missing values mainly in `location`
- No missing values in `text` or `target`

After removing conflicting text-label samples and duplicate text entries:

**Final cleaned dataset: 7,485 samples**

---

## Text Preprocessing

The preprocessing pipeline performs:

- Lowercase conversion
- URL removal
- Twitter mention removal
- Hashtag-symbol removal while retaining the hashtag word
- Removal of non-alphabetical symbols
- Extra whitespace removal

Example:

```text
Original:
Massive #WILDFIRE!!! https://example.com

Cleaned:
massive wildfire
```

The same preprocessing logic is used during both training and inference to maintain **training-serving consistency**.

---

## Feature Engineering

CrisisLens uses **TF-IDF — Term Frequency–Inverse Document Frequency** to convert text into numerical features.

Configuration:

| Hyperparameter | Value |
|---|---|
| `ngram_range` | `(1, 2)` |
| `min_df` | `2` |
| `max_features` | `10,000` |

Using `(1, 2)` enables both:

- Unigrams such as `fire`
- Bigrams such as `forest fire`

TF-IDF is fitted only on the training data to avoid **data leakage**.

---

## Train/Test Split

The cleaned dataset was split into:

- **80% training data**
- **20% holdout test data**

Configuration:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` preserves approximately the same class distribution in both training and testing data.

---

## Models Compared

Three classical text-classification algorithms were evaluated:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 79.4% | 83.2% | 64.6% | 72.7% |
| Linear SVM | 77.8% | 77.9% | 66.9% | 72.0% |
| Multinomial Naive Bayes | 79.2% | 87.2% | 60.0% | 71.1% |

Logistic Regression produced the strongest baseline F1 score, while Linear SVM achieved slightly higher recall and Naive Bayes achieved the highest precision.

---

## Hyperparameter Tuning

Hyperparameters were optimized using:

**GridSearchCV + 5-Fold Cross-Validation**

The optimization metric was:

**F1 Score**

### Search results

| Model | Best Hyperparameter | Cross-Validation F1 |
|---|---|---:|
| Logistic Regression | `C = 5.0` | **0.7554** |
| Linear SVM | `C = 0.5` | 0.7535 |
| Multinomial Naive Bayes | `alpha = 0.5` | 0.7409 |

The tuned Logistic Regression model was selected as the final classifier.

---

## Final Model Performance

### Tuned Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | **78.82%** |
| Precision | **80.23%** |
| Recall | **66.77%** |
| F1 Score | **72.88%** |
| Best 5-Fold CV F1 | **75.54%** |

Although accuracy decreased slightly compared with the untuned baseline, recall and F1 improved.

For a disaster-screening system, identifying more genuine disaster messages is an important consideration, so the final model was selected based on the balance between **precision, recall and F1**, rather than accuracy alone.

---

## Why Logistic Regression?

Logistic Regression was selected because:

- It achieved the strongest tuned F1 score among the compared models.
- It provided competitive recall.
- It performs well with high-dimensional sparse TF-IDF features.
- It is computationally efficient.
- It is interpretable compared with more complex models.
- It supports `predict_proba()`, allowing CrisisLens to display estimated disaster probabilities.

---

## Parameters vs Hyperparameters

### Learned model parameters

These are learned automatically from the training data:

- Logistic Regression coefficients / weights
- Intercept

### Configured hyperparameters

These are selected by the developer or tuning process:

- Logistic Regression `C`
- `max_iter`
- TF-IDF `ngram_range`
- TF-IDF `min_df`
- TF-IDF `max_features`
- Naive Bayes `alpha`
- SVM `C`
- Cross-validation folds

For the final Logistic Regression model:

```text
C = 5.0
max_iter = 1000
```

`C` controls inverse regularization strength.

- Smaller `C` → stronger regularization
- Larger `C` → weaker regularization

---

## Application Features

### Overview

Provides:

- Final model metrics
- Model configuration
- Dataset information
- Training configuration
- System architecture

### Message Analysis

Allows a user to submit a single social-media message.

The application displays:

- Disaster / Not Disaster classification
- Disaster probability
- Non-disaster probability
- Cleaned text representation
- Human-verification warning

Example:

```text
Input:
Massive wildfire spreading near homes, residents are evacuating.

Output:
Potential Disaster
Disaster Probability ≈ 98%
```

### Incident Intelligence

Allows batch screening of multiple social-media messages.

For each message, CrisisLens provides:

- ML classification
- Disaster probability
- Presentation-level review priority

The outputs are then passed to the Generative AI reporting layer.

---

## Generative AI Report Curation

CrisisLens integrates **Ollama Cloud** using:

```text
gpt-oss:20b-cloud
```

The LLM receives structured ML outputs containing:

- Original message
- ML classification
- Disaster probability

It then produces an incident-screening report containing:

1. Executive Summary
2. Potential Incident Signals
3. Highest-Priority Messages
4. Non-Disaster Messages
5. Common Themes
6. Recommended Human Review
7. Limitations

The prompt explicitly instructs the LLM:

- Not to override ML classifications
- Not to invent dates, locations or casualties
- Not to treat probabilities as certainty
- To treat social-media messages as unverified
- To recommend human verification
- To acknowledge false positives and false negatives

---

## Example End-to-End Input

```text
Massive wildfire spreading near homes, residents are evacuating.
That concert was absolutely fire last night.
Several homes are flooding after heavy rainfall.
I am going shopping tomorrow.
Earthquake shaking buildings near downtown.
```

Typical ML screening:

```text
Wildfire      → Disaster
Concert       → Not Disaster
Flooding      → Disaster
Shopping      → Not Disaster
Earthquake    → Disaster
```

The resulting classifications are then curated into the AI-assisted incident report.

---

## Model Limitations

CrisisLens is intentionally presented as a **decision-support prototype**, not an emergency verification system.

Known limitations include:

- TF-IDF does not deeply understand semantic context.
- Figurative language may create false positives.
- Indirect descriptions of emergencies may create false negatives.
- The training dataset is primarily English social-media text.
- The system does not verify real-world events.
- Probability estimates are not guarantees of correctness.
- Social-media data may contain misinformation or ambiguous language.

Example:

```text
"Best movie ever, that scene was explosive."
```

may be misclassified because words such as `explosive` are statistically associated with disaster-related language.

---

## Future Improvements

Potential extensions include:

- Decision-threshold tuning to optimize recall
- Class-weight experimentation
- Character-level TF-IDF features
- Additional n-gram configurations
- Larger and more diverse training datasets
- Geolocation extraction
- Timestamp-based grouping
- Multilingual classification
- Transformer models such as BERT
- Model probability calibration
- Explainability using feature importance
- Real-time social-media stream integration
- Incident clustering and duplicate-event detection

Any more complex model should still be evaluated objectively using metrics such as precision, recall and F1 rather than assuming that higher complexity guarantees better performance.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Feature Engineering | TF-IDF |
| Final Classifier | Logistic Regression |
| Hyperparameter Tuning | GridSearchCV |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Generative AI | Ollama Cloud |
| LLM | `gpt-oss:20b-cloud` |
| Training Environment | Google Colab |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

## Project Structure

```text
CrisisLens/
│
├── app.py
├── crisislens_model.pkl
├── crisislens_tfidf.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

Model experimentation and training were performed separately in:

```text
CrisisLens_Model_Training.ipynb
```

This separation prevents model retraining every time the web application starts.

---

## Installation

Python 3.13 is recommended for compatibility with the serialized model.

Clone the repository:

```bash
git clone <your-repository-url>
cd CrisisLens
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an Ollama Cloud API key and configure:

```toml
OLLAMA_API_KEY = "your_api_key"
```

For Streamlit Community Cloud, store this under **App Secrets** rather than committing credentials to GitHub.

Run the application:

```bash
streamlit run app.py
```

---

## Requirements

```text
streamlit
numpy==2.1.3
scipy==1.16.3
scikit-learn==1.6.1
joblib==1.5.3
ollama
```

---

## Security

The Ollama API key is stored using Streamlit Secrets.

Sensitive credentials are **not committed to GitHub**.

The `.gitignore` excludes local secrets and environment files such as:

```text
.env
.venv/
__pycache__/
.DS_Store
```

---

## Responsible Use

CrisisLens predictions and AI-generated reports are **decision-support outputs only**.

They should not be interpreted as verified emergency information.

Potential disaster messages should be independently reviewed using authoritative information before any real-world decision or action is taken.

---

## Author

**Kushi K**

BE Computer Science and Engineering

---

## Project Status

**Completed and deployed**

Live application:

https://crisislens-7zeotwfigvpxkuootzzh4b.streamlit.app/
