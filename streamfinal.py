
import html
import json
import re
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import f1_score

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    load_model = None
    pad_sequences = None


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"
DATASET_PATH = BASE_DIR / "News_Category_Dataset_v3.json"

CATEGORIES = [
    "TECH",
    "ENTERTAINMENT",
    "POLITICS",
    "CRIME",
    "QUEER VOICES",
    "WOMEN",
    "TRAVEL",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Linear SVM": "linear_svm.pkl",
    "LSTM": "lstm.keras",
    "GRU": "gru.keras",
}

st.set_page_config(
    page_title="The Daily Signal",
    page_icon="📰",
    layout="centered",
)


# =========================================================
# NEWSPAPER DESIGN — CONSISTENT TYPEWRITER FONT
# =========================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Special+Elite&display=swap'
    );
    @import url(
      'https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap'
    );

    /* Typewriter handwriting throughout the entire application */
    html,
    body,
    [class*="css"],
    [data-testid="stApp"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stTextArea"] *,
    [data-testid="stTextInput"] *,
    [data-testid="stButton"] *,
    [data-testid="stAlert"] *,
    label,
    input,
    textarea,
    button {
        font-family: 'Special Elite', monospace !important;
    }

    .stApp {
        background-color: #f5efdf;
        color: #302820;
    }

    /* Keep the distinctive Old English masthead */
    .masthead {
        font-family: 'UnifrakturMaguntia', cursive !important;
        color: #34271f;
        text-align: center;
        font-size: clamp(3rem, 8vw, 5rem);
        line-height: 1.15;
        padding-top: 12px;
    }

    .tagline {
        font-family: 'Special Elite', monospace !important;
        text-align: center;
        color: #76524b;
        letter-spacing: 3px;
        font-size: 11px;
        margin: 8px 0 18px;
    }

    .newspaper-rule {
        border: none;
        border-top: 2px solid #70443f;
        margin: 12px 0;
    }

    .thin-rule {
        border: none;
        border-top: 1px solid #b7a28b;
        margin: 10px 0;
    }

    .intro {
        font-family: 'Special Elite', monospace !important;
        text-align: center;
        color: #6b584a;
        font-size: 13px;
        margin: 16px 0 24px;
    }

    .section-heading {
        font-family: 'Special Elite', monospace !important;
        font-size: 19px;
        color: #3a2a24;
        margin: 20px 0 10px;
    }

    .result-paper {
        font-family: 'Special Elite', monospace !important;
        background: #eee4d0;
        border: 1px solid #c9b79e;
        border-radius: 15px;
        padding: 17px 19px;
        margin: 12px 0;
        box-shadow: 0 2px 5px rgba(55, 38, 27, 0.05);
    }

    .result-label {
        font-family: 'Special Elite', monospace !important;
        color: #80574c;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-value {
        font-family: 'Special Elite', monospace !important;
        color: #302820;
        font-size: 21px;
        overflow-wrap: anywhere;
    }

    /* Rounded vintage input */
    [data-testid="stTextArea"] textarea {
        background: #fffaf0 !important;
        color: #302820 !important;
        border: 1px solid #b8a58e !important;
        border-radius: 15px !important;
        font-family: 'Special Elite', monospace !important;
        font-size: 15px !important;
    }

    [data-testid="stTextArea"] label {
        font-family: 'Special Elite', monospace !important;
        color: #70443f !important;
    }

    /* Rounded burgundy newspaper button */
    div.stButton > button {
        background: #70443f !important;
        color: #fff8ed !important;
        border: 1px solid #70443f !important;
        border-radius: 14px !important;
        font-family: 'Special Elite', monospace !important;
        font-size: 15px !important;
        padding: 12px 18px !important;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #56332f !important;
        border-color: #56332f !important;
        color: #fff8ed !important;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
        font-family: 'Special Elite', monospace !important;
    }

    [data-testid="stAlert"] * {
        font-family: 'Special Elite', monospace !important;
    }

    footer {
        visibility: hidden;
    }
    </style>

    <div class="masthead">The Daily Signal</div>
    <div class="tagline">THE NEWS · DECODED</div>
    <hr class="newspaper-rule">
    <div class="intro">
        Seven categories. Countless stories. One headline at a time.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TEXT PROCESSING
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# LOAD SAVED RESOURCES
# =========================================================

@st.cache_resource(show_spinner=False)
def load_resources():
    resources = {
        "vectorizer": None,
        "tokenizer": None,
        "label_encoder": None,
        "models": {},
        "errors": {},
    }

    supporting_files = {
        "vectorizer": "tfidf_vectorizer.pkl",
        "tokenizer": "tokenizer.pkl",
        "label_encoder": "label_encoder.pkl",
    }

    for key, filename in supporting_files.items():
        path = MODEL_DIR / filename

        try:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

            resources[key] = joblib.load(path)

        except Exception:
            resources["errors"][filename] = traceback.format_exc()

    for model_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename

        try:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

            if filename.endswith(".keras"):
                if load_model is None:
                    raise ImportError(
                        "TensorFlow failed to import. "
                        "Use the environment in which the model was trained."
                    )
                model = load_model(str(path), compile=False)
            else:
                model = joblib.load(path)

            resources["models"][model_name] = model

        except Exception:
            resources["errors"][model_name] = traceback.format_exc()

    return resources


# =========================================================
# LABEL HANDLING
# =========================================================

def decode_label(value, encoder):
    if isinstance(value, (int, np.integer)) and encoder is not None:
        try:
            return str(
                encoder.inverse_transform([int(value)])[0]
            ).strip().upper()
        except Exception:
            pass

    return str(value).strip().upper()


def get_model_labels(model, encoder):
    classes = getattr(model, "classes_", None)

    if classes is not None:
        return [decode_label(c, encoder) for c in classes]

    if encoder is not None and hasattr(encoder, "classes_"):
        return [
            str(label).strip().upper()
            for label in encoder.classes_
        ]

    return CATEGORIES


# =========================================================
# PREDICT CATEGORY SCORES
# =========================================================

def predict_scores(headline, model_name, resources):
    cleaned = clean_text(headline)

    if not cleaned:
        raise ValueError("Enter a headline before predicting.")

    model = resources["models"][model_name]

    if model_name in (
        "Logistic Regression",
        "Naive Bayes",
        "Linear SVM",
    ):
        vectorizer = resources["vectorizer"]

        if vectorizer is None:
            raise RuntimeError(
                "The TF-IDF vectorizer could not be loaded. "
                "Check the loading errors."
            )

        features = vectorizer.transform([cleaned])
        labels = get_model_labels(
            model, resources["label_encoder"]
        )

        if hasattr(model, "predict_proba"):
            values = np.asarray(model.predict_proba(features))[0]
            raw_scores = {
                label: float(value)
                for label, value in zip(labels, values)
            }

        elif hasattr(model, "decision_function"):
            decision = np.asarray(
                model.decision_function(features)
            )

            classes = getattr(model, "classes_", None)

            if decision.ndim == 2 and decision.shape[1] > 1:
                values = decision[0]
                values = values - np.max(values)
                values = np.exp(values)
                values = values / values.sum()

                raw_scores = {
                    label: float(value)
                    for label, value in zip(labels, values)
                }
            else:
                margin = float(decision.reshape(-1)[0])
                margin = np.clip(margin, -50, 50)
                positive = 1.0 / (1.0 + np.exp(-margin))

                if classes is not None and len(classes) == 2:
                    negative_label = decode_label(
                        classes[0], resources["label_encoder"]
                    )
                    positive_label = decode_label(
                        classes[1], resources["label_encoder"]
                    )
                else:
                    negative_label, positive_label = labels[:2]

                raw_scores = {
                    negative_label: float(1 - positive),
                    positive_label: float(positive),
                }

        else:
            predicted = model.predict(features)[0]
            predicted = decode_label(
                predicted, resources["label_encoder"]
            )
            raw_scores = {predicted: 1.0}

    else:
        tokenizer = resources["tokenizer"]

        if tokenizer is None:
            raise RuntimeError(
                "The tokenizer could not be loaded. "
                "Check the loading errors."
            )

        if pad_sequences is None:
            raise ImportError(
                "TensorFlow sequence preprocessing could not be imported."
            )

        sequence = tokenizer.texts_to_sequences([cleaned])

        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        sequence_length = 100
        if input_shape is not None and len(input_shape) > 1:
            if input_shape[1] is not None:
                sequence_length = int(input_shape[1])

        padded = pad_sequences(
            sequence,
            maxlen=sequence_length,
            padding="post",
            truncating="post",
        )

        values = np.asarray(
            model.predict(padded, verbose=0)
        ).reshape(-1)

        encoder = resources["label_encoder"]
        if encoder is not None and hasattr(encoder, "classes_"):
            labels = [
                str(label).strip().upper()
                for label in encoder.classes_
            ]
        else:
            labels = CATEGORIES

        raw_scores = {
            label: float(value)
            for label, value in zip(labels, values)
        }

    normalized_scores = {}

    for label, value in raw_scores.items():
        normalized = str(label).strip().upper()
        if normalized in CATEGORIES:
            normalized_scores[normalized] = max(
                0.0, float(value)
            )

    if not normalized_scores:
        raise RuntimeError(
            "The model's output labels do not match the seven selected "
            "categories. Check the saved label encoder and training labels."
        )

    scores = {
        category: normalized_scores.get(category, 0.0)
        for category in CATEGORIES
    }

    total = sum(scores.values())

    if total <= 0:
        raise RuntimeError("The model returned empty category scores.")

    return {
        category: value / total
        for category, value in scores.items()
    }


# =========================================================
# AUTOMATIC MODEL SELECTION
# =========================================================

@st.cache_data(show_spinner=False)
def load_validation_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    rows = []

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            category = str(
                item.get("category", "")
            ).strip().upper()

            headline = clean_text(item.get("headline", ""))

            if category in CATEGORIES and headline:
                rows.append((headline, category))

    if not rows:
        raise ValueError(
            "No matching headlines were found in the dataset."
        )

    frame = pd.DataFrame(rows, columns=["headline", "category"])

    frame = (
        frame.groupby("category", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), 100),
                random_state=42,
            )
        )
        .reset_index(drop=True)
    )

    return frame


def evaluate_models(resources):
    validation = load_validation_data()
    results = {}
    errors = {}

    for model_name in resources["models"]:
        try:
            actual = []
            predicted = []

            for row in validation.itertuples(index=False):
                scores = predict_scores(
                    row.headline, model_name, resources
                )

                actual.append(row.category)
                predicted.append(max(scores, key=scores.get))

            results[model_name] = f1_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0,
            )

        except Exception:
            errors[model_name] = traceback.format_exc()

    if not results:
        details = "\n\n".join(
            f"{name}\n{error}"
            for name, error in {
                **resources["errors"],
                **errors,
            }.items()
        )
        raise RuntimeError(
            "No model could be evaluated.\n" + details
        )

    return max(results, key=results.get), results, errors


# =========================================================
# INITIALIZE APP
# =========================================================

try:
    resources = load_resources()

    if not resources["models"]:
        st.error("No saved models could be loaded.")

        with st.expander("Model loading errors", expanded=True):
            for name, error in resources["errors"].items():
                st.markdown(f"**{name}**")
                st.code(error)

        st.stop()

    if "selected_model" not in st.session_state:
        with st.spinner("Preparing the newsroom..."):
            best_model, model_scores, evaluation_errors = (
                evaluate_models(resources)
            )

        st.session_state["selected_model"] = best_model
        st.session_state["model_scores"] = model_scores
        st.session_state["evaluation_errors"] = evaluation_errors

except Exception as exc:
    st.error("The models could not be loaded or evaluated.")
    st.error(f"{type(exc).__name__}: {exc}")

    with st.expander("Full technical details", expanded=True):
        st.code(traceback.format_exc())

        if "resources" in locals():
            for name, error in resources.get("errors", {}).items():
                st.markdown(f"**{name}**")
                st.code(error)

    st.stop()


# =========================================================
# HEADLINE INPUT
# =========================================================

st.markdown(
    '<div class="section-heading">Today’s Headline</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)

headline = st.text_area(
    "Headline",
    placeholder="Type or paste a news headline here...",
    height=125,
    label_visibility="collapsed",
)

predict_clicked = st.button(
    "✦  Discover the Story  ✦",
    use_container_width=True,
)


# =========================================================
# PREDICTION RESULTS
# =========================================================

if predict_clicked:
    if not headline.strip():
        st.warning("Please enter a headline first.")

    else:
        try:
            chosen_model = st.session_state["selected_model"]

            scores = predict_scores(
                headline, chosen_model, resources
            )

            predicted_category = max(scores, key=scores.get)
            headline_score = scores[predicted_category]

            st.markdown(
                '<hr class="newspaper-rule">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-paper">
                    <div class="result-label">
                        Automatically Selected Model
                    </div>
                    <div class="result-value">
                        {html.escape(chosen_model)}
                    </div>
                </div>

                <div class="result-paper">
                    <div class="result-label">
                        Predicted Category
                    </div>
                    <div class="result-value">
                        {html.escape(predicted_category)}
                    </div>
                </div>

                <div class="result-paper">
                    <div class="result-label">
                        Headline-specific score
                    </div>
                    <div class="result-value">
                        {headline_score:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Decorative category score panel
            ordered_scores = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            bars_html = ""

            for category, score in ordered_scores:
                percentage = max(
                    0.0,
                    min(100.0, float(score) * 100),
                )

                bars_html += f"""
                <div class="score-row">
                    <div class="score-label">
                        <span>{html.escape(category)}</span>
                        <span>{percentage:.1f}%</span>
                    </div>
                    <div class="score-track">
                        <div class="score-fill"
                             style="width:{percentage:.4f}%;">
                        </div>
                    </div>
                </div>
                """

            components.html(
                f"""
                <!DOCTYPE html>
                <html>
                <head>
                <style>
                    @import url(
                      'https://fonts.googleapis.com/css2?family=Special+Elite&display=swap'
                    );

                    * {{
                        box-sizing: border-box;
                        font-family: 'Special Elite', monospace !important;
                    }}

                    body {{
                        margin: 0;
                        padding: 8px;
                        background: transparent;
                        color: #f5efdf;
                    }}

                    .score-paper {{
                        background:
                            repeating-linear-gradient(
                                0deg,
                                #302820,
                                #302820 27px,
                                #352c23 28px
                            );
                        border: 1px solid #a98770;
                        border-radius: 16px;
                        padding: 20px;
                    }}

                    .eyebrow {{
                        text-align: center;
                        color: #d0ad95;
                        font-size: 10px;
                        letter-spacing: 3px;
                        margin-bottom: 10px;
                    }}

                    .heading {{
                        text-align: center;
                        color: #f5efdf;
                        font-size: 22px;
                        margin-bottom: 5px;
                    }}

                    .subtitle {{
                        text-align: center;
                        color: #d2b9a6;
                        font-size: 11px;
                        margin-bottom: 18px;
                    }}

                    .rule {{
                        border: 0;
                        border-top: 1px solid #89695a;
                        margin: 14px 0 18px;
                    }}

                    .score-row {{
                        margin: 16px 0;
                    }}

                    .score-label {{
                        display: flex;
                        justify-content: space-between;
                        gap: 10px;
                        font-size: 12px;
                        margin-bottom: 7px;
                    }}

                    .score-label span:first-child {{
                        overflow-wrap: anywhere;
                    }}

                    .score-label span:last-child {{
                        color: #e4b9a5;
                        white-space: nowrap;
                    }}

                    .score-track {{
                        height: 11px;
                        border: 1px solid #776457;
                        background: #51463b;
                        border-radius: 20px;
                        overflow: hidden;
                    }}

                    .score-fill {{
                        height: 100%;
                        background: linear-gradient(
                            90deg, #8c4f4c, #d1a18b
                        );
                        border-radius: 20px;
                    }}

                    .footer {{
                        text-align: center;
                        color: #c5a995;
                        font-size: 10px;
                        letter-spacing: 1px;
                        margin-top: 18px;
                    }}
                </style>
                </head>
                <body>
                    <div class="score-paper">
                        <div class="eyebrow">
                            THE DAILY SIGNAL · ANALYSIS
                        </div>
                        <div class="heading">
                            Inside the Headline
                        </div>
                        <div class="subtitle">
                            Category score distribution
                        </div>
                        <hr class="rule">
                        {bars_html}
                        <hr class="rule">
                        <div class="footer">
                            EVERY HEADLINE HAS A STORY
                        </div>
                    </div>
                </body>
                </html>
                """,
                height=650,
                scrolling=False,
            )

        except Exception as exc:
            st.error("Prediction failed.")
            st.error(f"{type(exc).__name__}: {exc}")

            with st.expander("Technical details", expanded=True):
                st.code(traceback.format_exc())


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <hr class="newspaper-rule">
    <div class="tagline">
        THE DAILY SIGNAL · A NEWS CLASSIFICATION PROJECT
    </div>
    """,
    unsafe_allow_html=True,
)