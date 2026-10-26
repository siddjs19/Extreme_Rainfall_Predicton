from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "merged_imd_era5_pune_features.csv"
MODEL_PATH = ROOT / "models" / "rainfall_class_weighted_random_forest.pkl"
METADATA_PATH = ROOT / "models" / "rainfall_model_metadata.pkl"

TARGET = "target_rainfall_tomorrow"

# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Intelligent Weather Forecasting Research Prototype",
    page_icon="🌦️",
    layout="wide",
)

st.title("Intelligent Weather Forecasting Research Prototype")
st.caption(
    "Research demonstration: next-day rainfall-category prediction "
    "using historical weather observations and engineered temporal features."
)

# --------------------------------------------------
# Load data and model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return joblib.load(METADATA_PATH)
    return {}


model = load_model()
df = load_dataset()
metadata = load_metadata()

# --------------------------------------------------
# Feature discovery
# --------------------------------------------------

if hasattr(model, "feature_names_in_"):
    FEATURE_NAMES = list(model.feature_names_in_)
else:
    excluded_columns = {
        TARGET,
        "date",
        "Date",
        "datetime",
        "timestamp",
        "time",
    }

    FEATURE_NAMES = [
        column
        for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

missing_features = [
    feature for feature in FEATURE_NAMES
    if feature not in df.columns
]

if missing_features:
    st.error(
        "The saved model expects features that are missing from the dataset:\n\n"
        + ", ".join(missing_features)
    )
    st.stop()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

CLASS_NAMES = {
    0: "Dry",
    1: "Extreme Rain",
    2: "Heavy Rain",
    3: "Light Rain",
    4: "Moderate Rain",
}


def rainfall_amount_to_category(value):
    value = float(value)

    if value <= 0:
        return "Dry"
    elif value <= 15:
        return "Light Rain"
    elif value <= 64.5:
        return "Moderate Rain"
    elif value <= 115.5:
        return "Heavy Rain"
    else:
        return "Extreme Rain"


def prediction_label(prediction):
    """Convert the encoded model prediction to the original training label."""

    try:
        return CLASS_NAMES[int(prediction)]
    except (ValueError, TypeError, KeyError):
        return str(prediction)


def get_actual_label(row):
    """Convert actual rainfall amount into the same labels used during training."""

    value = row[TARGET]

    if pd.isna(value):
        return "Unavailable"

    if isinstance(value, str):
        text = value.strip()

        # Already a training category
        if text in CLASS_NAMES.values():
            return text

        try:
            value = float(text)
        except ValueError:
            return text

    return rainfall_amount_to_category(value)


def find_date_column():
    candidates = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


DATE_COLUMN = find_date_column()

if DATE_COLUMN:
    df["_display_date"] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")
else:
    df["_display_date"] = df.index.astype(str)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Research Model")

st.sidebar.write(
    "**Model:** Class-weighted Random Forest"
)

st.sidebar.write(
    f"**Input features:** {len(FEATURE_NAMES)}"
)

st.sidebar.write(
    f"**Dataset rows:** {len(df):,}"
)

if metadata:
    st.sidebar.subheader("Training Metadata")

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            st.sidebar.write(f"**{key}:** {value}")

# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Observations", f"{len(df):,}")
col2.metric("Model Features", f"{len(FEATURE_NAMES)}")
col3.metric("Target", "Next-Day Rainfall")
col4.metric("Location", "Pune, Maharashtra")

with st.expander("View available model features"):
    st.write(FEATURE_NAMES)

# --------------------------------------------------
# Historical prediction demonstration
# --------------------------------------------------

st.header("2. Historical Prediction Demonstration")

st.write(
    "Select one historical observation from the dataset. "
    "The application sends the stored feature vector to the trained model "
    "and compares the prediction with the actual next-day rainfall category."
)

display_options = df["_display_date"].tolist()

selected_position = st.selectbox(
    "Select a historical observation",
    options=range(len(df)),
    format_func=lambda position: display_options[position],
)

selected_row = df.iloc[selected_position]

st.subheader(
    f"Selected observation: {selected_row['_display_date']}"
)

input_data = selected_row[FEATURE_NAMES].to_frame().T

prediction = model.predict(input_data)[0]
predicted_label = prediction_label(prediction)
actual_label = get_actual_label(selected_row)

st.write("### Debug Information")

st.write({
    "raw_model_prediction": prediction,
    "model_classes": list(getattr(model, "classes_", [])),
    "raw_target_value": selected_row[TARGET],
    "normalized_prediction": predicted_label,
    "normalized_actual": actual_label,
})
result_col1, result_col2, result_col3 = st.columns(3)

result_col1.metric("Predicted Category", predicted_label)
result_col2.metric("Actual Category", actual_label)

correct = predicted_label.lower() == actual_label.lower()

if correct:
    result_col3.success("Prediction matches actual outcome")
else:
    result_col3.warning("Prediction differs from actual outcome")

with st.expander("View input feature vector"):
    st.dataframe(
        input_data.T.rename(columns={selected_position: "Value"}),
        use_container_width=True,
    )

# --------------------------------------------------
# Manual prediction
# --------------------------------------------------

st.header("3. Manual Feature-Based Prediction")

st.write(
    "Modify selected meteorological features to demonstrate how the "
    "trained classifier responds to changing weather conditions."
)

manual_values = {}

important_features = [
    feature for feature in [
        "rainfall_lag_1",
        "rainfall_rolling_3",
        "rainfall_rolling_7",
        "rain_today_flag",
        "temp_min",
        "temp_max",
        "rh_mean",
        "pressure_mean",
        "dewpoint_mean",
        "dewpoint_depression",
    ]
    if feature in FEATURE_NAMES
]

if not important_features:
    important_features = FEATURE_NAMES[:10]

manual_columns = st.columns(2)

for index, feature in enumerate(important_features):
    default_value = float(selected_row[feature])

    with manual_columns[index % 2]:
        manual_values[feature] = st.number_input(
            feature,
            value=default_value,
            format="%.4f",
        )

if st.button("Run Manual Prediction", type="primary"):
    manual_input = selected_row[FEATURE_NAMES].copy()

    for feature, value in manual_values.items():
        manual_input[feature] = value

    manual_input = manual_input.to_frame().T

    manual_prediction = model.predict(manual_input)[0]
    manual_label = prediction_label(manual_prediction)

    st.success(f"Predicted next-day rainfall category: **{manual_label}**")

    with st.expander("View modified feature vector"):
        st.dataframe(
            manual_input.T.rename(columns={selected_position: "Value"}),
            use_container_width=True,
        )

# --------------------------------------------------
# Model comparison
# --------------------------------------------------

st.header("4. Model Comparison")

comparison_path = (
    ROOT
    / "results"
    / "phase11_model_improvement"
    / "model_comparison.csv"
)

if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    metric_candidates = [
        column for column in [
            "high_risk_recall",
            "High-Risk Recall",
            "recall_high_risk",
            "accuracy",
            "Accuracy",
        ]
        if column in comparison_df.columns
    ]

    if metric_candidates:
        metric_column = metric_candidates[0]

        chart_df = comparison_df.copy()
        chart_df = chart_df.set_index(chart_df.columns[0])

        st.bar_chart(chart_df[metric_column])

else:
    st.info("Model comparison file was not found.")

# --------------------------------------------------
# Explainability
# --------------------------------------------------

st.header("5. Model Explainability")

importance_path = (
    ROOT
    / "results"
    / "phase11_model_improvement"
    / "feature_importance_class_weighted_random_forest.csv"
)

if importance_path.exists():
    importance_df = pd.read_csv(importance_path)

    st.dataframe(
        importance_df.head(15),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The most influential variables help researchers understand "
        "which atmospheric and temporal signals contribute to the model."
    )

else:
    st.info("Feature-importance file was not found.")

# --------------------------------------------------
# Research interpretation
# --------------------------------------------------

st.header("6. Research Interpretation")

st.markdown(
    """
### Why this model was selected

The class-weighted Random Forest was selected because the research objective
prioritizes identifying high-risk rainfall events rather than maximizing
overall accuracy alone.

Class weighting increases the importance of minority high-risk rainfall
categories during training. This helps reduce missed heavy and extreme
rainfall events.

### Important limitations

- This is a research prototype, not an operational warning system.
- Predictions depend on the quality and availability of historical features.
- The current module predicts next-day rainfall category.
- Disaster prediction requires additional hazards, exposure, vulnerability,
  and impact data.
- Operational deployment would require live data pipelines, validation,
  uncertainty estimates, and human review.
"""
)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Study period: 2021–2025 | Location: Pune, Maharashtra | "
    "Research prototype for intelligent weather forecasting and disaster prediction"
)