"""Student Lifestyle & Academic Performance Prediction: Streamlit app."""
import pandas as pd
import streamlit as st

from src import charts
from src.config import (
    ALGORITHM, DATA_PATH, FEATURE_COLUMNS, FEATURES, N_ESTIMATORS, NOTEBOOK_REPORTED,
    RANDOM_STATE, TARGET, TEST_SIZE,
)
from src.data import column_info, load_dataset, missing_values, numeric_correlation
from src.model import ModelBundle, out_of_range_features, predict_gpa, train_model, validate_inputs

st.set_page_config(page_title="Student GPA Predictor", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(79,70,229,0.07); border: 1px solid rgba(79,70,229,0.25);
        border-radius: 12px; padding: 14px 18px;
    }
    .hero {padding: 1.4rem 1.6rem; border-radius: 16px; color: white;
           background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%); margin-bottom: 1.2rem;}
    .hero h1 {margin: 0 0 .3rem 0; font-size: 2rem; color: white;}
    .hero p {margin: 0; opacity: .92;}
    .gpa-card {text-align: center; padding: 1.6rem; border-radius: 16px;
               background: rgba(79,70,229,0.08); border: 2px solid #4F46E5;}
    .gpa-card .value {font-size: 3.4rem; font-weight: 800; color: #4F46E5; line-height: 1.1;}
    .gpa-card .label {text-transform: uppercase; letter-spacing: .08em; font-size: .8rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data & model (cached) ----------
@st.cache_data(show_spinner="Loading dataset…")
def _read_repo_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Training Random Forest model…")
def _train(df: pd.DataFrame) -> ModelBundle:
    return train_model(df)


def get_dataset() -> pd.DataFrame | None:
    """Use the CSV in data/ if present; otherwise let the visitor upload one."""
    if DATA_PATH.exists():
        try:
            return _read_repo_dataset()
        except ValueError as e:
            st.error(str(e))
            return None

    st.warning(
        f"`{DATA_PATH.relative_to(DATA_PATH.parents[1])}` was not found. "
        "Add the Kaggle CSV to the repository's `data/` folder, or upload it below to try the app."
    )
    upload = st.file_uploader("Upload student_lifestyle_dataset.csv", type="csv")
    if upload is None:
        return None
    try:
        return load_dataset(upload)
    except ValueError as e:
        st.error(str(e))
        return None


# ---------- Pages ----------
def page_home(df: pd.DataFrame, bundle: ModelBundle):
    st.markdown(
        """<div class="hero"><h1>🎓 Student Lifestyle & Academic Performance</h1>
        <p>Estimate a student's GPA from five daily lifestyle habits using a Random Forest
        regression model trained on real survey-style data.</p></div>""",
        unsafe_allow_html=True,
    )

    st.subheader("Dataset & model overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student records", f"{len(df):,}")
    c2.metric("Input features", len(FEATURE_COLUMNS))
    c3.metric("Target variable", TARGET)
    c4.metric("Algorithm", ALGORITHM)

    st.subheader("Model evaluation")
    st.caption(
        f"Computed live on the {bundle.n_test:,}-record test set "
        f"({int(TEST_SIZE * 100)}% hold-out, random_state={RANDOM_STATE}, {N_ESTIMATORS} trees). "
        "The delta shows the difference from the figures reported in the original notebook."
    )
    cols = st.columns(3)
    for col, (name, value) in zip(cols, bundle.metrics.items()):
        ref = NOTEBOOK_REPORTED[name]
        col.metric(name, f"{value:.3f}", delta=f"{value - ref:+.3f} vs notebook ({ref:.3f})", delta_color="off")

    r2 = bundle.metrics["R²"]
    st.info(
        f"An R² of {r2:.3f} means the model explains roughly {r2 * 100:.0f}% of the variation in GPA; "
        "the rest depends on factors outside these five habits. Treat predictions as rough estimates."
    )

    st.subheader("Input features")
    st.table(
        pd.DataFrame(
            [(label, col, hint) for col, (label, _, hint) in FEATURES.items()],
            columns=["Feature", "Column", "Description"],
        ).set_index("Feature")
    )


def page_predict(df: pd.DataFrame, bundle: ModelBundle):
    st.header("🔮 GPA Prediction")
    st.write("Enter a typical day's routine (hours). The five values together cannot exceed 24 hours.")

    left, right = st.columns([3, 2], gap="large")
    values = {}
    with left:
        for col, (label, default, hint) in FEATURES.items():
            values[col] = st.slider(label, 0.0, 24.0, default, step=0.5, help=hint, key=col)

    total = sum(values.values())
    with right:
        st.metric("Total accounted hours", f"{total:g} / 24")
        st.progress(min(total / 24, 1.0))
        if total <= 24:
            st.caption(f"{24 - total:g} hours not accounted for (work, classes, travel, etc.).")

    errors = validate_inputs(values)
    for msg in errors:
        st.error(msg)

    if st.button("Predict GPA", type="primary", disabled=bool(errors)):
        gpa = predict_gpa(bundle, values)
        mae = bundle.metrics["MAE"]
        avg = df[TARGET].mean()

        with right:
            st.markdown(
                f"""<div class="gpa-card"><div class="label">Estimated GPA</div>
                <div class="value">{gpa:.2f}</div>
                <div class="label">typical error ≈ ±{mae:.2f}</div></div>""",
                unsafe_allow_html=True,
            )
            st.caption(f"Dataset average GPA: {avg:.2f} ({gpa - avg:+.2f} difference)")

        outside = out_of_range_features(df, values)
        if outside:
            st.warning(
                "These values are outside the range seen in the training data, so the estimate is "
                "less reliable: " + ", ".join(outside) + "."
            )

    st.info(
        "⚠️ This is a statistical **estimate**, not a guaranteed academic outcome. GPA depends on many "
        "factors this model does not see, and the patterns found here do not prove cause and effect."
    )


def page_dataset(df: pd.DataFrame):
    st.header("📊 Dataset Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Preview")
    n = st.slider("Rows to show", 5, 50, 10)
    st.dataframe(df.head(n), width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Column information")
        st.dataframe(column_info(df), width="stretch", hide_index=True)
    with right:
        st.subheader("Missing values")
        st.dataframe(missing_values(df), width="stretch", hide_index=True)

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe().T.round(3), width="stretch")


def page_visuals(df: pd.DataFrame, bundle: ModelBundle):
    st.header("📈 Data Visualizations")
    tabs = st.tabs(["GPA Distribution", "Actual vs Predicted", "Feature Importance", "Correlation Heatmap"])

    with tabs[0]:
        st.pyplot(charts.gpa_histogram(df))
        st.caption("How GPA values are spread across all students in the dataset.")
    with tabs[1]:
        st.pyplot(charts.actual_vs_predicted(bundle.y_test, bundle.y_pred))
        st.caption("Points closer to the dashed diagonal are more accurate predictions (test set only).")
    with tabs[2]:
        st.pyplot(charts.feature_importance(bundle.importances))
        st.dataframe(
            bundle.importances.sort_values(ascending=False).rename("Importance").to_frame().round(4),
            width="stretch",
        )
        st.caption("A model-based measure of how much each feature is used, not proof of cause and effect.")
    with tabs[3]:
        st.pyplot(charts.correlation_heatmap(numeric_correlation(df)))
        st.caption("Pairwise linear correlation between numeric columns. Correlation is not causation.")


def page_about():
    st.header("ℹ️ About this project")
    st.markdown(
        f"""
**Goal:** estimate GPA from five daily lifestyle habits.

**Method:** {ALGORITHM} (scikit-learn, {N_ESTIMATORS} trees) trained on an
{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test split with `random_state={RANDOM_STATE}`.
`Student_ID` and `Stress_Level` are not used as inputs.

**Limitations**
- The model learns patterns in one dataset and says nothing about causes.
- GPA is also driven by prior grades, teaching quality, circumstances and more.
- Random Forests cannot extrapolate beyond the ranges seen in training.
- Do not use predictions to make decisions about real students.

The original Colab notebook is kept in `notebooks/` as the academic reference.
"""
    )


# ---------- Router ----------
PAGES = ["🏠 Home", "🔮 GPA Prediction", "📊 Dataset Analysis", "📈 Visualizations", "ℹ️ About"]

with st.sidebar:
    st.title("🎓 GPA Predictor")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.caption("Random Forest Regression · scikit-learn")

if page == PAGES[4]:
    page_about()
else:
    dataset = get_dataset()
    if dataset is None:
        st.stop()
    model_bundle = _train(dataset)
    if page == PAGES[0]:
        page_home(dataset, model_bundle)
    elif page == PAGES[1]:
        page_predict(dataset, model_bundle)
    elif page == PAGES[2]:
        page_dataset(dataset)
    else:
        page_visuals(dataset, model_bundle)
