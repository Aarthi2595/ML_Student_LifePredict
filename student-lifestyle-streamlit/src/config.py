"""Central configuration: paths, feature definitions and notebook-reported results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_lifestyle_dataset.csv"

TARGET = "GPA"

# Same five features, same order as the original notebook.
# (label, default hours from the notebook's sample student, help text)
FEATURES = {
    "Study_Hours_Per_Day": ("Study Hours Per Day", 4.0, "Time spent studying each day."),
    "Extracurricular_Hours_Per_Day": ("Extracurricular Hours Per Day", 2.0, "Clubs, sports teams, hobbies, etc."),
    "Sleep_Hours_Per_Day": ("Sleep Hours Per Day", 8.0, "Average nightly sleep."),
    "Social_Hours_Per_Day": ("Social Hours Per Day", 3.0, "Time spent with friends and family."),
    "Physical_Activity_Hours_Per_Day": ("Physical Activity Hours Per Day", 1.0, "Exercise and other physical activity."),
}
FEATURE_COLUMNS = list(FEATURES)

# Identical to the notebook so results match.
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 100

ALGORITHM = "Random Forest Regression"

# Results reported in the original notebook, shown only as a reference.
# The app always computes its own metrics from the data.
NOTEBOOK_REPORTED = {"MAE": 0.184, "RMSE": 0.228, "R²": 0.444}
