# 🎓 Student Lifestyle & Academic Performance Prediction

A Streamlit web app that estimates a student's **GPA** from five daily lifestyle habits using
**Random Forest Regression** (scikit-learn). Converted from the original Google Colab project.

| Item | Value |
|---|---|
| Dataset | `student_lifestyle_dataset.csv` (Kaggle, 2,000 records) |
| Inputs | Study, Extracurricular, Sleep, Social, Physical Activity hours per day |
| Target | `GPA` |
| Model | `RandomForestRegressor(n_estimators=100, random_state=42)`, 80/20 split |

## Features
- **Home**: dataset stats and live-computed MAE / RMSE / R²
- **GPA Prediction**: validated sliders (each 0–24 h, total ≤ 24 h) and an estimated GPA
- **Dataset Analysis**: preview, shape, column info, missing values, duplicates, statistics
- **Visualizations**: GPA histogram, actual vs predicted, feature importance, correlation heatmap

Metrics are **computed from your data on every deploy**, never hard-coded. The notebook's reported
figures (MAE 0.184, RMSE 0.228, R² 0.444) appear only as a labelled reference.

## Project structure
```
app.py                  Streamlit UI
src/config.py           Paths, feature list, hyperparameters
src/data.py             Loading and dataset inspection
src/model.py            Training, evaluation, validation, prediction
src/charts.py           Matplotlib charts
data/                   Put student_lifestyle_dataset.csv here
notebooks/              Original Colab notebook (academic reference)
tests/                  pytest suite
.streamlit/config.toml  Theme
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy student_lifestyle_dataset.csv into data/
streamlit run app.py
```
Run tests: `pip install -r requirements-dev.txt && pytest`

## Deploy: GitHub → Streamlit Community Cloud
1. Copy `student_lifestyle_dataset.csv` into `data/` (check the Kaggle licence for public repos).
2. Push to GitHub:
   ```bash
   git init && git add . && git commit -m "Streamlit GPA predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo>.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo,
   branch `main`, main file `app.py` → **Deploy**.

If the CSV isn't in the repo, the app shows an upload box instead of crashing.

## Limitations
Predictions are estimates from patterns in one dataset, not proof of cause and effect, and not a
guaranteed academic outcome. Do not use them to make decisions about real students.
