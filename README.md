# Heart Disease Prediction Dashboard

A Streamlit machine learning demo based on the supplied `heart.csv` and Jupyter notebook. It provides an individual prediction form, batch CSV scoring, and a small dataset analysis view.

> **Educational use only.** This project is not a medical device and does not diagnose heart disease or estimate an individual's clinical risk. Do not use its output to make care or treatment decisions. Speak with a qualified healthcare professional about medical concerns.

## Features

- Logistic Regression classifier trained from the included CSV when the app starts.
- One-hot encoding for categorical columns and standard scaling for numeric columns, fitted together in a scikit-learn pipeline.
- Individual input form with probability for target class 1.
- Batch prediction from a CSV and downloadable scored results.
- Dataset summary and target/age charts.
- No pre-trained model binary is required; the project retrains from `heart.csv` at runtime.

## Run locally (Windows PowerShell)

Install Python 3.10 or newer, open PowerShell in this folder, and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and run the environment's executable directly:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Input data

Keep `heart.csv` beside `app.py`. It must have the 13 feature columns below. The batch uploader accepts those columns; `target` is optional and not used to make predictions.

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

The included dataset has a `target` column used as the training label. In the provided notebook, `target=1` is the positive class and `target=0` is the negative class. This class label and model probability are dataset-specific, not a clinical risk score.

## Upload to GitHub

Create an empty repository on GitHub, then run these commands from this project folder (replace the URL with your repository URL):

```powershell
git init
git add app.py heart.csv requirements.txt README.md .gitignore .streamlit/config.toml
git commit -m "Add heart disease prediction dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Do not add `.venv`, environment files, API keys, or private data. Review the dataset's source and license before redistributing it publicly.

## Deploy with Streamlit Community Cloud

Push this repository to GitHub, create a new app in Streamlit Community Cloud, select the repository and `main` branch, and set the app file to `app.py`. The dependency list is in `requirements.txt`. Keep `heart.csv` in the repository root beside `app.py` so the hosted app can load it.

## Project files

- `app.py` — Streamlit interface and machine learning pipeline.
- `heart.csv` — supplied dataset used for training and analysis.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — dark theme settings.
- `.gitignore` — ignores local environments and secrets.
