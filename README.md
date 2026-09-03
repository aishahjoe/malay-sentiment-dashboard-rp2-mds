# Streamlit Dashboard Tutorial

## Local setup

1. Install Python 3.10 or 3.11.
2. Create a folder and place `app.py` and `requirements.txt` inside it.
3. Open Terminal in that folder.
4. Run:

```bash
python -m venv venv
```

Activate it:

### Windows
```bash
venv\Scripts\activate
```

### macOS
```bash
source venv/bin/activate
```

5. Install packages:

```bash
pip install -r requirements.txt
```

6. Start Streamlit:

```bash
streamlit run app.py
```

The dashboard normally opens at `http://localhost:8501`.

## What is included

- Overview of the best ML, DL and Transformer models
- Complete ML results
- Complete DL results
- Complete Transformer results
- Transformer confusion matrices
- Normalised confusion matrices
- Transformer training/validation curves
- Literature comparison
- Direct CNN + Word2Vec comparison with Ying et al. (2020)
- Transformer training-time comparison
- Accuracy versus training-time chart

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud.
4. Connect GitHub.
5. Select your repository.
6. Select `app.py` as the main file.
7. Deploy.

This version is a results dashboard. It does not retrain the models.

## Important

The dashboard uses the current experimental results and describes the data as a **combined Malay sentiment dataset**, not MALAYA 20K alone.
