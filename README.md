# KindlyHeart Backend

A Flask backend that turns the static HeartCare site into a real client–server
app.

## What it actually does (for the demo)

- On startup, it **trains a real scikit-learn Logistic Regression model**
  directly from `heart.csv` (303 patients from the UCI heart disease
  dataset) — not hardcoded numbers. You'll see the training/test accuracy
  printed in the terminal when it starts.
- `index.html`'s form now calls this backend (`POST /api/predict`) to get
  its risk prediction, instead of computing it in the browser. If the
  backend can't be reached, the page quietly falls back to its old local
  JS model, so it never breaks — but with the server running, the
  prediction really does come from the trained model.
- `patient-details.html` now saves each submitted patient into a real
  SQLite database (`kindlyheart.db`) via `POST /api/patients`, in addition
  to keeping a local copy in the browser.
- The same server also serves the website itself, so everything runs from
  one command.

## How to run it

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000/patient-details.html** in your browser
and use the site normally.

## Endpoints (useful to show your teacher directly)

| Method | Endpoint         | What it does                                      |
|--------|------------------|----------------------------------------------------|
| GET    | `/api/health`    | Confirms the server is up and the model is loaded, with its accuracy |
| POST   | `/api/predict`   | Takes patient vitals, returns risk probability from the trained model |
| POST   | `/api/patients`  | Saves a patient's details to the database          |
| GET    | `/api/patients`  | Lists every patient saved so far                   |

Quick things to try live:

- Open `http://127.0.0.1:5000/api/health` in a browser tab — shows the
  model's training/test accuracy pulled straight from `heart.csv`.
- Fill in the patient details form → submit → open
  `http://127.0.0.1:5000/api/patients` in another tab to show the saved
  record just landed in the database.
- Fill in the risk predictor form on the main page → the result now comes
  from the real trained model, not a hardcoded formula.

## Files

- `app.py` — the Flask server (model training + API + serving the site)
- `heart.csv` — the training dataset
- `requirements.txt` — Python dependencies
- `kindlyheart.db` — created automatically the first time you run the
  server; stores submitted patients
