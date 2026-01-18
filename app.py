import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

# =========================
# LOAD MODEL & PREPROCESSOR
# =========================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "gender": request.form["gender"],
            "race_ethnicity": request.form["race_ethnicity"],
            "parental_level_of_education": request.form["parental_level_of_education"],
            "lunch": request.form["lunch"],
            "test_preparation_course": request.form["test_preparation_course"],
            "reading_score": float(request.form["reading_score"]),
            "writing_score": float(request.form["writing_score"]),
        }

        df = pd.DataFrame([data])

        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Math Score: {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(debug=True)
