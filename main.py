from flask import Flask, request, render_template, redirect, url_for
import ast
import pandas as pd
import pickle
import json
import os
from sklearn.preprocessing import LabelEncoder

# ================== INIT APP ==================
app = Flask(__name__)

# ================== LOAD DATA ==================
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions_df = pd.read_csv("dataset/precautions_df.csv")
workout_df = pd.read_csv("dataset/workout_df.csv")
description_df = pd.read_csv("dataset/description.csv")
medications_df = pd.read_csv('dataset/medications.csv')
diets_df = pd.read_csv("dataset/diets.csv")
training_df = pd.read_csv("dataset/Training.csv")

# ================== LOAD MODEL ==================
svc = pickle.load(open('models/svc.pkl', 'rb'))

# ================== MODEL SCHEMA ==================
SYMPTOM_COLUMNS = training_df.columns[:-1].tolist()
label_encoder = LabelEncoder()
label_encoder.fit(training_df['prognosis'])

# ================== NAME NORMALIZATION ==================
DISEASE_NAME_ALIASES = {
    'peptic ulcer diseae': 'Peptic ulcer disease',
    'diabetes': 'Diabetes',
    'hypertension': 'Hypertension',
    '(vertigo) paroymsal positional vertigo': '(vertigo) Paroymsal Positional Vertigo',
}


def normalize_text(value):
    return " ".join(str(value).strip().split()).lower()


def canonicalize_disease_name(disease_name):
    normalized = normalize_text(disease_name)
    return DISEASE_NAME_ALIASES.get(normalized, str(disease_name).strip())


def parse_list_cell(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [str(parsed).strip()]

# ================== HELPER FUNCTION ==================
def helper(dis):
    canonical_disease = canonicalize_disease_name(dis)
    normalized_disease = normalize_text(canonical_disease)

    desc_row = description_df[
        description_df['Disease'].map(normalize_text) == normalized_disease
    ]
    desc = " ".join(desc_row['Description'].tolist())

    precautions = precautions_df[
        precautions_df['Disease'].map(normalize_text) == normalized_disease
    ][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    ].values.flatten().tolist()
    precautions = [item for item in precautions if pd.notna(item) and str(item).strip()]

    medication_rows = medications_df[
        medications_df['Disease'].map(normalize_text) == normalized_disease
    ]['Medication'].tolist()
    medications = []
    for row in medication_rows:
        medications.extend(parse_list_cell(row))

    diet_rows = diets_df[
        diets_df['Disease'].map(normalize_text) == normalized_disease
    ]['Diet'].tolist()
    diet = []
    for row in diet_rows:
        diet.extend(parse_list_cell(row))

    workout = workout_df[
        workout_df['disease'].map(normalize_text) == normalized_disease
    ]['workout'].tolist()

    return desc, precautions, medications, diet, workout

# ================== PREDICTION FUNCTION ==================
def get_predicted_value(patient_symptoms):
    input_data = {symptom: 0 for symptom in SYMPTOM_COLUMNS}

    for item in patient_symptoms:
        if item in input_data:
            input_data[item] = 1
        else:
            return "Invalid"

    input_frame = pd.DataFrame([input_data], columns=SYMPTOM_COLUMNS)
    predicted_class = svc.predict(input_frame)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    return canonicalize_disease_name(predicted_disease)

# ================== ROUTES ==================
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')

    # Empty check
    if not symptoms or not symptoms.strip():
        return render_template('index.html', message="Please enter symptoms")

    # Clean input
    user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]

    predicted_disease = get_predicted_value(user_symptoms)

    if predicted_disease == "Invalid":
        return render_template('index.html', message="Invalid symptoms entered")

    desc, precautions, medications, diet, workout = helper(predicted_disease)

    return render_template('index.html',
                           predicted_disease=predicted_disease,
                           dis_des=desc,
                           my_precautions=precautions,
                           medications=medications,
                           my_diet=diet,
                           workout=workout)

# ================== FEEDBACK SYSTEM ==================
FEEDBACK_FILE = "feedback.json"

def get_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return ["Amazing tool!", "Really helped me understand my symptoms.", "Modern and clean UI!"]
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def save_feedback(text):
    feedbacks = get_feedback()
    feedbacks.append(text)
    feedbacks = feedbacks[-20:] # Keep latest 20
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedbacks, f)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/hero')
def hero():
    feedbacks = get_feedback()
    return render_template("hero.html", feedbacks=feedbacks)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_text = request.form.get('feedback')
    if feedback_text:
        save_feedback(feedback_text)
    return redirect('/hero')

# ================== RUN ==================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
