import os
import ast
import pandas as pd
import pickle
import json
from flask import Flask, request, render_template, redirect, url_for
from sklearn.preprocessing import LabelEncoder

# ================== INIT APP ==================
app = Flask(__name__)

# ================== CONFIGURATION ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")

# ================== GLOBALS & LOAD DATA ==================
# Initialize as None or empty to handle missing files gracefully
sym_des = precautions_df = workout_df = description_df = medications_df = diets_df = training_df = None
svc = None
SYMPTOM_COLUMNS = []
label_encoder = LabelEncoder()

def load_resources():
    global sym_des, precautions_df, workout_df, description_df, medications_df, diets_df, training_df
    global svc, SYMPTOM_COLUMNS, label_encoder

    try:
        sym_des = pd.read_csv(os.path.join(DATASET_DIR, "symtoms_df.csv"))
        precautions_df = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
        workout_df = pd.read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))
        description_df = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
        medications_df = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
        diets_df = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))
        training_df = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))
        
        # Load Model
        with open(os.path.join(MODELS_DIR, 'svc.pkl'), 'rb') as f:
            svc = pickle.load(f)
            
        # Model Schema
        SYMPTOM_COLUMNS = training_df.columns[:-1].tolist()
        label_encoder.fit(training_df['prognosis'])
        
        print("All resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        # Allows app to start but prevents crash; will show error on prediction

load_resources()

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
    if description_df is None:
        return "Data not available", [], [], [], []

    try:
        canonical_disease = canonicalize_disease_name(dis)
        normalized_disease = normalize_text(canonical_disease)

        desc_row = description_df[description_df['Disease'].map(normalize_text) == normalized_disease]
        desc = " ".join(desc_row['Description'].tolist()) if not desc_row.empty else "No description available."

        precautions_row = precautions_df[precautions_df['Disease'].map(normalize_text) == normalized_disease]
        if not precautions_row.empty:
            precautions = precautions_row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
            precautions = [item for item in precautions if pd.notna(item) and str(item).strip()]
        else:
            precautions = []

        medications = []
        medication_rows = medications_df[medications_df['Disease'].map(normalize_text) == normalized_disease]['Medication'].tolist()
        for row in medication_rows:
            medications.extend(parse_list_cell(row))

        diet = []
        diet_rows = diets_df[diets_df['Disease'].map(normalize_text) == normalized_disease]['Diet'].tolist()
        for row in diet_rows:
            diet.extend(parse_list_cell(row))

        workout = []
        workout_rows = workout_df[workout_df['disease'].map(normalize_text) == normalized_disease]['workout'].tolist()
        for row in workout_rows:
            workout.append(str(row).strip())

        return desc, precautions, medications, diet, workout
    except Exception as e:
        print(f"Error in helper function: {e}")
        return "Error retrieving data", [], [], [], []

# ================== PREDICTION FUNCTION ==================
def get_predicted_value(patient_symptoms):
    if svc is None or not SYMPTOM_COLUMNS:
        return "Error: Model not loaded"
        
    try:
        input_data = {symptom: 0 for symptom in SYMPTOM_COLUMNS}
        
        valid_symptoms_found = False
        for item in patient_symptoms:
            if item in input_data:
                input_data[item] = 1
                valid_symptoms_found = True
            else:
                return "Invalid"
        
        if not valid_symptoms_found:
            return "Invalid"

        input_frame = pd.DataFrame([input_data], columns=SYMPTOM_COLUMNS)
        predicted_class = svc.predict(input_frame)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
        return canonicalize_disease_name(predicted_disease)
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error: Prediction failed"

# ================== ROUTES ==================
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form.get('symptoms')

        # Input validation
        if not symptoms or not str(symptoms).strip():
            return render_template('index.html', message="Please enter valid symptoms")

        # Clean input
        user_symptoms = [str(s).strip().lower().replace(" ", "_") for s in symptoms.split(',') if str(s).strip()]
        
        if not user_symptoms:
             return render_template('index.html', message="Please enter valid symptoms")

        predicted_disease = get_predicted_value(user_symptoms)

        if predicted_disease == "Invalid":
            return render_template('index.html', message="Invalid symptoms entered. Please try again.")
        elif str(predicted_disease).startswith("Error"):
            return render_template('index.html', message="An error occurred during prediction. Please try again later.")

        desc, precautions, medications, diet, workout = helper(predicted_disease)

        return render_template('index.html',
                               predicted_disease=predicted_disease,
                               dis_des=desc,
                               my_precautions=precautions,
                               medications=medications,
                               my_diet=diet,
                               workout=workout)
    except Exception as e:
        print(f"Route error: {e}")
        return render_template('index.html', message="An unexpected error occurred.")

# ================== FEEDBACK SYSTEM ==================
def get_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return ["Amazing tool!", "Really helped me understand my symptoms.", "Modern and clean UI!"]
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Feedback load error: {e}")
        return []

def save_feedback(text):
    try:
        feedbacks = get_feedback()
        feedbacks.append(str(text).strip())
        feedbacks = feedbacks[-20:] # Keep latest 20
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedbacks, f)
    except Exception as e:
        print(f"Feedback save error: {e}")

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
    try:
        feedback_text = request.form.get('feedback')
        if feedback_text and str(feedback_text).strip():
            save_feedback(feedback_text)
    except Exception as e:
        print(f"Submit feedback error: {e}")
    return redirect('/hero')

# ================== RUN ==================
if __name__ == '__main__':
    # Use environment variable for PORT, default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode for production safety
    app.run(host='0.0.0.0', port=port, debug=False)
