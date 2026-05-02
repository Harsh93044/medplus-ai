# MedPlus AI - Disease Prediction System

A machine learning-powered web application for predicting diseases based on symptoms. Built with Flask and scikit-learn, this system provides accurate disease predictions along with personalized recommendations for precautions, medications, diet, and workouts.

## Features

- **Symptom-Based Prediction**: Input symptoms to get instant disease predictions
- **Comprehensive Recommendations**: Get detailed descriptions, precautions, medications, diet plans, and workout suggestions
- **User-Friendly Interface**: Clean and responsive web interface
- **Machine Learning Model**: Trained on medical datasets using Support Vector Classifier (SVC)
- **Feedback System**: Collect user feedback for continuous improvement

## Tech Stack

- **Backend**: Python Flask
- **Machine Learning**: scikit-learn (SVC model)
- **Data Processing**: pandas, numpy
- **Frontend**: HTML, CSS, Jinja2 templates
- **Data Storage**: CSV datasets

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Harsh93044/medplus-ai.git
   cd medplus-ai
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure datasets and models are in place**:
   - The `dataset/` folder should contain all CSV files
   - The `models/` folder should contain the trained model (`svc.pkl`)

## Usage

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Enter symptoms**: Type symptoms separated by commas (e.g., "itching, skin_rash, nodal_skin_eruptions")

4. **Get predictions**: The system will predict the disease and provide:
   - Disease description
   - Precautions to take
   - Recommended medications
   - Diet suggestions
   - Workout recommendations

## Project Structure

```
medplus-ai/
├── main.py                 # Flask application
├── requirements.txt        # Python dependencies
├── feedback.json          # User feedback storage
├── disease.ipynb          # Model training notebook
├── project/
│   └── disease.ipynb      # Additional training notebook
├── dataset/               # Medical datasets
│   ├── Training.csv       # Training data
│   ├── symtoms_df.csv     # Symptoms data
│   ├── description.csv    # Disease descriptions
│   ├── precautions_df.csv # Precautions data
│   ├── medications.csv    # Medications data
│   ├── diets.csv          # Diet recommendations
│   ├── workout_df.csv     # Workout suggestions
│   └── Symptom-severity.csv
├── models/                # Trained ML models
│   └── svc.pkl           # Support Vector Classifier model
├── static/                # Static files (CSS, JS, images)
└── templates/             # HTML templates
    ├── index.html        # Main page
    ├── about.html        # About page
    ├── blog.html         # Blog page
    ├── contact.html      # Contact page
    ├── developer.html    # Developer info
    └── hero.html         # Hero section
```

## Model Training

The machine learning model is trained using the Jupyter notebook in `project/disease.ipynb`. It uses:

- **Algorithm**: Support Vector Classifier (SVC)
- **Dataset**: Training.csv with symptom features and disease labels
- **Preprocessing**: Label encoding for categorical targets
- **Evaluation**: Train-test split (70-30)

## API Endpoints

- `GET /`: Home page
- `POST /predict`: Disease prediction endpoint

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical datasets used for training the model
- Flask framework for web development
- scikit-learn for machine learning algorithms
- Contributors and maintainers

## Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.
