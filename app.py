from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv('final.csv')
df['Cleaned_text'] = df['Cleaned_text'].fillna('')
df['Health'] = df['Health'].fillna('')

# Prepare data for model training
independent_category = df.Cleaned_text
dependent_category = df.Category
in_train_cat, in_test_cat, de_train_cat, de_test_cat = train_test_split(independent_category, dependent_category, test_size=0.3)

df_subset = df.head(17780)
independent_health = df_subset.Cleaned_text
dependent_health = df_subset.Health
in_train_health, in_test_health, de_train_health, de_test_health = train_test_split(independent_health, dependent_health, test_size=0.3)

# Create and train models
tf_category = TfidfVectorizer()
lr_category = LogisticRegression(solver="sag", max_iter=200)
model_category = Pipeline([("vectorizer", tf_category), ("classifier", lr_category)])
model_category.fit(in_train_cat, de_train_cat)

tf_health = TfidfVectorizer()
lr_health = LogisticRegression(solver="sag", max_iter=200)
model_health = Pipeline([("vectorizer", tf_health), ("classifier", lr_health)])
model_health.fit(in_train_health, de_train_health)

# Doctor data
doctors = {
    "General": "Dr. Smith\nPh:9353134049\n email: drsmith@example.com",
    "Cardiology": "Dr. Emily\nPh:9353134049\n email: dremily@example.com",
    "Orthopedics": "Dr. Rahul\nPh:9353134049\n email: drrahul@example.com",
    "Neurology": "Dr. Priya\nPh:9353134049\n email: drpriya@example.com",
    "Dermatology": "Dr. Alice\nPh:9353134049\n email: dralice@example.com"
}

@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = ""
    
    # Predict category
    result_category = model_category.predict([user_input])[0]
    if result_category.lower() == 'medical':
        result_health = model_health.predict([user_input])[0]
        response = f"Classification Result: {result_category}\nHealth Specialty Prediction: {result_health}"
        
        # Include doctor information for the specialty
        doctor_info = doctors.get(result_health, "No doctor available for this specialty")
        response += f"\nContact Doctor: {doctor_info}"
    else:
        response = f"Classification Result: {result_category}"
    
    return jsonify({"response": response})

# Route to handle direct doctor contact simulation
@app.route('/contact_doctor', methods=['POST'])
def contact_doctor():
    specialty = request.form['specialty']
    doctor_response = f"You've contacted the doctor for {specialty}. Doctor says: 'Please provide detailed symptoms, and I will assist you further!'"
    return jsonify({"response": doctor_response})

if __name__ == '__main__':
    app.run(debug=True)
