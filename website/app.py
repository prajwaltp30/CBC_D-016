from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
import subprocess
import os
import numpy as np
from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.databases import Databases
from appwrite.id import ID
from database.save_user import save_user_data
from ml_logic.predict import get_user_details, predict_manually
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch

# Import from updated scan.py
from opencv_scan.scan import generate_frames, get_live_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this

# Appwrite setup
client = Client()
client.set_endpoint('https://cloud.appwrite.io/v1')
client.set_project('667f8c85002229461ca8')
client.set_key('[YOUR_API_KEY]')  # Replace with your API key

# ---- ROUTES ----

@app.route('/')
def index():
    if 'email' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        data = request.json
        username = data['username']
        email = data['email']
        nickname = data['nickname']
        password = data['password']

        account = Account(client)
        databases = Databases(client)

        try:
            account.create(
                user_id=ID.unique(),
                email=email,
                password=password,
                name=username
            )

            databases.create_document(
                database_id='667f8d010031471a488a',
                collection_id='667f8d16003418fd93a2',
                document_id=ID.unique(),
                data={
                    'email': email,
                    'name': username,
                    'nickname': nickname,
                    'password': password,
                }
            )
            return jsonify({"message": "User registered successfully"}), 200

        except Exception as e:
            return jsonify({"message": str(e)}), 400

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400

        databases = Databases(client)
        try:
            result = databases.list_documents(
                database_id='667f8d010031471a488a',
                collection_id='667f8d16003418fd93a2'
            )

            user = None
            for document in result['documents']:
                if document['email'] == email:
                    user = document
                    break

            if user is None:
                return jsonify({"success": False, "message": "Email not found"}), 404

            if user['password'] == password:
                session['email'] = email
                return jsonify({"success": True, "redirect": url_for('index')})
            else:
                return jsonify({"success": False, "message": "Incorrect password"}), 401

        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('signup'))

@app.route('/home')
def home():
    if 'email' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

# --- MODIFIED SECTION FOR VIDEO AND LIVE DATA ---

@app.route('/start', methods=['POST'])
def start_script():
    try:
        return render_template('scan.html')
    except Exception as e:
        return f"Error starting scan: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_data')
def live_data():
    data = get_live_data()
    return jsonify(data)

# --- END OF MODIFIED SECTION ---

@app.route('/results', methods=['GET', 'POST'])
def show_results():
    additional_graph_exists = os.path.exists("static/final_graph.png")

    bpm_data = []
    if os.path.exists("bpm_values.txt"):
        with open("bpm_values.txt", "r") as file:
            for line in file:
                bpm = float(line.strip())
                bpm_data.append(bpm)

    average_bpm = np.mean(bpm_data[5:]) if len(bpm_data) > 5 else 0

    return render_template('results.html',
                           additional_graph_exists=additional_graph_exists,
                           average_bpm=average_bpm)

@app.route('/details_form')
def details_form():
    return render_template('detail.html')

@app.route('/save_details', methods=['POST'])
def save_details():
    try:
        form = request.form
        age = form['age']
        sex = form.get('sex') == '1'
        diabetes = 'diabetes' in form
        famhistory = 'famhistory' in form
        smoking = 'smoking' in form
        obesity = 'obesity' in form
        alcohol = 'alcohol' in form
        exercise = form['exercise']
        diet = form['diet']
        heartproblem = 'heartproblem' in form
        bmi = form['bmi']
        physicalactivity = form['physicalactivity']
        sleep = form['sleep']
        bp1 = form['bp1']
        bp2 = form['bp2']
        email = session.get('email')

        save_user_data(email, age, sex, diabetes, famhistory, smoking, obesity, alcohol,
                       exercise, diet, heartproblem, bmi, physicalactivity, sleep, bp1, bp2)

        return render_template('index.html')

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return "An error occurred while processing your request.", 500

# [rest of ml prediction, download_report remains same, not changed]

@app.route('/ml')
def ml():
    email = session.get('email')
    user_details = None
    if email:
        user_details = get_user_details(email)
    return render_template('ml.html', user_details=user_details)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return render_template('ml.html', error="Session expired. Please log in again.")

        user_details = get_user_details(email)
        if not user_details:
            return render_template('ml.html', error="User details not found in the database.")

        age = int(user_details['age'])
        sex = int(user_details['sex'])
        diabetes = int(user_details['diabetes'])
        family_history = int(user_details['famhistory'])
        smoking = int(user_details['smoking'])
        obesity = int(user_details['obesity'])
        alcohol_consumption = int(user_details['alcohol'])
        exercise_hours_per_week = float(user_details['exercise_hours'])
        diet = int(user_details['diet'])
        previous_heart_problems = int(user_details['heart_problem'])
        bmi = float(user_details['bmi'])
        physical_activity_days_per_week = int(user_details['physical_activity'])
        sleep_hours_per_day = float(user_details['sleep_hours'])
        bp1 = int(user_details['blood_pressure_systolic'])
        bp2 = int(user_details['blood_pressure_diastolic'])

        try:
            heart_rate = int(request.form['heart_rate'])
            stress_level = int(request.form['stress_level'])
            hrv = int(request.form['hrv'])
            spo2 = int(request.form['spo2'])
        except KeyError as e:
            return render_template('ml.html', error=f"Missing form data: {e}")

        predicted_heart_attack_risk, predicted_percentage = predict_manually(
            age, sex, heart_rate, diabetes, family_history, smoking, obesity,
            alcohol_consumption, exercise_hours_per_week, diet, previous_heart_problems,
            0, stress_level, bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2
        )

        messages = []
        if hrv < 45:
            messages.append("Your HRV is too low")
            predicted_percentage += 25.8
        if hrv > 200:
            messages.append("Your HRV is too high")
            predicted_percentage += 5.65
        if spo2 < 90:
            messages.append("Your SpO2 level is below normal.")
            predicted_percentage += 4.5

        if age > 50:
            messages.append("Age greater than 50.")
        if heart_rate < 60 or heart_rate > 100:
            messages.append("Your heart rate is not good.")
        if diabetes:
            messages.append("You have diabetes.")
        if family_history:
            messages.append("You have a family history of heart problems.")
        if smoking:
            messages.append("You are a smoker.")
        if obesity:
            messages.append("You are obese.")
        if alcohol_consumption:
            messages.append("You consume alcohol.")
        if previous_heart_problems:
            messages.append("You have had previous heart problems.")
        if stress_level > 6:
            messages.append("Stress Level is High, Take rest or Hangout.")
        if bp1 > 140 or bp2 > 80:
            messages.append("Your blood pressure is high.")

        if predicted_heart_attack_risk == 0 and predicted_percentage < 50:
            category = "Heart attack risk 0, heart attack risk percentage less than 50. You are safe. Please take care of yourself."
        elif predicted_heart_attack_risk == 0 and predicted_percentage >= 50:
            category = "Heart attack risk 0, heart attack risk percentage greater than 50. Please consult the doctor by sharing the report."
        elif predicted_heart_attack_risk == 1 and predicted_percentage < 50:
            category = "Heart attack risk 1, heart attack risk percentage less than 50. Something unpredictable. Please consult a doctor."
        else:
            category = "Heart attack risk 1, heart attack risk percentage greater than 50. Don't be afraid. Just contact the nearest hospital. That's all."

        return render_template('result.html', heart_attack_risk=predicted_heart_attack_risk,
                               percentage=predicted_percentage, messages=messages, category=category)
    else:
        return render_template('ml.html')

@app.route('/download_report', methods=['POST'])
def download_report():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return "Session expired. Please log in again."

        user_details = get_user_details(email)
        if not user_details:
            return "User details not found in the database."

        report_content = request.form['report_content']

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=14,
        )

        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            fontSize=12,
            leading=14,
            spaceAfter=10,
        )

        elements = []

        elements.append(Paragraph("Heart Attack Risk Report", title_style))
        elements.append(Spacer(1, 12))

        user_info = f"""
        <b>User Details:</b><br/>
        Age: {user_details['age']}<br/>
        Sex: {user_details['sex']}<br/>
        Diabetes: {user_details['diabetes']}<br/>
        Family History: {user_details['famhistory']}<br/>
        Smoking: {user_details['smoking']}<br/>
        Obesity: {user_details['obesity']}<br/>
        Alcohol Consumption: {user_details['alcohol']}<br/>
        Exercise Hours Per Week: {user_details['exercise_hours']}<br/>
        Diet: {user_details['diet']}<br/>
        Previous Heart Problems: {user_details['heart_problem']}<br/>
        BMI: {user_details['bmi']}<br/>
        Physical Activity Days Per Week: {user_details['physical_activity']}<br/>
        Sleep Hours Per Day: {user_details['sleep_hours']}<br/>
        Blood Pressure (Systolic/Diastolic): {user_details['blood_pressure_systolic']}/{user_details['blood_pressure_diastolic']}<br/>
        """

        elements.append(Paragraph(user_info, body_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(report_content, body_style))
        elements.append(Spacer(1, 12))

        image_path = 'static/final_graph.png'
        elements.append(Image(image_path, width=4 * inch, height=4 * inch))
        elements.append(Spacer(1, 12))

        doc.build(elements)

        pdf_buffer.seek(0)
        response = Response(pdf_buffer, mimetype='application/pdf')
        response.headers.set("Content-Disposition", "attachment", filename="heart_attack_report.pdf")

        return response
    else:
        return "Method not allowed."

if __name__ == '__main__':
    app.run(debug=True)
