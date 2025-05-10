import mysql.connector
from flask import jsonify

def save_user_data(email, age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet,
                   heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic,
                   blood_pressure_diastolic):
    conn = None  # Initialize conn outside the try block

    try:
        # conn = mysql.connector.connect(
        #     host="cardio-care-nagavenisgowdru-bd3e.k.aivencloud.com",
        #     port=21607,
        #     user= "avnadmin",
        #     password="AVNS_kU3Vg3PUMtzarpZ_vdl",
        #     database="cardio_care"
        # )
        conn = mysql.connector.connect(
            host="localhost",
            user= "root",
            password="dbms",
            database="cardio_care"
        )

        cursor = conn.cursor()

        query = """
        INSERT INTO user_details 
        (email, age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet, 
        heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic, blood_pressure_diastolic) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (email, age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet,
                  heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic,
                  blood_pressure_diastolic)

        cursor.execute(query, values)
        conn.commit()
        

    except Exception as e:
        # If an exception occurs, return the error message
        return jsonify({'message': str(e)}), 500

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
