import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import mysql.connector

# Load dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Drop unnecessary columns
df1 = df.drop(['Patient ID', 'Income', 'Country', 'Continent', 'Hemisphere', 'Cholesterol'], axis=1)

# Convert 'Sedentary Hours Per Day' to integer
df1['Sedentary Hours Per Day'] = df1['Sedentary Hours Per Day'].astype(int)

# Encode categorical variables
le = LabelEncoder()
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['Diet'] = le.fit_transform(df1['Diet'])

# Split 'Blood Pressure' into 'BP1' and 'BP2'
def split_blood_pressure(blood_pressure):
    return pd.Series(blood_pressure.split('/', 1))

df1[['BP1', 'BP2']] = df1['Blood Pressure'].apply(split_blood_pressure)
df1 = df1.drop(['Blood Pressure', 'Triglycerides', 'Sedentary Hours Per Day'], axis=1)

# Convert 'BP1' and 'BP2' to numeric
df1['BP1'] = pd.to_numeric(df1['BP1'], errors='coerce')
df1['BP2'] = pd.to_numeric(df1['BP2'], errors='coerce')

# Define weights for features
weights = {
    'Age': 0.1,
    'Sex': 0.05,
    'Heart Rate': 0.15,
    'Diabetes': 0.2,
    'Family History': 0.15,
    'Smoking': 0.25,
    'Obesity': 0.2,
    'Alcohol Consumption': 0.1,
    'Exercise Hours Per Week': 0.1,
    'Diet': 0.15,
    'Previous Heart Problems': 0.3,
    'Medication Use': 0.05,
    'Stress Level': 0.2,
    'BMI': 0.15,
    'Physical Activity Days Per Week': 0.1,
    'Sleep Hours Per Day': 0.1,
    'Heart Attack Risk': 0.4,  # This is a composite measure, not a factor itself
    'BP1': 0.25,  # Systolic Blood Pressure
    'BP2': 0.25   # Diastolic Blood Pressure
}

# Modify weights based on conditions
for index, row in df1.iterrows():
    if row['Age'] >= 45:
        weights['Age'] = 0.2
    if row['Sex'] == 0:
        weights['Sex'] = 0.1
    if row['Heart Rate'] < 60:
        weights['Heart Rate'] = 10 + (row['Heart Rate'] - 1) * 0.02
    elif row['Heart Rate'] > 100:
        weights['Heart Rate'] = 0.2 + (row['Heart Rate'] - 100) * 0.02
    if row['BP1'] > 150:
        weights['BP1'] = 0.2 + (row['BP1'] - 150) * 0.02
    if row['BP2'] > 90:
        weights['BP2'] = 0.2 + (row['BP2'] - 90) * 0.02

# Calculate total weighted sum
total_weighted_sum = df1.apply(lambda row: sum(row[col] * weights[col] for col in df1.columns), axis=1)

# Normalize total weighted sum
max_weighted_sum = total_weighted_sum.max()
min_weighted_sum = total_weighted_sum.min()
df1['percentage'] = ((total_weighted_sum - min_weighted_sum) / (max_weighted_sum - min_weighted_sum)) * 100

# Features and target variable
X = df1.drop(columns=['Heart Attack Risk', 'percentage'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_scaled, df1['Heart Attack Risk'])

# Initialize and train Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_scaled, df1['percentage'])

# Function to predict manually
def predict_manually(age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                     exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                     bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2):
    input_data = pd.DataFrame([[age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                                exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                                bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2]],
                              columns=X.columns)
    input_scaled = scaler.transform(input_data)
    predicted_heart_attack_risk = random_forest_classifier.predict(input_scaled)[0]
    predicted_percentage = random_forest_regressor.predict(input_scaled)[0]
    return predicted_heart_attack_risk, predicted_percentage

# Function to get user details from the database
def get_user_details(email):
    conn = mysql.connector.connect(user='root', password='dbms', host='localhost', database='cardio_care')
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet, heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic, blood_pressure_diastolic FROM user_details WHERE email = %s", (email,))
    user_details = cursor.fetchone()
    cursor.close()
    conn.close()
    return user_details
