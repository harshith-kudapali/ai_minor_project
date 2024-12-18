from flask import Flask, render_template, request
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the model
model = joblib.load('flight_fare_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from form
        airline = int(request.form['airline'])
        source = int(request.form['source'])
        destination = int(request.form['destination'])

        journey_date = request.form['journey_date']
        dep_time = request.form['dep_time']
        arrival_time = request.form['arrival_time']
        
        # Parse date and time
        journey_day = int(datetime.strptime(journey_date, '%Y-%m-%d').day)
        journey_month = int(datetime.strptime(journey_date, '%Y-%m-%d').month)
        
        dep_hour = int(dep_time.split(':')[0])
        dep_min = int(dep_time.split(':')[1])
        
        arrival_hour = int(arrival_time.split(':')[0])
        arrival_min = int(arrival_time.split(':')[1])
        
        # Other features
        duration_hours = int(request.form['duration_hours'])
        duration_mins = int(request.form['duration_mins'])
        total_stops = int(request.form['total_stops'])
        additional_info = int(request.form['additional_info'])

        # Prepare data for model prediction
        features = np.array([[airline, source, destination, journey_day, journey_month,
                              dep_hour, dep_min, arrival_hour, arrival_min, duration_hours,
                              duration_mins, total_stops, additional_info]])
        
        # Predict fare
        prediction = model.predict(features)[0]
        return render_template('index.html', prediction_text=f'Predicted Fare: ₹{prediction:.2f}')
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
