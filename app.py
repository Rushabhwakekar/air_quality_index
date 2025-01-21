from flask import Flask, request
import pickle
import pandas as pd

# Load models and encoders
with open('aqi_model.pkl', 'rb') as f:
    rf_aqi = pickle.load(f)

aux_models = {}
label_encoders = {}
auxiliary_targets = ['Primary Pollutant', 'Health Advisory', 'Suggested Solution']

for col in auxiliary_targets:
    with open(f'{col}_model.pkl', 'rb') as f:
        aux_models[col] = pickle.load(f)
    with open(f'{col}_encoder.pkl', 'rb') as f:
        label_encoders[col] = pickle.load(f)

# Define features
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # HTML form for input (directly in the route, no templates used)
    return '''
    <h1>Air Quality Prediction API</h1>
    <form action="/predict" method="POST">
        <label for="pm25">PM2.5:</label>
        <input type="text" id="pm25" name="PM2.5"><br><br>

        <label for="pm10">PM10:</label>
        <input type="text" id="pm10" name="PM10"><br><br>

        <label for="no">NO:</label>
        <input type="text" id="no" name="NO"><br><br>

        <label for="no2">NO2:</label>
        <input type="text" id="no2" name="NO2"><br><br>

        <label for="nox">NOx:</label>
        <input type="text" id="nox" name="NOx"><br><br>

        <label for="nh3">NH3:</label>
        <input type="text" id="nh3" name="NH3"><br><br>

        <label for="co">CO:</label>
        <input type="text" id="co" name="CO"><br><br>

        <label for="so2">SO2:</label>
        <input type="text" id="so2" name="SO2"><br><br>

        <label for="o3">O3:</label>
        <input type="text" id="o3" name="O3"><br><br>

        <input type="submit" value="Submit">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        no = float(request.form['NO'])
        no2 = float(request.form['NO2'])
        nox = float(request.form['NOx'])
        nh3 = float(request.form['NH3'])
        co = float(request.form['CO'])
        so2 = float(request.form['SO2'])
        o3 = float(request.form['O3'])

        # Create a DataFrame for the input values
        input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]], columns=features)

        # Predict AQI
        predicted_aqi = rf_aqi.predict(input_data)[0]

        # Predict auxiliary targets
        aux_predictions = {}
        for col in auxiliary_targets:
            encoded_pred = aux_models[col].predict(input_data)[0]
            aux_predictions[col] = label_encoders[col].inverse_transform([int(round(encoded_pred))])[0]

        # Return the predictions
        response = f"""
        <h2>Prediction Results</h2>
        <p>Predicted AQI: {predicted_aqi}</p>
        <p>Predicted Primary Pollutant: {aux_predictions['Primary Pollutant']}</p>
        <p>Predicted Health Advisory: {aux_predictions['Health Advisory']}</p>
        <p>Predicted Suggested Solution: {aux_predictions['Suggested Solution']}</p>
        """
        return response

    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True)