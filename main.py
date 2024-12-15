from flask import Flask, request, render_template, jsonify
import torch
import pickle
import numpy as np
from api.source import forecast

app = Flask(__name__)

with open('./data/source_data.pkl', 'rb') as f:
    df = pickle.load(f)

@app.route('/basic-end-to-end-forecasting', methods=['POST'])
def api():
    if request.method == "POST":
        city_name = request.form.get('city_name', '臺北市')
        model_name = request.form.get('model_name', 'GPT')
        lookback = int(request.form.get('lookback', 150))
        num_of_days = int(request.form.get('num_of_days', 30))
        device = request.form.get('device', 'cpu')

    device = torch.device(device)
    params = {
        'lookback': lookback,
        'config': {}
    }
    forecast_result, text, training_data, (start_date, end_date) = forecast(
        df=df,
        city_name=city_name, 
        model_name=model_name,
        params=params,
        num_of_days=num_of_days,
        device=device,
        model_params_directory='./output/result/'
        # depending on the file relative to the output directory
    )        
    if model_name == "GPT":
        forecast_result_list = forecast_result.tolist()  # Convert ndarray to list
    else:
        forecast_result_list = [ele[0] for ele in forecast_result]
    response = {
        
        'forecast_result': forecast_result_list,
        'text': text,
        'training_data': training_data,
        "training_starting_date": start_date,
        "training_ending_date": end_date,
        "notes":  \
            '''
                training_starting_date is the earliest date for model training.
                training_ending_date is the latest date for model training.
                for testing or predicting, the starting date comes one day after the training_ending_date.    
            '''
    }

    # Return as JSON response
    return jsonify(response)

@app.route('/')
def index():
    return render_template('form.html')

if __name__ == '__main__':
    app.run(port=8000, debug=True)