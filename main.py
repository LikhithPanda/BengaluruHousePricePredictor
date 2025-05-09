import warnings
warnings.filterwarnings('ignore')


import pandas as pd
from click.core import batch
from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
data = pd.read_csv('Cleaned_Data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    total_sqft = float(request.form.get('total_sqft'))

    print(locations, bhk, bath, total_sqft)
    input_df = pd.DataFrame([[locations, bhk, bath, total_sqft]],columns=['location', 'bhk', 'bath', 'total_sqft'])
    prediction = pipe.predict(input_df)[0] * 1e5

    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(debug=True, port=5000)