from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

data = pd.read_csv("dataset/district wise rainfall normal.csv")

state_encoder = LabelEncoder()
district_encoder = LabelEncoder()

data['STATE_CODE'] = state_encoder.fit_transform(data['STATE_UT_NAME'])
data['DIST_CODE'] = district_encoder.fit_transform(data['DISTRICT'])

X = data[['STATE_CODE','DIST_CODE','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']]
y = data['ANNUAL']

model = RandomForestRegressor()
model.fit(X,y)

states = sorted(data['STATE_UT_NAME'].unique())
districts = sorted(data['DISTRICT'].unique())

@app.route('/')
def home():
    return render_template("index.html", states=states, districts=districts)

@app.route('/predict', methods=['POST'])
def predict():

    state = request.form['state']
    district = request.form['district']

    janfeb = float(request.form['janfeb'])
    marmay = float(request.form['marmay'])
    junsep = float(request.form['junsep'])
    octdec = float(request.form['octdec'])

    state_code = state_encoder.transform([state])[0]
    district_code = district_encoder.transform([district])[0]

    input_data = pd.DataFrame([[state_code,district_code,janfeb,marmay,junsep,octdec]],
                              columns=['STATE_CODE','DIST_CODE','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec'])

    prediction = round(model.predict(input_data)[0],2)

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)