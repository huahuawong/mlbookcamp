import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model1.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

dict_file = 'dv.bin'
with open(dict_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    x = dv.transform([customer])
    y_pred = model.predict_proba(x)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'Churn probability': float(y_pred),
        'Churn': bool(churn)
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
# print('input', customer)
# print('churn probability', y_pred)