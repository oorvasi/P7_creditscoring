import pickle
import pandas as pd
from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Prêt à dépenser API"

def load_model():
    # load model
    pickle_classifier = open('LGBMClassifier.pkl','rb')
    clf=pickle.load(pickle_classifier)
    return clf

@app.get('/{customerID}')
def predict():
    """
    This function is used for making prediction.
    """

    # Input data from dashboard request
    request_json = request.get_json()
    print(request_json)
    data = []
    for key in request_json.keys():
        data.append(request_json[key])

    # Loading the model
    model = load_model()

    # Making prediction
    y_proba = model.predict_proba([data])[0][0]

    # Looking for the customer situation (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.36
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Customer score calculation
    score = int(y_class * 100)

    # Customer credit application result
    if customer_class == 1:
        result = 'à risque'
        status = 'refusée'
    else:
        result = 'sans risque'
        status = 'acceptée'

    # API response to the dashboard
    response = json.dumps(
        {'score': score, 'class': result, 'application': status})
    return response, 200


if __name__ == '__main__':
    app.run()
