# Library imports
import pickle
import pandas as pd
import json
from flask import Flask, request


app = Flask(__name__)
app.config["DEBUG"] = True

def load_model():
    # load model
    pickle_classifier = open('../models/LGBMClassifier.pkl','rb')
    clf=pickle.load(pickle_classifier)
    return clf

# load data
df_data = pd.read_csv('../datas/sample_preproc.csv', index_col='SK_ID_CURR')

@app.route('/predict/{customerID}', methods=['GET', 'POST'])
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
    clf = load_model()

    # Making prediction
    y_proba = clf.predict_proba([data])[0][0]

    # Looking for the customer situation (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.320
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Customer score calculation
    score = int(y_class * 100)

    # Customer credit application result
    if customer_class >= threshold_optimized:
        prediction = 1
    else:
        prediction = 0

    # API response to the dashboard
    response = json.dumps(
        {'score': score, 'class': result, 'prediction': prediction})
    return response, 200

if __name__ == '__main__':
    app.run(debug=True)
