import pickle
import pandas as pd
from flask import Flask
from flask import jsonify
import json
from lightgbm import LGBMClassifier
app = Flask(__name__)

@app.route("/")
def home():
    return "Prêt à dépenser API"

def load_data():
    # load datas
    # data of the customers preprocessed (imputed, normalized)
    sample = pd.read_csv('sample_preproc.csv.zip', index_col='SK_ID_CURR')

    return sample

def load_model():
    # load model
    pickle_classifier = open('LGBMClassifier.pkl','rb')
    clf=pickle.load(pickle_classifier)

    return clf

def load_prediction(sample, id, clf):
    X=sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score

@app.route('/predict/<customer_id>', methods=['GET','POST'])
def predict(customer_id):

    customer_id = str(customer_id)
    clf = load_model()
    sample = load_data()

    prediction = load_prediction(sample, customer_id, clf)

    # Compute decision according to the best threshold
    if prediction >= 0.35:
        decision = "Prêt Accordé 🎉😎"
    else:
        decision = "Prêt Rejeté 😥🤯"

    response = json.dumps(
        {'decision': decision, 'prediction': round(float(prediction)*100, 2)}, ensure_ascii=False)

    return response


if __name__ == '__main__':
    app.run()
