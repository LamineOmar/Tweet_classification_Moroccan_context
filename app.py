import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from cleaning import data_final
app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tweet = request.form.values()
    tweet = ' '.join(tweet)

    #final_features = [np.array(int_features)]
    final_features = data_final(tweet)
    # Use the trained model to predict the probability of the tweet belonging to each class
    prediction = model.predict(final_features)
    prediction = prediction[0]
    
    # Dictionnaire des classes
    classe = {1: "ثقافة", 0: "اقتصاد", 6: "ميديا", 2: "دولي", 4: "سياسة", 5: "مجتمع", 3: "رياضة"}

    
    return render_template('index.html', prediction_text=f'Ce tweet est {classe[prediction]}')
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000,debug=True)