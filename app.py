from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('sms_spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sms_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form['sms']
    vectorized_sms = vectorizer.transform([sms])
    prediction = model.predict(vectorized_sms)[0]

    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('result.html', sms=sms, result=result)

@app.route('/bulk', methods=['GET', 'POST'])
def bulk():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            import pandas as pd

            data = pd.read_csv(file)
            data['Prediction'] = model.predict(vectorizer.transform(data['message']))
            data['Prediction'] = data['Prediction'].map({1: 'Spam', 0: 'Not Spam'})
            output = data.to_html(index=False)
            return render_template('upload.html', table=output)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
