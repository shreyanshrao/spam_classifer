from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)[0]

        label = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', prediction_text=f"This message is: {label}")


if __name__ == '__main__':
    app.run(debug=True)
