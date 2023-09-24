from flask import request, render_template
from flask import Flask
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['message']
    if request.method == 'POST':
        # preprocess
        trans_text = transform_text(input_text)
        # vectorize
        vector_text = tfidf.transform([trans_text])
        # predict
        result = model.predict(vector_text)[0]
        if result == 1:
            return render_template('index.html', prediction=1)
        else:
            return render_template('index.html', prediction=0)

tfidf = pickle.load(open('vectorized.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

if __name__ == '__main__':
    app.debug = True
    app.run()
