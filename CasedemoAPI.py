# Imports
import re
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
nltk.download('stopwords')

# Cleaning Description
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(word for word in text.split() if word not in stop_words)
app = Flask(__name__)

# Load models and vectorizers
model_cat = joblib.load("model_category.pkl")
model_sub = joblib.load("model_subcat.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le_cat = joblib.load("label_encoder_cat.pkl")
le_sub = joblib.load("label_encoder_sub.pkl")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DCR Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 40px; background: #f0f2f5; }
        textarea, input[type=submit] {
            font-size: 16px; width: 100%; padding: 10px; margin: 10px 0;
        }
        .result { background: white; padding: 20px; border-radius: 10px; margin-top: 20px; }
        .buttons { display: flex; gap: 10px; }
    </style>
</head>
<body>
    <h1>DCR Category & Sub-category Predictor</h1>
    <form method="POST">
        <textarea name="description" rows="5" placeholder="Enter DCR comment here..." required>{{ request.form.description or '' }}</textarea>
        <div class="buttons">
            <input type="submit" name="action" value="Predict">
            <input type="submit" name="action" value="Clear">
        </div>
    </form>

    {% if category and subcategory %}
    <div class="result">
        <h2>Predictions:</h2>
        <p><strong>Category:</strong> {{ category }}</p>
        <p><strong>Sub-category:</strong> {{ subcategory }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    category = subcategory = None
    description_text = ""

    if request.method == 'POST':
        if request.form['action'] == 'Clear':
            return render_template_string(HTML_TEMPLATE, category=None, subcategory=None, request={"form": {"description": ""}})

        description_text = request.form['description']
        cleaned = clean_text(description_text)
        vect = tfidf.transform([cleaned])

        pred_cat = model_cat.predict(vect)[0]
        pred_sub = model_sub.predict(vect)[0]

        category = le_cat.inverse_transform([pred_cat])[0]
        subcategory = le_sub.inverse_transform([pred_sub])[0]

    return render_template_string(HTML_TEMPLATE, category=category, subcategory=subcategory, request={"form": {"description": description_text}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
