# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import SGDClassifier
from flask import Flask, request, render_template_string
import joblib
#nltk.download('stopwords')
""""
# extracting the data
df = pd.read_excel("DM_Report.xlsx", engine="openpyxl")
print("Initial shape:", df.shape)

# Drop rows with missing
df.dropna(subset=['Category'], inplace=True)

# Removing exact duplicates on (Description + Category + Sub-category)
df.drop_duplicates(subset=['Description', 'Category', 'Sub-Category'], inplace=True)

# Cleaning Description
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(word for word in text.split() if word not in stop_words)

df['Cleaned_Description'] = df['Description'].apply(clean_text)

# cleaned descriptions to avoid label noise
df.drop_duplicates(subset=['Cleaned_Description'], inplace=True)

# Label Encoding
le_cat = LabelEncoder()
le_subcat = LabelEncoder()

df['Category_Label'] = le_cat.fit_transform(df['Category'])
df['Subcategory_Label'] = le_subcat.fit_transform(df['Sub-Category'])

# Features and Labels
X_text = df['Cleaned_Description']
y = df[['Category_Label', 'Subcategory_Label']]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['Cleaned_Description'])

# Separate targets
y_cat = df['Category_Label']
y_sub = df['Subcategory_Label']

# ------------------------
# CATEGORY MODEL
# ------------------------
ros_cat = RandomOverSampler(random_state=42)
X_cat, y_cat_resampled = ros_cat.fit_resample(X, y_cat)

X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y_cat_resampled, test_size=0.2, random_state=42)

model_cat = LogisticRegression(max_iter=1000)
model_cat.fit(X_cat_train, y_cat_train)

y_cat_pred = model_cat.predict(X_cat_test)
print("\n Category Classification Report:\n", classification_report(y_cat_test, y_cat_pred))

# ------------------------
# SUBCATEGORY MODEL
# ------------------------
ros_sub = RandomOverSampler(random_state=42)
X_sub, y_sub_resampled = ros_sub.fit_resample(X, y_sub)

X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X_sub, y_sub_resampled, test_size=0.2, random_state=42)

model_sub = SGDClassifier(loss='log', max_iter=1000, random_state=42)
model_sub.fit(X_sub_train, y_sub_train)

y_sub_pred = model_sub.predict(X_sub_test)
print("\n Sub-category Classification Report:\n", classification_report(y_sub_test, y_sub_pred))


# Save encoders and model for API later
joblib.dump(model_cat, 'model_category.pkl')
joblib.dump(model_sub, 'model_subcat.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le_cat, 'label_encoder_cat.pkl')
joblib.dump(le_subcat, 'label_encoder_sub.pkl')
""""
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
