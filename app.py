# from flask import Flask, request, render_template
# import pickle
# # app.py er upore import koro ba define koro
# from my_module import TextCleaner

# # Import or define your TextCleaner (same as when training) 
# from sklearn.base import BaseEstimator, TransformerMixin

# class TextCleaner(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         # Make sure this matches your training logic!
#         return [x.lower().strip() for x in X]

# app = Flask(__name__)

# # Load your trained pipeline. 
# model = pickle.load(open("final_pipeline.pkl", "rb"))

# @app.route("/")
# def home():
#     result = ""
#     model_preds = {'lr': '-', 'knn': '-', 'nb': '-'}
#     majority_msg = ""
#     return render_template("index.html", 
#                            result=result, 
#                            model_preds=model_preds, 
#                            majority_msg=majority_msg)

# @app.route("/predict", methods=["POST"])
# def predict():
#     result = ""
#     model_preds = {}
#     majority_msg = ""

#     user_text = request.form.get("input_text")
#     if user_text:
#         # Step 1: Clean the input text using the pipeline's cleaning step
#         cleaner = model.named_steps['cleaner']
#         vectorizer = model.named_steps['vectorizer']
#         voting_clf = model.named_steps['classifier']

#         # Apply text cleaning as done in training
#         cleaned = cleaner.transform([user_text])
#         # Vectorize the cleaned text
#         features = vectorizer.transform(cleaned)

#         pred_labels = []
#         for key, clf in voting_clf.named_estimators_.items():
#             pred = clf.predict(features)[0]
#             label = "Positive" if pred == 1 else "Negative"
#             model_preds[key] = label
#             pred_labels.append(label)

#         # Majority vote
#         majority = max(set(pred_labels), key=pred_labels.count)
#         count = pred_labels.count(majority)
#         majority_msg = f"{count} out of {len(pred_labels)} models predict '{majority}'"
#         result = majority
#     else:
#         result = "Please enter some text!"
#         model_preds = {'lr': '-', 'knn': '-', 'nb': '-'}
#         majority_msg = ""

#     return render_template("index.html",
#                            result=result,
#                            model_preds=model_preds,
#                            majority_msg=majority_msg)

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Define your custom transformer (same as during training)
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [x.lower().strip() for x in X]

app = Flask(__name__)

# Load your trained pipeline using joblib
model = joblib.load("final_pipeline.pkl")

@app.route("/")
def home():
    result = ""
    model_preds = {'lr': '-', 'knn': '-', 'nb': '-'}
    majority_msg = ""
    return render_template("index.html", 
                           result=result, 
                           model_preds=model_preds, 
                           majority_msg=majority_msg)

@app.route("/predict", methods=["POST"])
def predict():
    result = ""
    model_preds = {}
    majority_msg = ""

    user_text = request.form.get("input_text")
    if user_text:
        # Step 1: Clean the input text using the pipeline's cleaning step
        cleaner = model.named_steps['cleaner']
        vectorizer = model.named_steps['vectorizer']
        voting_clf = model.named_steps['classifier']

        # Apply text cleaning as done in training
        cleaned = cleaner.transform([user_text])
        # Vectorize the cleaned text
        features = vectorizer.transform(cleaned)

        pred_labels = []
        for key, clf in voting_clf.named_estimators_.items():
            pred = clf.predict(features)[0]
            label = "Positive" if pred == 1 else "Negative"
            model_preds[key] = label
            pred_labels.append(label)

        # Majority vote
        majority = max(set(pred_labels), key=pred_labels.count)
        count = pred_labels.count(majority)
        majority_msg = f"{count} out of {len(pred_labels)} models predict '{majority}'"
        result = majority
    else:
        result = "Please enter some text!"
        model_preds = {'lr': '-', 'knn': '-', 'nb': '-'}
        majority_msg = ""

    return render_template("index.html",
                           result=result,
                           model_preds=model_preds,
                           majority_msg=majority_msg)

if __name__ == "__main__":
    app.run(debug=True)
