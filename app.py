from flask import Flask, render_template, request, jsonify
from scraper import scrape_reviews
from model import load_models, classify_reviews
from preprocessing import preprocess_text
import pandas as pd
import os
import traceback
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models at startup
try:
    word2vec_model, svm_model = load_models()#commenting on the models
    print("[DEBUG] Models loaded successfully")
except Exception as e:
    print("[ERROR] Failed to load models:", e)
    traceback.print_exc()
@app.route('/test', methods=['POST'])
def test():
    data = request.json
    print("[DEBUG] /test received:", data)
    return jsonify({"message": "Test successful", "data": data})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')
    print(f"[DEBUG] Received URL: {url}")  # Log the received URL

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Step 1: Scrape reviews
        reviews = scrape_reviews(url)
        print(f"[DEBUG] Scraped {len(reviews)} reviews")  # Log count
        if reviews.empty:
            return jsonify({"error": "No reviews found"}), 404

        if "Review Text" not in reviews.columns or "Rating" not in reviews.columns:
            return jsonify({"error": "Invalid reviews format"}), 400

        # Step 2: Preprocess reviews
        preprocessed_reviews = []
        for i, review in enumerate(reviews["Review Text"]):
            review_text = preprocess_text(review)
            rating = reviews.iloc[i]["Rating"]
            preprocessed_reviews.append({"Review Text": review_text, "Rating": rating})
        print(f"[DEBUG] Preprocessed {len(preprocessed_reviews)} reviews")

        # Step 3: Classify reviews
        predictions = classify_reviews(preprocessed_reviews, word2vec_model, svm_model)
        print(f"[DEBUG] First 5 predictions: {predictions[:5]}")  # Log first few predictions

        # Step 4: Build response
        df = pd.DataFrame({
            "Review": reviews["Review Text"],
            "Rating": reviews["Rating"],
            "Prediction": predictions
        })
        df["Prediction"] = df["Prediction"].map({1: "Fake (Computer Generated)", 0: "Real (Original)"})
        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        print("[ERROR] Exception in /analyze route:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
