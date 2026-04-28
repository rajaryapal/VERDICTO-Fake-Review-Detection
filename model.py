import joblib
from gensim.models import Word2Vec
import numpy as np
from preprocessing import preprocess_text  # Importing text preprocessing function

# Function to load trained models (Word2Vec and SVM)
def load_models():
    """
    Loads the pre-trained Word2Vec model and the SVM model.
    Returns:
        word2vec_model: Pre-trained Word2Vec model.
        svm_model: Pre-trained SVM model.
    """
    word2vec_model = Word2Vec.load(r"D:\Fake-Review-Detection\word2vec_model.model")  # Loading Word2Vec model
    svm_model = joblib.load(r"D:\Fake-Review-Detection\SVM_model.pkl")  # Loading SVM model
    return word2vec_model, svm_model

# Function to classify reviews using Word2Vec and SVM
def classify_reviews(reviews, word2vec_model, svm_model):
    """
    Classifies reviews based on text features, rating, and length using an SVM model.
    Args:
        reviews (list): List of review dictionaries with 'Review Text' and 'Rating'.
        word2vec_model (Word2Vec): Trained Word2Vec model for vectorization.
        svm_model: Trained SVM model for classification.
    Returns:
        list: Predicted labels for the given reviews.
    """
    predictions = []
    
    for review in reviews:
        preprocessed_review = preprocess_text(review['Review Text'])  # Preprocessing text

        # Converting words to vectors using Word2Vec
        words = preprocessed_review.split()
        vectors = np.array([word2vec_model.wv[word] for word in words if word in word2vec_model.wv])
        
        if vectors.size > 0:
            # Averaging word vectors to get a single feature vector
            text_vector = np.mean(vectors, axis=0).reshape(1, -1)
            
            try:
                rating = float(review['Rating'])  # Extracting rating
            except ValueError:
                rating = 3.0  # Default rating if extraction fails

            review_length = len(words)  # Calculating length of the review
            
            rating_vector = np.array([[rating]])  # Converting rating to numpy array
            length_vector = np.array([[review_length]])  # Converting length to numpy array
            
            # Combining all features into a single feature vector
            combined_features = np.hstack([rating_vector, length_vector, text_vector])  

            # Predicting using the trained SVM model
            prediction = svm_model.predict(combined_features)
            predictions.append(prediction[0])
        else:
            predictions.append(0)  # Default prediction when no valid vectors are found

    return predictions

# Main execution: Load models when script runs
if __name__ == "__main__":
    word2vec_model, svm_model = load_models()  # Loading models
