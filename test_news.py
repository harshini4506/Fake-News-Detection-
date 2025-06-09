import pickle
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model_and_vectorizer(model_name):
    """Load the trained model and vectorizer."""
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            print("Error: Models directory not found. Please run train_models.py first.")
            return None, None

        # Load vectorizer
        vectorizer_path = 'models/vectorizer.pkl'
        if not os.path.exists(vectorizer_path):
            print("Error: Vectorizer not found. Please run train_models.py first.")
            return None, None
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load model
        model_path = f'models/{model_name}_model.pkl'
        if not os.path.exists(model_path):
            print(f"Error: Model {model_name} not found. Available models: logistic, svm, random_forest")
            return None, None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return vectorizer, model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def predict_news(text, vectorizer, model, model_name):
    """Make prediction for a given news text."""
    try:
        # Transform the text
        text_tfidf = vectorizer.transform([text])
        
        # Get prediction
        prediction = model.predict(text_tfidf)[0]
        
        # Get probability/confidence score based on model type
        if model_name == 'svm':
            # For SVM, use decision function and convert to probability
            score = model.decision_function(text_tfidf)[0]
            # Convert decision function score to probability using sigmoid
            probability = 1 / (1 + np.exp(-score))
            # Ensure probability is between 0 and 1
            probability = max(0, min(1, probability))
            # Create probability array [fake_prob, real_prob]
            probability = [1 - probability, probability] if prediction == 1 else [probability, 1 - probability]
        elif model_name == 'logistic':
            # Logistic Regression has predict_proba
            probability = model.predict_proba(text_tfidf)[0]
        else:  # random_forest
            # Random Forest has predict_proba
            probability = model.predict_proba(text_tfidf)[0]
        
        # Ensure probabilities sum to 1
        probability = np.array(probability)
        probability = probability / probability.sum()
        
        return prediction, probability
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def display_model_info():
    """Display information about available models."""
    print("\nModel Information:")
    print("-----------------")
    print("1. Logistic Regression (logistic)")
    print("   - Good for binary classification")
    print("   - Provides probability estimates")
    print("   - Fast training and prediction")
    print("\n2. Support Vector Machine (svm)")
    print("   - Best accuracy in our tests")
    print("   - Good for high-dimensional data")
    print("   - Robust to overfitting")
    print("\n3. Random Forest (random_forest)")
    print("   - Ensemble of decision trees")
    print("   - Good for complex patterns")
    print("   - Less prone to overfitting")
    print("\n4. K-Nearest Neighbors (knn)")
    print("   - Simple and intuitive")
    print("   - Works well with text data")
    print("   - Uses cosine similarity")
    print("\n5. Naive Bayes (naive_bayes)")
    print("   - Fast and efficient")
    print("   - Good for text classification")
    print("   - Works well with high-dimensional data")
    print("\n6. Decision Tree (decision_tree)")
    print("   - Easy to interpret")
    print("   - Can capture non-linear relationships")
    print("   - Good for feature importance analysis")
    print("\n7. AdaBoost (adaboost)")
    print("   - Boosting ensemble method")
    print("   - Good for reducing bias")
    print("   - Works well with weak learners")
    print("\n8. Gradient Boosting (gradient_boosting)")
    print("   - Powerful ensemble method")
    print("   - Good for complex patterns")
    print("   - Often achieves high accuracy")
    print("\n9. Extra Trees (extra_trees)")
    print("   - Random forest variant")
    print("   - More randomization")
    print("   - Good for reducing variance")
    print("\n10. XGBoost (xgboost)")
    print("    - Advanced gradient boosting")
    print("    - High performance")
    print("    - Good for large datasets")

def main():
    print("\nFake News Detection System")
    print("-------------------------")
    
    # Display model information
    display_model_info()
    
    # Model selection
    print("\nSelect a model to use:")
    print("1. Logistic Regression (logistic)")
    print("2. Support Vector Machine (svm) - Recommended")
    print("3. Random Forest (random_forest)")
    print("4. K-Nearest Neighbors (knn)")
    print("5. Naive Bayes (naive_bayes)")
    print("6. Decision Tree (decision_tree)")
    print("7. AdaBoost (adaboost)")
    print("8. Gradient Boosting (gradient_boosting)")
    print("9. Extra Trees (extra_trees)")
    print("10. XGBoost (xgboost)")
    
    while True:
        model_choice = input("\nSelect a model (1-10): ").strip()
        model_map = {
            '1': 'logistic',
            '2': 'svm',
            '3': 'random_forest',
            '4': 'knn',
            '5': 'naive_bayes',
            '6': 'decision_tree',
            '7': 'adaboost',
            '8': 'gradient_boosting',
            '9': 'extra_trees',
            '10': 'xgboost'
        }
        
        if model_choice in model_map:
            model_name = model_map[model_choice]
            break
        print("Invalid choice. Please select 1-10.")
    
    print(f"\nLoading {model_name} model...")
    vectorizer, model = load_model_and_vectorizer(model_name)
    
    if vectorizer is None or model is None:
        return
    
    print("\nModel loaded successfully!")
    print("\nEnter news articles to check if they are real or fake.")
    print("Type 'quit' to exit or 'change' to switch models.")
    print("Type 'example' to see example news articles.")
    
    while True:
        print("\nEnter a news article (or 'quit'/'change'/'example'):")
        text = input("> ")
        
        if text.lower() == 'quit':
            break
        elif text.lower() == 'change':
            return main()  # Restart with model selection
        elif text.lower() == 'example':
            print("\nExample news articles you can try:")
            print("1. 'NASA successfully launched the Artemis mission to the moon today.'")
            print("2. 'Scientists discover that drinking water makes you invisible.'")
            continue
        
        if not text.strip():
            print("Please enter some text!")
            continue
        
        # Make prediction
        prediction, probability = predict_news(text, vectorizer, model, model_name)
        
        if prediction is None:
            print("Failed to make prediction. Please try again.")
            continue
        
        # Display results
        result = "REAL" if prediction == 1 else "FAKE"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        print("\nResults:")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2%}")
        
        # Add some explanation
        if confidence > 0.8:
            print("The model is very confident about this prediction.")
        elif confidence > 0.6:
            print("The model is somewhat confident about this prediction.")
        else:
            print("The model is not very confident about this prediction.")

if __name__ == "__main__":
    main() 