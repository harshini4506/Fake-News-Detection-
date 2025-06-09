import pickle
import pandas as pd

def load_model_and_vectorizer():
    # Load the vectorizer
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load the model
    with open('models/logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return vectorizer, model

def predict_news(text, vectorizer, model):
    # Transform the text using the vectorizer
    text_tfidf = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

def main():
    # Sample news articles for testing
    test_news = [
        # Real news example
        "NASA's Perseverance rover successfully landed on Mars on February 18, 2021, marking a historic achievement in space exploration. The rover will search for signs of ancient life and collect samples for future return to Earth.",
        
        # Fake news example
        "Scientists discover that drinking coffee makes you immortal. A new study shows that drinking 10 cups of coffee daily can extend your life indefinitely.",
        
        # Real news example
        "The World Health Organization has approved a new COVID-19 vaccine for emergency use, following rigorous safety and efficacy testing.",
        
        # Fake news example
        "Aliens have been discovered living in the White House basement, according to anonymous sources. The government has been hiding this information for decades."
    ]
    
    print("Loading model and vectorizer...")
    vectorizer, model = load_model_and_vectorizer()
    
    print("\nTesting predictions on sample news articles:")
    print("-" * 80)
    
    for i, news in enumerate(test_news, 1):
        prediction, probability = predict_news(news, vectorizer, model)
        result = "REAL" if prediction == 1 else "FAKE"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        print(f"\nTest {i}:")
        print(f"News: {news[:100]}...")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 80)

if __name__ == "__main__":
    main() 