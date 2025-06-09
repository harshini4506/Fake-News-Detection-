import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
import shutil
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def safe_create_models_dir():
    """Safely create models directory."""
    try:
        # Try to remove existing directory
        if os.path.exists('models'):
            try:
                shutil.rmtree('models')
            except PermissionError:
                print("Warning: Could not remove existing models directory. Will try to use existing directory.")
                return
        # Create new directory
        os.makedirs('models')
    except Exception as e:
        print(f"Warning: Could not create models directory: {str(e)}")
        print("Will try to use existing directory.")

def preprocess_text(text):
    """Enhanced text preprocessing."""
    # Ensure text is a string
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Split into words (simple whitespace tokenization)
    words = text.split()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def save_model(model, model_name):
    """Safely save a model."""
    try:
        model_path = f'models/{model_name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Warning: Could not save {model_name} model: {str(e)}")
        return False

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models with enhanced feature extraction."""
    # Initialize TF-IDF Vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    # Transform the text data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save the vectorizer
    try:
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    except Exception as e:
        print(f"Warning: Could not save vectorizer: {str(e)}")
        return {}
    
    # Initialize all 10 models
    models = {
        'logistic': LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced'
        ),
        'svm': LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            dual=False
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine'
        ),
        'naive_bayes': MultinomialNB(
            alpha=0.1
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        'adaboost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    
    # Train and evaluate each model
    results = {}
    model_names = [
        'logistic', 'svm', 'random_forest', 'knn', 'naive_bayes',
        'decision_tree', 'adaboost', 'gradient_boosting', 'extra_trees', 'xgboost'
    ]
    
    for model_name in model_names:
        print(f"\nTraining {model_name} model...")
        model = models.get(model_name)
        
        if model is None:
            print(f"Warning: Model {model_name} not found in the models dictionary. Skipping.")
            continue

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train the model
        model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        if save_model(model, model_name):
            results[model_name] = accuracy
    
    return results

def main():
    print("Setting up models directory...")
    safe_create_models_dir()

    print("Loading and preprocessing data...")
    data_path = os.path.join('data', 'news.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please make sure 'news.csv' is in the 'data' directory.")
        return
    df = pd.read_csv(data_path)

    # Ensure labels are in the correct format (0 for FAKE, 1 for REAL) if they are strings
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

    # Drop rows with missing labels or text
    df.dropna(subset=['text', 'label'], inplace=True)

    # Convert label column to integer type
    df['label'] = df['label'].astype(int)
    
    # Apply text preprocessing
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Use processed text for training
    X = df['processed_text']
    y = df['label']
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    if results:
        print("\nTraining complete! Model accuracies:")
        for model_name, accuracy in results.items():
            print(f"{model_name}: {accuracy:.4f}")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 