from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pickle
import pytesseract
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import logging
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Set the path to the Tesseract executable
# IMPORTANT: Replace with your actual Tesseract installation path if different.
# Typically: C:\Program Files\Tesseract-OCR\tesseract.exe on Windows.
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Dell\Downloads\Tesseract-OCR\tesseract.exe'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create Flask app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a_super_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database'), exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables to store models, vectorizer, and news data
models = {}
vectorizer = None
news_df = None

# Initialize the pre-trained model for image classification
image_model = ResNet50(weights='imagenet', include_top=False)

# Add min and max to Jinja2 environment
app.jinja_env.globals.update(min=min, max=max)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(10), nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Changed to nullable=True
    confidence = db.Column(db.Float, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        cv2.imwrite(image_path, gray)
        return True
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return False

def extract_text_from_image(image_path):
    try:
        if not preprocess_image(image_path):
            raise ValueError("Image preprocessing failed")
        text = pytesseract.image_to_string(Image.open(image_path))
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            raise ValueError("No text could be extracted from the image")
        return text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH. Image processing skipped.")
        return "" # Return empty string if Tesseract is not found
    except Exception as e:
        logging.error(f"Error in OCR: {str(e)}")
        raise

def load_models():
    global models, vectorizer, news_df
    # Check if models, vectorizer, and news_df are already loaded
    if models is not None and vectorizer is not None and news_df is not None:
        logging.info("Models, vectorizer, and news data already loaded.")
        return models, vectorizer, news_df

    try:
        logging.info("Attempting to load models and vectorizer...")
        loaded_models = {}
        model_files = {
            'logistic': 'logistic_model.pkl',
            'svm': 'svm_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'naive_bayes': 'naive_bayes_model.pkl',
            'knn': 'knn_model.pkl',
            'decision_tree': 'decision_tree_model.pkl',
            'adaboost': 'adaboost_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'extra_trees': 'extra_trees_model.pkl',
            'xgboost': 'xgboost_model.pkl'
        }

        for model_name, file_name in model_files.items():
            try:
                model_path = os.path.join('models', file_name)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        loaded_models[model_name] = pickle.load(f)
                        logging.info(f"Successfully loaded {model_name}.")
                else:
                    logging.warning(f"Model file not found: {model_path}")

            except Exception as e:
                logging.error(f"Error loading model {model_name}: {str(e)}")
                continue

        if not loaded_models:
            raise FileNotFoundError("No models could be loaded. Please train the models first by running train_models.py.")

        vectorizer_path = os.path.join('models', 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
             with open(vectorizer_path, 'rb') as f:
                loaded_vectorizer = pickle.load(f)
                logging.info("Vectorizer loaded successfully.")
        else:
             raise FileNotFoundError("Vectorizer file not found: models/vectorizer.pkl. Please train the models first.")



        # Load news data for related articles
        news_data_path = os.path.join('data', 'news.csv')
        if os.path.exists(news_data_path):
            logging.info("Loading news data for related articles...\nMake sure 'data/news.csv' is available.")
            loaded_news_df = pd.read_csv(news_data_path)
            loaded_news_df.dropna(subset=['text', 'label'], inplace=True)
            logging.info(f"News data loaded successfully. {len(loaded_news_df)} articles.")
        else:
             logging.warning(f"News data file not found: {news_data_path}. Related news functionality will be limited.")
             loaded_news_df = pd.DataFrame()


        # Assign to global variables
        models = loaded_models
        vectorizer = loaded_vectorizer
        news_df = loaded_news_df

        return models, vectorizer, news_df

    except Exception as e:
        logging.error(f"Critical error loading models, vectorizer, or news data: {str(e)}")
        # Set to empty dict/None to indicate failure but avoid UndefinedError
        models = {} # Set to empty dict on failure
        vectorizer = None
        news_df = None
        # The application can still run, but predictions will fail if models are not loaded.
        pass # Continue running but with broken prediction

def predict_from_image(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get image features
        features = image_model.predict(x)
        
        # Simple heuristic: if the image contains text-like features, it's more likely to be real news
        # This is a simplified approach - you might want to train a specific model for news images
        text_score = np.mean(features[0])
        
        # Convert score to prediction
        if text_score > 0.5:
            return 'REAL', 0.8
        else:
            return 'FAKE', 0.7
            
    except Exception as e:
        logging.error(f"Error in image prediction: {str(e)}")
        return None, None

def get_related_news(text, result):
    """Get related news articles based on the prediction result."""
    global news_df, vectorizer # Use global variables
    if news_df is None or vectorizer is None or news_df.empty:
         logging.warning("News data, vectorizer, or empty news data. Cannot get related news.")
         return []

    try:
        # Load the dataset (use the pre-loaded one)
        df = news_df.copy() # Use a copy to avoid modifying the global DataFrame

        # If the prediction is FAKE, find similar REAL news
        if result == 'FAKE':
            df = df[df['label'] == 'REAL']
        else:
            df = df[df['label'] == 'FAKE']

        if df.empty:
            logging.warning(f"No {result} news found in the dataset for related news after filtering.")
            return []

        # Use TF-IDF to find similar articles (use the pre-loaded vectorizer)
        # Note: The vectorizer is trained on the *training* data, not just the news.csv used here for related news. This might affect similarity results.
        # For ideal related news, a vectorizer trained specifically on the news.csv used for related news would be better, but this reuses the main vectorizer for simplicity.
        try:
            tfidf_matrix = vectorizer.transform(df['text'])
            query_vector = vectorizer.transform([text])
        except Exception as e:
             logging.error(f"Error transforming text or news data for related news: {str(e)}")
             return []


        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        # Get top 3 similar articles, excluding the query text itself if it\'s in the dataset
        # Simple approach: get top N and filter if needed, here just top 3 by similarity
        similar_indices = cosine_similarities.argsort()[-3:][::-1]

        # Filter out results below a certain similarity threshold if desired
        # filtered_indices = [i for i in similar_indices if cosine_similarities[i] > 0.5] # Example threshold


        return df.iloc[similar_indices][['text', 'label']].to_dict('records')
    except Exception as e:
        logging.error(f"Error getting related news: {str(e)}")
        return []


# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']

            if User.query.filter_by(username=username).first():
                flash('Username already exists. Please login.')
                return redirect(url_for('login'))

            if User.query.filter_by(email=email).first():
                flash('Email already registered. Please login.')
                return redirect(url_for('login'))

            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            flash('An error occurred during registration: ' + str(e))
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username).first()

            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('home'))

            flash('Invalid username or password')
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            flash('An error occurred during login: ' + str(e))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    global models, vectorizer
    if not models or not vectorizer:
        models, vectorizer, _ = load_models()
        if not models or not vectorizer:
            flash('Application models failed to load. Please try again later.', 'error')
            return redirect(url_for('home'))

    if request.method == 'POST':
        try:
            model_choice = request.form.get('model')
            text = request.form.get('text', '').strip()
            prediction = None
            confidence = None
            prediction_label = None
            image_path = None

            if not model_choice:
                flash('Please select a model.', 'error')
                return redirect(url_for('predict'))

            # Handle image upload
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename and allowed_file(file.filename):
                    try:
                        filename = secure_filename(file.filename)
                        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(image_path)
                        
                        # Try image prediction first
                        prediction_label, confidence = predict_from_image(image_path)
                        
                        # If image prediction fails, try text extraction
                        if prediction_label is None:
                            try:
                                text = extract_text_from_image(image_path)
                                if text:
                                    # Use text-based prediction if text extraction succeeds
                                    features = vectorizer.transform([text])
                                    model = models[model_choice]
                                    
                                    if hasattr(model, 'predict_proba'):
                                        proba = model.predict_proba(features)[0]
                                        prediction = model.predict(features)[0]
                                        confidence = float(proba[1] if prediction == 1 else proba[0])
                                    else:
                                        prediction = model.predict(features)[0]
                                        confidence = 0.5
                                    
                                    prediction_label = 'REAL' if prediction == 1 else 'FAKE'
                            except Exception as e:
                                logging.error(f"Error extracting text from image: {str(e)}")
                                flash('Error processing image. Please try again.', 'error')
                                return redirect(url_for('predict'))
                    except Exception as e:
                        logging.error(f"Error saving image: {str(e)}")
                        flash('Error processing image. Please try again.', 'error')
                        return redirect(url_for('predict'))

            # If no image or image prediction failed, use text-based prediction
            if not prediction_label and text:
                try:
                    features = vectorizer.transform([text])
                    model = models[model_choice]

                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)[0]
                        prediction = model.predict(features)[0]
                        confidence = float(proba[1] if prediction == 1 else proba[0])
                    elif hasattr(model, 'decision_function'):
                        decision_scores = model.decision_function(features)
                        prediction = model.predict(features)[0]
                        confidence = float(1 / (1 + np.exp(-np.abs(decision_scores)))[0])
                    else:
                        prediction = model.predict(features)[0]
                        confidence = 0.5

                    prediction_label = 'REAL' if prediction == 1 else 'FAKE'
                except Exception as e:
                    logging.error(f"Error during model prediction: {str(e)}")
                    flash('Error during prediction. Please try again.', 'error')
                    return redirect(url_for('predict'))

            if not prediction_label:
                flash('Please provide news text or upload an image.', 'warning')
                return redirect(url_for('predict'))

            # Ensure confidence is a valid float
            confidence = float(confidence)
            if not (0 <= confidence <= 1):
                confidence = 0.5

            # Prepare prediction data for visualization
            prediction_data = {
                'labels': ['FAKE', 'REAL'],
                'values': [0, 0]
            }
            
            # Scale confidence to percentage
            confidence_percentage = float(confidence * 100)
            
            if prediction_label == 'FAKE':
                prediction_data['values'][0] = confidence_percentage
                prediction_data['values'][1] = 100 - confidence_percentage
            else:
                prediction_data['values'][1] = confidence_percentage
                prediction_data['values'][0] = 100 - confidence_percentage

            # Get related news
            related_news = get_related_news(text if text else "Image content", prediction_label)

            # Save prediction to database
            result_db = Prediction(
                text=text if text else "Image content",
                result=prediction_label,
                model_used=model_choice,
                user_id=current_user.id,
                confidence=confidence
            )
            db.session.add(result_db)
            db.session.commit()

            # Clean up uploaded image if it exists
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    logging.error(f"Error removing temporary image file: {str(e)}")

            return render_template('result.html',
                               prediction=prediction_label,
                               confidence=confidence_percentage,
                               model_used=model_choice,
                               text=text if text else "Image content",
                               related_news=related_news,
                               processing_time=0.0,
                               prediction_data=prediction_data)

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            flash('An error occurred during prediction. Please try again.', 'error')
            return redirect(url_for('predict'))

    return render_template('predict.html', models=models)

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of predictions per page
    
    # Get paginated predictions for the current user
    pagination = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.timestamp.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    predictions = pagination.items
    
    # Format the predictions for display
    formatted_predictions = []
    for pred in predictions:
        formatted_predictions.append({
            'date': pred.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'text': pred.text,
            'prediction': pred.result,
            'confidence': round(pred.confidence * 100, 2) if pred.confidence is not None else 0
        })
    
    return render_template('history.html',
                         predictions=formatted_predictions,
                         page=page,
                         total_pages=pagination.pages)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied')
        return redirect(url_for('home'))

    users = User.query.all()
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('admin.html', users=users, predictions=predictions)

@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@login_required
def toggle_admin(user_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'})

    try:
        user = User.query.get_or_404(user_id)
        user.is_admin = not user.is_admin
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error toggling admin status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Load models, vectorizer, and news data when the app starts
        models, vectorizer, news_df = load_models() # Explicitly assign return values

    app.run(debug=True)
