# Fake News Detection

This project provides a web application for detecting fake news using various AI models.

## Setup Instructions

### 1. Clone the repository:

```bash
git clone <repository_url>
cd fake_news_detection
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR:

Tesseract is required for processing images.

#### For Windows:
1. Download the Tesseract installer from [here](https://tesseract-ocr.github.io/tessdoc/Downloads.html).
2. Run the installer and follow the on-screen instructions.
3. Make sure to select "Install for all users" and keep the default installation path (usually `C:\Program Files\Tesseract-OCR`).
4. Add the Tesseract installation directory to your system's PATH environment variable. Typically, this is `C:\Program Files\Tesseract-OCR`. You might need to add `C:\Program Files\Tesseract-OCR\tesseract.exe` to your PATH as well.

#### For macOS/Linux:
Refer to the official Tesseract documentation for installation instructions: [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)

### 5. Initialize the database:

```bash
python init_db.py
```

### 6. Train the models (optional, if models are not already provided):

```bash
python train_models.py
```

### 7. Run the application:

```bash
python app.py
```

The application should now be accessible at `http://127.0.0.1:5000/`.

## Usage

- Enter news text or upload an image to predict its authenticity.
- Select an AI model for prediction.

## Project Structure

- `app.py`: Main Flask application file.
- `requirements.txt`: Python dependencies.
- `templates/`: HTML templates.
- `static/`: Static files (CSS, JS, images).
- `models/`: Directory for trained AI models.
- `data/`: Directory for datasets.
- `train_models.py`: Script for training models.
- `init_db.py`: Script for initializing the database.
- `combine_news.py`: Script for combining news datasets.
- `prepare_dataset.py`: Script for preparing datasets.
- `test_news.py`: Contains tests for news processing.
- `test_predictions.py`: Contains tests for predictions.
- `app.log`: Application logs.
- `training.log`: Training logs. 