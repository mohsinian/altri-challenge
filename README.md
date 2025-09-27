# Flippit - Real Estate Property Investment Analysis

Flippit is a web application that helps real estate investors identify properties with high flipping potential. It uses machine learning and natural language processing to analyze property data and provide investment recommendations.

## Features

- **Property Scoring**: Scores properties based on their flipping potential
- **Machine Learning**: Uses trained models to predict resale values and renovation costs
- **NLP Analysis**: Analyzes property descriptions to gauge renovation needs
- **Interactive Map**: Visualizes properties on a map with color-coded grades
- **Property Filtering**: Filter properties by price, bedrooms, bathrooms, ROI, and more
- **Detailed Analysis**: Provides detailed investment analysis for each property

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, XGBoost
- **NLP**: NLTK, spaCy, Transformers
- **Frontend**: HTML, CSS, JavaScript, Bootstrap, Leaflet
- **Data Processing**: Pandas, NumPy

## Project Structure

```
flippit/
├── app/                    # Backend application code
│   ├── api.py             # Flask API endpoints
│   ├── models.py          # Machine learning models
│   ├── nlp_processor.py   # Natural language processing
│   ├── scoring.py         # Property scoring logic
│   └── utils.py           # Utility functions
├── web/                   # Frontend code
│   ├── index.html         # Main HTML template
│   └── static/            # Static assets
│       ├── css/           # CSS styles
│       │   └── style.css
│       └── js/            # JavaScript code
│           └── main.js
├── notebooks/             # Jupyter notebooks for analysis
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── run.py               # Application entry point
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:mohsinian/altri-challenge.git
cd altri-challenge
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Data Requirements

The application requires two CSV files:

1. **sold_properties.csv**: Contains historical data of sold properties with:
   - property_id, property_url, formatted_address, city, state, zip_code
   - beds, full_baths, half_baths, sqft, lot_sqft, stories, year_built
   - sold_price, text (property description), latitude, longitude

2. **for_sale_properties.csv**: Contains current listings with:
   - property_id, property_url, formatted_address, city, state, zip_code
   - beds, full_baths, half_baths, sqft, lot_sqft, stories, year_built
   - list_price, text (property description), latitude, longitude, days_on_mls

These files should be placed in a `data` directory at the project root.

## Running the Application

Start the Flask application:
```bash
python run.py
```

The application will be available at `http://localhost:6969`

## Usage

1. **Train Models**:
   - Navigate to the "Train Models" page
   - Click "Train Models" to train the machine learning models using sold properties data
   - This may take a few minutes to complete

2. **View Properties**:
   - On the home page, you'll see properties displayed on a map and in a list
   - Properties are color-coded by grade (A=green, B=blue, C=yellow, D=orange, F=red)
   - Click on any property to view detailed investment analysis

3. **Filter Properties**:
   - Use the filter panel to narrow down properties by:
     - Price range
     - Number of bedrooms/bathrooms
     - Minimum ROI
     - Grade (A-F)
     - Neighborhood
     - Renovation level

4. **Upload Custom Data**:
   - Navigate to the "Upload Data" page
   - Upload a CSV file with your own for-sale properties data
   - The system will analyze and score your properties

## How It Works

### Machine Learning Models

The application uses two machine learning models:

1. **Resale Value Prediction Model**:
   - Predicts the after-renovation resale value of a property
   - Uses features like square footage, location, renovation level, and property characteristics
   - Compares multiple algorithms (Linear Regression, Random Forest, Gradient Boosting, XGBoost) and selects the best performer

2. **Renovation Cost Estimation Model**:
   - Estimates the cost of renovations needed for a property
   - Uses property characteristics and NLP-extracted renovation level
   - Also evaluates multiple algorithms to find the best fit

### Natural Language Processing

The NLP processor analyzes property descriptions to:

1. **Extract Renovation Level**:
   - Categorizes properties as "low", "medium", or "high" renovation needs
   - Uses keyword matching to identify renovation requirements

2. **Count Value-Adding Features**:
   - Counts features that increase property value (granite, stainless steel, hardwood floors, etc.)

3. **Sentiment Analysis**:
   - Analyzes the sentiment of property descriptions
   - Uses a pre-trained transformer model for sentiment classification

### Property Scoring

The scoring system evaluates properties based on:

1. **Financial Metrics**:
   - Calculates expected profit after renovation and sale
   - Factors in carrying costs, selling costs, and contingency
   - Computes Return on Investment (ROI)

2. **Risk Assessment**:
   - Evaluates property age, days on market, and renovation level
   - Assigns a risk score from 0-100

3. **Grade Assignment**:
   - Assigns letter grades (A-F) based on ROI and risk
   - A-grade properties have high ROI (>18%) and low risk (<35)
   - F-grade properties have negative ROI or very high risk

## API Endpoints

- `GET /`: Main page
- `POST /api/train`: Train machine learning models
- `POST /api/score`: Score for-sale properties
- `POST /api/score/upload`: Score uploaded properties
- `GET /api/property/<property_id>`: Get score for a specific property
- `GET /api/filters`: Get filter options for the UI
- `POST /api/cache/clear`: Clear the scored properties cache

## Configuration

Configuration settings can be modified in `config.py`:

- `SECRET_KEY`: Flask secret key for session management
- `MODELS_FOLDER`: Directory to store trained models
- `DATA_FOLDER`: Directory containing property data files