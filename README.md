# Student Performance Predictor

## Overview

Machine learning project to predict student exam performance based on demographic and educational factors.

## Directory Structure

```
src/
├── components/
│   ├── data_ingestion.py # Handles data loading and splitting
│   ├── data_transformation.py # Handles feature preprocessing
│   └── model_trainer.py # Trains and evaluates ML models
├── pipeline/
│   └── predict_pipeline.py # Handles prediction workflow
├── exception.py # Custom exception handling
├── logger.py # Logging configuration
└── utils.py # Utility functions
```

## Installation

```bash
# Create virtual environment(Here I used conda env)
python -m venv my_env
my_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Run Flask application
python app.py
```

Visit http://127.0.0.1:5000/ in your browser

## Key Features

- Data preprocessing with scikit-learn pipelines
- Model training with XGBoost and CatBoost
- Web interface for predictions
- Custom exception handling and logging
- Gradient background UI

## Technical Stack

- Python 3.8
- Flask
- scikit-learn
- XGBoost
- CatBoost
- pandas
- numpy

## API Reference

### Predict Endpoint

POST `/predict_datapoint`

Parameters:
- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course
- reading_score
- writing_score

Returns predicted math score.

## Development

### Logging

Logs are stored in `logs/` directory with format:

```
YYYY_MM_DD_HH_MM_SS.log
```

### Model Training

```python
python src/components/model_trainer.py
```

### Exception Handling

Custom exceptions are logged with:
- Error message
- Error detail
- Line number
- File name

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License