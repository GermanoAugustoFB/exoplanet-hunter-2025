# 🌌 Exoplanet Hunter 2025

AI-powered prediction of exoplanet habitability using NASA Exoplanet Archive data.

## Overview

Exoplanet Hunter 2025 leverages machine learning to analyze exoplanet data and estimate the probability of habitability. The project features model training, batch predictions, interactive visualizations, and a web interface.

## Features

- Habitability prediction using Random Forest classifier
- Class balancing with SMOTE
- Batch and individual predictions for exoplanets
- Visualization of metrics (confusion matrix, feature importance, label distribution)
- Interactive web app built with Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/exoplanet-hunter-2025.git
   cd exoplanet-hunter-2025
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the model
```bash
python3 src/model_training.py
```

### Run batch predictions
```bash
python3 src/predict.py
```

### Launch the web app
```bash
streamlit run src/app.py
```

## Project Structure

- `src/` - Main scripts
- `graficos/` - Generated charts
- `PythonNasaAppChallenge_VoyIAger/` - NASA data files

## Credits

Developed by Germano, Kevin e Gabriel for NASA Space Apps Challenge 2025.

## License

MIT
