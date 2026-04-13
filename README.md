# Advanced Business Analytics - Notebook Flow

## Notebook Order

1. **Dataset Preparation** (`dataset_preparation.ipynb`)  
   Prepares and cleans the raw dataset for analysis and modeling.

2. **EDA** (`eda.ipynb`)  
   Explores the data and patterns; includes initial feature engineering work.

3. **Feature Engineering** (`feature_engineering.ipynb`)  
   Consolidates feature engineering logic into reusable functions for the pipeline.

4. **Feature Selection** (`feature_selection.ipynb`)  
   Selects the most relevant features for model training.

5. **Model Development** (`model_development.ipynb`)  
   Trains and evaluates machine learning models.

6. **Explainability and Bias** (`explainability_and_bias.ipynb`)  
   Reviews model explainability outputs and bias checks.

## Run Live Simulation and Scoring

From the `Advanced_Business_Analytics` folder:

```powershell
python simulate_live_data.py --input-csv data/hotel_bookings_live_sample_300.csv --output-dir data/simulations
```

This command uses `data/hotel_bookings_live_sample_300.csv`, creates fake future arrival dates to simulate incoming live data, and writes a new CSV file in `data/simulations/`. It prints the path and name of the new file created.

Then score the newest generated simulation file:

Copy the path printed

```powershell
python score_bookings.py --input-csv path/to/csv
```
