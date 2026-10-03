# A3 - Predicting Car Price III

Name: Kan Lin  
Student ID: st127119

This assignment changes the car selling price problem into a 4-class classification problem.

I use the same car dataset and cleaning steps from my previous assignments.

## Task 1

I use Multinomial Logistic Regression and add these classification metrics from scratch:

- Accuracy
- Precision
- Recall
- F1-score
- Macro average
- Weighted average

I also compare my results with scikit-learn classification report.

## Task 2

I add Ridge or L2 penalty to Logistic Regression.

I try different lambda values and compare the results.

## Task 3

I log the experiments to the class MLflow server.

Experiment name:

`st127119-a3`

Registered model name:

`st127119-a3-model`

The web app uses the model from Staging.

## Files

- `KanLin_st127119_A3_PredictingCarPrices.ipynb` - Assignment notebook
- `KanLin_st127119_A3_PredictingCarPrices.pdf` - PDF version of the notebook
- `Cars.csv` - Car price dataset
- `app` - Dash web application and Docker files
- `tests` - Model unit tests
- `.github/workflows/ci-cd.yml` - GitHub Actions for testing and deployment

## Run the Tests

```bash
pip install -r app/code/requirements.txt
pip install pytest
pytest -q
```

## Run the Web App

The app loads the registered MLflow model.

Set the MLflow login first.

Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_USERNAME="admin"
$env:MLFLOW_TRACKING_PASSWORD="<MLFLOW_PASSWORD>"
```

Linux/macOS:

```bash
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD="<MLFLOW_PASSWORD>"
```

Then run:

```bash
docker compose -f app/docker-compose.yaml up --build
```

Open:

`http://127.0.0.1:8050/`
