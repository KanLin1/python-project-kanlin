import os

import dash
from dash import dcc, html, Input, Output, State
import mlflow
import pandas as pd


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow.ml.brain.cs.ait.ac.th"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "st127119-a3-model"
)

MODEL_STAGE = os.getenv(
    "MODEL_STAGE",
    "Staging"
)

os.environ.setdefault(
    "MLFLOW_TRACKING_USERNAME",
    "student"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_registry_uri(
    MLFLOW_TRACKING_URI
)

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)


app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1(
        "Car Price Classification"
    ),

    html.P(
        "Enter the car information and click Predict."
    ),

    html.Label("Year"),
    dcc.Input(
        id="year",
        type="number",
        value=2015
    ),
    html.Br(),

    html.Label("Kilometers Driven"),
    dcc.Input(
        id="km_driven",
        type="number",
        value=50000
    ),
    html.Br(),

    html.Label("Fuel"),
    dcc.Dropdown(
        id="fuel",
        options=[
            "Diesel",
            "Petrol"
        ],
        value="Diesel"
    ),

    html.Label("Seller Type"),
    dcc.Dropdown(
        id="seller_type",
        options=[
            "Individual",
            "Dealer",
            "Trustmark Dealer"
        ],
        value="Individual"
    ),

    html.Label("Transmission"),
    dcc.Dropdown(
        id="transmission",
        options=[
            "Manual",
            "Automatic"
        ],
        value="Manual"
    ),

    html.Label("Owner"),
    dcc.Dropdown(
        id="owner",
        options=[
            {
                "label": "First Owner",
                "value": 1
            },
            {
                "label": "Second Owner",
                "value": 2
            },
            {
                "label": "Third Owner",
                "value": 3
            },
            {
                "label": "Fourth & Above Owner",
                "value": 4
            }
        ],
        value=1
    ),

    html.Label("Mileage"),
    dcc.Input(
        id="mileage",
        type="number",
        value=20
    ),
    html.Br(),

    html.Label("Engine"),
    dcc.Input(
        id="engine",
        type="number",
        value=1200
    ),
    html.Br(),

    html.Label("Max Power"),
    dcc.Input(
        id="max_power",
        type="number",
        value=80
    ),
    html.Br(),

    html.Label("Seats"),
    dcc.Input(
        id="seats",
        type="number",
        value=5
    ),
    html.Br(),

    html.Label("Brand"),
    dcc.Input(
        id="brand",
        type="text",
        value="Maruti"
    ),
    html.Br(),

    html.Button(
        "Predict",
        id="predict_button"
    ),

    html.H2(
        id="result"
    )
])


@app.callback(
    Output(
        "result",
        "children"
    ),
    Input(
        "predict_button",
        "n_clicks"
    ),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
    State("brand", "value")
)
def predict_price_class(
    n_clicks,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    max_power,
    seats,
    brand
):
    if not n_clicks:
        return ""

    data = pd.DataFrame([
        {
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "brand": brand
        }
    ])

    prediction = model.predict(
        data
    )

    prediction = pd.DataFrame(
        prediction
    )

    row = prediction.iloc[0]

    price_class = int(
        row["price_class"]
    )

    lower = float(
        row["lower_bound"]
    )

    upper = float(
        row["upper_bound"]
    )

    return (
        "Predicted Price Class: "
        + str(price_class)
        + " | Price Range: "
        + f"{lower:,.0f}"
        + " - "
        + f"{upper:,.0f}"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False
    )
