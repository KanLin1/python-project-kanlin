from pathlib import Path
import sys

import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    ROOT /
    "app" /
    "code"
)

sys.path.insert(
    0,
    str(MODEL_PATH)
)

from model import LogisticRegression


def make_model():
    X = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.2, 0.1],
        [1.0, 0.9, 0.2],
        [1.0, 0.1, 0.9],
        [1.0, 0.9, 0.9]
    ])

    y = np.array([
        0, 1, 2, 3,
        0, 1, 2, 3
    ])

    Y = np.eye(4)[y]

    model = LogisticRegression(
        k=4,
        n=3,
        method="batch",
        alpha=0.1,
        max_iter=20,
        random_state=42
    )

    model.fit(
        X,
        Y
    )

    return model


def test_model_takes_expected_input():
    model = make_model()

    X = np.array([
        [1.0, 0.4, 0.3],
        [1.0, 0.8, 0.7]
    ])

    prediction = model.predict(X)

    assert np.all(
        np.isin(
            prediction,
            [0, 1, 2, 3]
        )
    )


def test_model_output_shape():
    model = make_model()

    X = np.array([
        [1.0, 0.1, 0.2],
        [1.0, 0.2, 0.3],
        [1.0, 0.3, 0.4],
        [1.0, 0.4, 0.5],
        [1.0, 0.5, 0.6]
    ])

    prediction = model.predict(X)

    assert prediction.shape == (5,)
