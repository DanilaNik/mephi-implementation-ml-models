from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
Обучение базовой модели v1 для прогноза дефолта по кредитным картам.

В качестве алгоритма используется LogisticRegression с предварительным масштабированием
признаков.
"""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "UCI_Credit_Card.csv"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET = "default.payment.next.month"
RANDOM_STATE = 17


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Загружено строк: {len(df)}")

    X = df.drop(columns=["ID", TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\nОтчёт по тестовой выборке:")
    print(classification_report(y_test, y_pred, digits=3))
    print(
        "Кратко: F1={:.3f}  Precision={:.3f}  Recall={:.3f}".format(
            f1_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
        )
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "model_v1.pkl"
    joblib.dump(pipe, out_path)
    print(f"\nМодель v1 сохранена: {out_path}")


if __name__ == "__main__":
    main()
