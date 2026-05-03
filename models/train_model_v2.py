from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

"""
Обучение модели v2

Используется GradientBoostingClassifier без предварительного масштабирования.
"""



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "UCI_Credit_Card.csv"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET = "default.payment.next.month"
RANDOM_STATE = 17


def build_estimator() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Загружено строк: {len(df)}")

    X = df.drop(columns=["ID", TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = build_estimator()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
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
    out_path = MODELS_DIR / "model_v2.pkl"
    joblib.dump(clf, out_path)
    print(f"\nМодель v2 сохранена: {out_path}")


if __name__ == "__main__":
    main()
