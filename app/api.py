from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request

FEATURE_COLS: list[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

DEFAULT_MODEL = "v1"
AB_SPLIT_THRESHOLD = 50  # процентов трафика, которые получает v1


# Логирование в JSON
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload, ensure_ascii=False)


def _build_logger() -> logging.Logger:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("credit_default_svc")
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    log.handlers.clear()
    log.addHandler(handler)
    log.propagate = False
    return log


logger = _build_logger()


def log_event(level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


# Реестр моделей
class ModelRegistry:
    """Загружает и хранит обученные pipeline'ы по ключу версии."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._models: dict[str, Any] = {}

    def load_all(self) -> None:
        for filename in sorted(self._base_dir.glob("model_v*.pkl")):
            version = filename.stem.split("_")[-1]  # 'v1', 'v2', ...
            self._models[version] = joblib.load(filename)
            log_event(logging.INFO, "model_loaded", version=version, path=str(filename))
        if not self._models:
            raise RuntimeError(
                f"В каталоге {self._base_dir} не найдено ни одного файла model_v*.pkl"
            )

    @property
    def versions(self) -> list[str]:
        return sorted(self._models.keys())

    def get(self, version: str):
        if version not in self._models:
            raise KeyError(version)
        return self._models[version]


def _resolve_models_dir() -> Path:
    """Поддерживаем запуск как локально (./models), так и в контейнере (/app/models)."""
    if Path("/app/models").is_dir():
        return Path("/app/models")
    return Path(__file__).resolve().parent.parent / "models"


registry = ModelRegistry(_resolve_models_dir())


# A/B-распределение трафика
def assign_ab_bucket(user_id: str) -> str:
    """Стабильный сплит: один user_id всегда попадает в одну и ту же группу."""
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "v1" if bucket < AB_SPLIT_THRESHOLD else "v2"


# Flask-приложение
app = Flask(__name__)


@app.before_request
def _attach_request_context() -> None:
    g.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    g.started_at = time.perf_counter()


@app.after_request
def _log_response(response: Response) -> Response:
    elapsed_ms = round((time.perf_counter() - g.get("started_at", time.perf_counter())) * 1000, 2)
    log_event(
        logging.INFO,
        "http_request",
        request_id=g.get("request_id"),
        method=request.method,
        path=request.path,
        status=response.status_code,
        latency_ms=elapsed_ms,
    )
    response.headers["X-Request-Id"] = g.get("request_id", "")
    return response


def _validate_payload(payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    missing = [col for col in FEATURE_COLS if col not in payload]
    if missing:
        return pd.DataFrame(), missing
    row = {col: payload[col] for col in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS), []


def _score(version: str, frame: pd.DataFrame) -> dict[str, Any]:
    estimator = registry.get(version)
    proba = float(estimator.predict_proba(frame)[0][1])
    label = int(estimator.predict(frame)[0])
    return {
        "default_predicted": label,
        "risk_score": round(proba, 4),
        "served_by": version,
    }


@app.errorhandler(404)
def _not_found(_):
    return jsonify({"error": "endpoint_not_found"}), 404


@app.errorhandler(405)
def _method_not_allowed(_):
    return jsonify({"error": "method_not_allowed"}), 405


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "credit-default-svc",
            "endpoints": ["/health", "/models", "/predict", "/predict/ab"],
            "available_versions": registry.versions,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "loaded_models": registry.versions,
        }
    )


@app.route("/models", methods=["GET"])
def models():
    return jsonify({"available_versions": registry.versions, "default": DEFAULT_MODEL})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "expected_json_object"}), 400

    frame, missing = _validate_payload(body)
    if missing:
        return (
            jsonify(
                {
                    "error": "missing_features",
                    "missing": missing,
                    "required": FEATURE_COLS,
                }
            ),
            400,
        )

    requested_version = body.get("model_version")
    user_id = body.get("user_id")
    routing: str
    if requested_version is not None:
        if requested_version not in registry.versions:
            return (
                jsonify(
                    {
                        "error": "unknown_model_version",
                        "requested": requested_version,
                        "available": registry.versions,
                    }
                ),
                400,
            )
        version = requested_version
        routing = "explicit"
    elif user_id:
        version = assign_ab_bucket(str(user_id))
        routing = "ab_split"
    else:
        version = DEFAULT_MODEL
        routing = "default"

    try:
        result = _score(version, frame)
    except Exception as exc:  # noqa: BLE001
        log_event(logging.ERROR, "predict_failed", error=str(exc), version=version)
        return jsonify({"error": "internal_inference_error"}), 500

    result["routing"] = routing
    log_event(
        logging.INFO,
        "prediction",
        request_id=g.get("request_id"),
        version=version,
        routing=routing,
        risk_score=result["risk_score"],
        default_predicted=result["default_predicted"],
    )
    return jsonify(result)


@app.route("/predict/ab", methods=["POST"])
def predict_ab():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "expected_json_object"}), 400

    user_id = body.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id_required_for_ab"}), 400

    frame, missing = _validate_payload(body)
    if missing:
        return jsonify({"error": "missing_features", "missing": missing}), 400

    version = assign_ab_bucket(str(user_id))
    try:
        result = _score(version, frame)
    except Exception as exc:  # noqa: BLE001
        log_event(logging.ERROR, "predict_failed", error=str(exc), version=version)
        return jsonify({"error": "internal_inference_error"}), 500

    result["routing"] = "ab_split"
    result["ab_group"] = "control" if version == "v1" else "treatment"
    log_event(
        logging.INFO,
        "prediction",
        request_id=g.get("request_id"),
        version=version,
        routing="ab_split",
        ab_group=result["ab_group"],
        user_id=str(user_id),
    )
    return jsonify(result)


registry.load_all()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log_event(logging.INFO, "service_starting", port=port, models=registry.versions)
    app.run(host="0.0.0.0", port=port, debug=False)
