# Прогнозирование дефолта по кредитным картам

**Студент** Никулин Данила, M255616


## Постановка задачи

Банк хочет на этапе принятия решения по карте оценивать вероятность того, что клиент
не закроет следующий платёж. На входе - анкета и история по шести предыдущим месяцам,
на выходе — бинарный ответ и risk-score `[0; 1]`. 


## Запуск

### Вариант A. Готовый образ из Docker Hub

Образ доступен: https://hub.docker.com/r/danilanik22/credit-default-svc

```bash
docker pull danilanik22/credit-default-svc:latest
docker run -d --name credit-default -p 5000:5000 danilanik22/credit-default-svc:latest
curl http://localhost:5000/health
```

### Вариант Б. Локальный venv

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt

# (необязательно так как модели уже лежат в models/)
python models/train_model.py
python models/train_model_v2.py

python app/api.py
```

### Вариант В. docker compose

```bash
docker compose up -d --build               # api + log-viewer
docker compose --profile broker up -d      # дополнительно RabbitMQ как и требовалось в задании предусмотреть
docker compose down -v
```

### Прогон тестов

```bash
pip install -r requirements.txt
pytest -q       # поднимать сервис не требуется
```

## API

Сервис слушает порт `5000`. Все ответы — `application/json`. На каждый запрос
добавляется заголовок `X-Request-Id` (проксируется из запроса либо генерируется).

### `GET /` — общее описание сервиса

```bash
curl -s http://localhost:5000/ | jq
```

```json
{
  "service": "credit-default-svc",
  "endpoints": ["/health", "/models", "/predict", "/predict/ab"],
  "available_versions": ["v1", "v2"]
}
```

### `GET /health`

```bash
curl -s http://localhost:5000/health
```

```json
{ "status": "ok", "loaded_models": ["v1", "v2"] }
```

### `GET /models`

```json
{ "available_versions": ["v1", "v2"], "default": "v1" }
```

### `POST /predict`

Параметры запроса:
`model_version`: тип string, нужно для того чтобы принудительно выбрать версию (`"v1"` или `"v2"`)
`user_id`: тип string, если `model_version` не задан, то будет применён A/B-сплит по `user_id`


```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 5000, "BILL_AMT2": 4500, "BILL_AMT3": 4000,
    "BILL_AMT4": 3500, "BILL_AMT5": 3000, "BILL_AMT6": 2500,
    "PAY_AMT1": 1000, "PAY_AMT2": 900, "PAY_AMT3": 800,
    "PAY_AMT4": 700, "PAY_AMT5": 600, "PAY_AMT6": 500,
    "model_version": "v2"
  }'
```

```json
{
  "default_predicted": 0,
  "risk_score": 0.1973,
  "served_by": "v2",
  "routing": "explicit"
}
```

Поле `routing` показывает, как была выбрана версия:

- `explicit` — явно указали `model_version`;
- `ab_split` — выбрана по `user_id`;
- `default` — версия по умолчанию.

### `POST /predict/ab`

Принудительный A/B-сплит. `user_id` обязателен.

```bash
curl -s -X POST http://localhost:5000/predict/ab \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "client-12345",
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 5000, "BILL_AMT2": 4500, "BILL_AMT3": 4000,
    "BILL_AMT4": 3500, "BILL_AMT5": 3000, "BILL_AMT6": 2500,
    "PAY_AMT1": 1000, "PAY_AMT2": 900, "PAY_AMT3": 800,
    "PAY_AMT4": 700, "PAY_AMT5": 600, "PAY_AMT6": 500
  }'
```

```json
{
  "default_predicted": 0,
  "risk_score": 0.1973,
  "served_by": "v2",
  "routing": "ab_split",
  "ab_group": "treatment"
}
```


## Модели и метрики

Признаки одинаковые для обеих версий (23 поля: `LIMIT_BAL`, `SEX`, `EDUCATION`,
`MARRIAGE`, `AGE`, `PAY_0`, `PAY_2..PAY_6`, `BILL_AMT1..6`, `PAY_AMT1..6`).

### Модель v1
`StandardScaler` -> `LogisticRegression(class_weight=balanced, C=0.5, solver=liblinear)`

### Модель v2
`GradientBoostingClassifier(n_estimators=200, lr=0.05, max_depth=3, subsample=0.8)`


| Метрика                  |  v1 (LogReg) | v2 (GBM) |
|--------------------------|-------------:|---------:|
| Accuracy                 |        0.702 |    0.816 |
| Precision (класс default)|        0.393 |    0.654 |
| Recall (класс default)   |        0.638 |    0.353 |
| F1 (класс default)       |        0.486 |    0.459 |

Видна классическая дилемма: v1 ловит больше дефолтников за счёт большего числа
ложных срабатываний, v2 точнее на положительных, но пропускает больше реальных
дефолтов. Именно поэтому имеет смысл прогнать настоящий A/B-тест, а не выбирать
победителя по одной метрике.


## Архитектура

### Монолит vs микросервисы

В проекте выбран **монолит**, и для текущих условий это правильно:

- весь сервис — это один синхронный обработчик `predict`, разделять его на
  отдельные процессы нечего: разделение по слоям (валидация -> модель -> ответ)
  логически уже есть внутри одного контейнера;
- одну модель проще откатить и тестировать (один Dockerfile, один CI-пайплайн);

Переходить на микросервисы стоит, когда появятся:
ETL-пайплайн с фичами в реальном времени, отдельный сервис правил, сервис
логирования предсказаний или модели разной природы
(онлайн скоринг + офлайн ранжирование).

### Брокер сообщений

Брокер уместен в трёх сценариях для этого кейса:

1. **Батч-скоринг.** Раз в час банк хочет переоценить весь портфель.
   Заявки складываются в очередь `scoring.batch`, пул воркеров читает её и пишет
   результаты в БД, не мешая онлайн-эндпоинту.
2. **Логирование предсказаний.** Каждый ответ публикуется в очередь
   `predictions.audit`, а отдельный consumer кладёт их в Elasticsearch / Kafka
   для аналитики.

В `docker-compose.yml` под профилем `broker` уже доступен RabbitMQ — UI на
`http://localhost:15672` (guest/guest), AMQP-порт 5672.

### Логирование и мониторинг

Сервис пишет JSON-логи в stdout — каждая строка одно событие:

```json
{"ts":"2026-05-03T20:14:11Z","level":"INFO","logger":"credit_default_svc","msg":"prediction","request_id":"a1b2","version":"v2","routing":"ab_split","risk_score":0.1437,"default_predicted":0}
```

В production такие логи собирает sidecar (Filebeat / Fluent Bit) и отдаёт например в Loki. По `request_id` можно склеить лог входа и ответ
сервиса. На метриках для примера Grafana полезно было бы строить:

- долю запросов с `error`;
- распределение `risk_score`.


## Бизнес-метрики

1. **Ожидаемые потери от дефолтов.**

   Метрика показывает, сколько теоретических потерь модель помогла предотвратить, отказав рискованным клиентам.

2. **Доля одобренных при том же уровне риска.**

   Фиксируем целевой Bad Rate в портфеле (например, 5%) и считаем, какая
   доля заявок проходит при таком пороге. При повышении Approval rate без
   роста Bad Rate банк зарабатывает больше процентов.


## A/B-тестирование

### Постановка

 Контроль (control): модель v1 — `LogisticRegression`
 Тест (treatment): модель v2 — `GradientBoostingClassifier`   

Сплит трафика 50/50 по `user_id` реализован в `app/api.py`:


(Повторный запрос того же `user_id` всегда
попадает в одну и ту же группу.)

### Метрики

- **Основная — F1-score на классе default = 1.** Финансовая боль возникает
  и при FN (пропустили дефолтника, выдали кредит, потеряли тело и %), и при
  FP (отказали хорошему клиенту, упустили процентный доход и репутацию).
  F1 балансирует обе ошибки в одном числе.

- **Дополнительная — Precision на классе default = 1.** Бизнес особенно
  чувствителен к ложным отказам: при равном F1 модель с большим Precision
  меньше бьёт по добросовестным клиентам, что важно для NPS и регуляторных
  жалоб.


### Статистический анализ

- **Тест:** двухвыборочный z-тест для долей (F1 пересчитывается из
  TP/FP/FN, что для двух больших выборок даёт нормальное приближение).

  ```text
  z = (p_treatment − p_control)
      / sqrt( p_pool · (1 − p_pool) · (1/n_t + 1/n_c) )
  ```

- **Доверительный интервал 95% для разности F1:**
  `(p_treatment − p_control) ± 1.96 x SE)`.

### Критерий принятия решения

| Условие                                     | Действие                                       |
|---------------------------------------------|------------------------------------------------|
| `p < 0.05` и `прирост F1 > 0.02`                  | раскатываем v2 на 100% трафика                 |
| `p < 0.05` и `снижение F1 < 0.01`                  | останавливаем тест досрочно, оставляем v1      |
| `p >= 0.05`                                  | продолжаем сбор данных или закрываем как ничью |
| Precision вырос >5% при сохранении F1        | дополнительный аргумент в пользу v2            |

### Связь с архитектурой

A/B-сплит реализован прямо в API: один контейнер обслуживает обе модели,
направление трафика — это вычисление по `user_id`. Альтернативой был бы
отдельный router service + два инстанса модели, но это даёт лишний сетевой
hop без выигрыша при текущей нагрузке.

## ONNX и production-обвязка

### Перевод модели в ONNX

`scikit-learn` штатно конвертируется в ONNX через `skl2onnx`:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

pipe = joblib.load("models/model_v1.pkl")
onnx_model = convert_sklearn(pipe, initial_types=[("X", FloatTensorType([None, 23]))])
with open("models/model_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### Зачем uWSGI + NGINX

- **uWSGI / gunicorn** держит пул воркеров, перезапускает упавших, ограничивает
  RSS, умеет gracefully degrade.
- **NGINX** обратный прокси, раздает статику, балансирует нагрузку, защищает от DDoS