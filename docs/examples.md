# Ensemble Use Cases

## 1. Classification Ensemble
**When to use:**  You want to predict a category or class label (e.g., spam vs. not spam, disease vs. healthy, type of flower).
**Why ensemble?**  Combining different classifiers (e.g., Random Forest, Logistic Regression) can improve accuracy and robustness compared to a single model.

```python
from atris import ensemble, models
model = ensemble.call(models.RandomForestClassifier, models.LogisticRegression)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## 2. Regression Ensemble
**When to use:**  You want to predict a continuous value (e.g., house price, temperature, sales).
**Why ensemble?**  Combining regressors (e.g., Random Forest, Linear Regression) can reduce error and improve generalization.

```python
from atris import ensemble, models
model = ensemble.call(models.RandomForestRegressor, models.LinearRegression)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## 3. Anomaly Detection Ensemble
**When to use:**  You want to detect outliers or anomalies (e.g., fraud detection, fault detection in machines, rare event detection).
**Why ensemble?**  Different anomaly detectors may catch different types of outliers. Combining them increases reliability.

```python
from atris import ensemble, models
from atris.anomaly_models import ZScoreAnomaly, IQRAnomaly
model = ensemble.call(ZScoreAnomaly, IQRAnomaly, models.IsolationForest)
model.fit(X_train)
preds = model.predict(X_test)
```

## 4. Per-Model Pipelines
**When to use:**  You want each model in your ensemble to have its own preprocessing or feature engineering steps (e.g., scaling, PCA, polynomial features).
**Why?**  Some models work best with certain preprocessing. This lets you optimize each model's input pipeline.

```python
from atris import ensemble, models
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe1 = make_pipeline(StandardScaler())
model = ensemble.call((pipe1, models.RandomForestClassifier), models.LogisticRegression)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## 5. Monitoring Ensemble Performance
**When to use:**  You want to know when adding more models to your ensemble stops improving validation accuracy (to avoid overfitting and wasted computation).
**Why?**  This helps you build the smallest, most effective ensemble.

```python
from atris import monitor_performance, evalModel, models
model_specs = [models.RandomForestClassifier, models.LogisticRegression]
best_ensemble, scores = monitor_performance(
    model_specs, X_train, y_train, X_val, y_val, eval_fn=evalModel
)
``` 