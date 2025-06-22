from atris import ensemble, evalModel
from sklearn.datasets import load_iris, fetch_california_housing, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Try to import LightGBM for demonstration
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    has_lgbm = True
except ImportError:
    has_lgbm = False

# Try to import anomaly wrappers
try:
    from atris.anomaly_models import ZScoreAnomaly, IQRAnomaly
    has_anomaly = True
except ImportError:
    has_anomaly = False

# Classification test with model classes and param dicts
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

models = [
    (RandomForestClassifier, {"n_estimators": 10, "random_state": 1}),
]
if has_lgbm:
    models.append((LGBMClassifier, {"n_estimators": 20, "random_state": 42, "verbose": -1}))

model = ensemble.call(*models)
model.fit(X_train, y_train)
preds = model.predict(X_test)
score = evalModel(model, X_test, y_test, task='classification')
print('Classification score (with param dicts):', score)

# Classification test with model classes (default params)
default_model = ensemble.call(RandomForestClassifier, RandomForestClassifier)
default_model.fit(X_train, y_train)
default_score = evalModel(default_model, X_test, y_test, task='classification')
print('Classification score (default params):', default_score)

# Multiclass classification test
X_mc, y_mc = make_classification(n_samples=300, n_features=8, n_classes=3, n_informative=5, random_state=42)
X_mc_train, X_mc_test, y_mc_train, y_mc_test = train_test_split(X_mc, y_mc, random_state=42)
mc_model = ensemble.call(RandomForestClassifier, RandomForestClassifier)
mc_model.fit(X_mc_train, y_mc_train)
mc_score = evalModel(mc_model, X_mc_test, y_mc_test, task='classification')
print('Multiclass classification score:', mc_score)

# Regression test with model classes and param dicts
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=42)
reg_model = ensemble.call(
    (RandomForestRegressor, {"n_estimators": 10, "random_state": 1}),
    (RandomForestRegressor, {"n_estimators": 10, "random_state": 2})
)
reg_model.fit(X_train, y_train)
reg_score = evalModel(reg_model, X_test, y_test, task='regression')
print('Regression RMSE:', reg_score)

# Regression with LightGBM
if has_lgbm:
    lgbm_reg_model = ensemble.call(
        (LGBMRegressor, {"n_estimators": 20, "random_state": 42}),
        (RandomForestRegressor, {"n_estimators": 10, "random_state": 1})
    )
    lgbm_reg_model.fit(X_train, y_train)
    lgbm_reg_score = evalModel(lgbm_reg_model, X_test, y_test, task='regression')
    print('Regression RMSE (with LightGBM):', lgbm_reg_score)

# Anomaly detection test
if has_anomaly:
    X_anom = np.random.randn(200, 5)
    # Inject some outliers
    X_anom[::20] += 10
    y_anom = np.zeros(200)
    y_anom[::20] = 1
    anomaly_model = ensemble.call(ZScoreAnomaly, IQRAnomaly)
    anomaly_model.fit(X_anom)
    anomaly_preds = anomaly_model.predict(X_anom)
    anomaly_score = evalModel(anomaly_model, X_anom, y_anom, task='anomaly')
    print('Anomaly detection score:', anomaly_score)

# Error handling: too few models
try:
    bad_model = ensemble.call(RandomForestClassifier)
except ValueError as e:
    print('Error (too few models):', e)

# Error handling: too many models
try:
    too_many = [RandomForestClassifier]*11
    bad_model = ensemble.call(*too_many)
except ValueError as e:
    print('Error (too many models):', e) 