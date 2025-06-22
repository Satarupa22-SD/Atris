# atris

A Python library for seamless integration and ensembling of multiple machine learning models for classification, regression, and anomaly detection tasks.

## Why Atris?

Machine learning practitioners often need to combine multiple models (from scikit-learn, XGBoost, LightGBM, CatBoost, and more) for better accuracy, robustness, and reliability. However, managing different APIs, preprocessing, and evaluation strategies can be tedious and error-prone.

**Atris** was built to:
- Provide a unified, simple interface for ensembling any combination of models.
- Make it easy to preprocess data consistently and use per-model pipelines.
- Offer built-in tools for evaluation, feature engineering, and monitoring ensemble performance.
- Save you time and reduce boilerplate, so you can focus on building better models.

## Features
- **Flexible Ensembling:** Combine 2 to 10 models of any type (scikit-learn, XGBoost, LightGBM, CatBoost, statistical wrappers, etc.)
- **Unified API:** Use the same interface for classification, regression, and anomaly detection.
- **Built-in Evaluation:** Easily evaluate ensemble performance with a single function.
- **Statistical Anomaly Detection:** Includes Z-Score, IQR, and more as scikit-learn compatible wrappers.
- **Extensible:** Add your own models or wrappers easily.

## Installation

```sh
pip install atris
```

This will automatically install all required dependencies (scikit-learn, xgboost, lightgbm, catboost, pyod, etc.).

## Usage & Examples
See [docs/examples.md](../docs/examples.md) for detailed usage examples and best practices.

## License
MIT License. See LICENSE for details. 