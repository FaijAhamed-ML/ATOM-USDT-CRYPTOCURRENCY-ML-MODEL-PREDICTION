# ATOM-USDT Cryptocurrency ML Model Prediction

Hi, I'm **FAIJAHAMED**, an AI/ML student. This is my individual project where i try to predict the next day movment of the ATOM/USDT crypto coin using supervised machine learning.

> NOTE: this project is for learning only. its NOT a financial advice, please dont trade real money with it.

## What is this project about?

I have daily data of ATOM (about 2,436 rows, from Jan 2020 to Sep 2026) with price, volume and lots of indicators (RSI, MACD, bollinger bands, moving averages etc).

I tried to answer two questions:

1. **Regression** - how much will the price change tomorrow? (`target_next_return`)
2. **Classification** - will tomorrow go UP or DOWN? (`target_direction`)

## Folder structure

```
ATOM-USDT-CRYPTOCURRENCY-ML-MODEL-PREDICTION
|-- 01.data
|   |-- external
|   |-- raw                    <- ATOM_USD_ML_features.csv
|-- 02.notebook
|   |-- 01.data_PreProcessing.ipynb
|   |-- 02.feature_Engineering.ipynb
|   |-- 03.model_Selection     <- one notebook for each algorithm (8 total)
|   |-- 03.model_selection.ipynb
|   |-- 04.hyper_parameter_tuning.ipynb
|   |-- 05.Evaluation.ipynb
|-- 03.results
|   |-- eda_visualizations
|   |-- logs
|   |-- outputs                <- engineered csv, X_train, X_test, y_train, y_test ...
|-- 04.model_Creation          <- final saved models
|-- 05.interface               <- streamlit app
|-- complete_pipeline.ipynb    <- everything in one notebook
```

## Steps i followed

1. **Preprocessing** - checked data types, duplicates, missing values (they are only at the start becuase of rolling windows like sma_200), dropped the constant `symbol` column.
2. **Feature engineering** - raw prices are not good for models because price was $40 in 2021 and about $4 now, so i made ratios (price vs moving averages, candle shapes, volume ratios, lag returns, day/month sin-cos).
3. **Train/test split** - split by TIME (80% old data for train, 20% newest for test). Not random split because that leaks the future.
4. **Model selection** - tried 8 algorithms:
   Decision Tree, Random Forest, XGBoost, Ridge and Lasso, Polynomial Regression, Logistic Regression, KNN, SVM
5. **Hyper parameter tuning** - RandomizedSearchCV with TimeSeriesSplit.
6. **Evaluation** - test set used only once at the end.
7. **Interface** - small streamlit app.

## Data info

| | rows | dates |
|---|---|---|
| train | 1786 | 2020-08-15 to 2025-07-05 |
| test | 447 | 2025-07-06 to 2026-09-25 |

39 features after feature engineering.

## Results

Honestly the results are not good, and thats ok. Daily crypto price is mostly noise.

- Classification (up/down) test accuracy was **49.7%**, but just guessing the majority class gives **53%**. ROC-AUC = 0.48 (0.5 is random).
- Regression RMSE = 0.0368 and the baseline "predict 0 return" is also 0.0368. R2 is almost zero.
- Binomial test p-value = 0.93, so no proof that the model is better than guessing.

So my main learning: **always compare with a baseline** and always split time series by time.

## How to run

```bash
pip install -r requirements.txt
```

Then run the notebooks in order (01 -> 02 -> 03 models -> 03.model_selection -> 04 -> 05 -> model creation).

To start the app:

```bash
streamlit run 05.interface/app.py
```

## Things i want to improve later

- add more features (sentiment, on-chain data, bitcoin price)
- try LSTM / time series models
- try other coins too
- make the interface look better

## Tools used

Python, pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, Streamlit, Jupyter

## Author

**FAIJAHAMED** - AI/ML student
