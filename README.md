# Stock Price Prediction Using RNNs

Predicting intraday stock price direction for four major technology companies using Recurrent Neural Networks (SimpleRNN and LSTM), trained on 12 years of historical market data.

---

## Problem Statement

> Given the stock prices of Amazon, Google, IBM, and Microsoft for a set number of days, predict the intraday price change (Open → Close movement) after that window.

Raw closing prices are non-stationary and cause RNNs to learn "repeat yesterday's price." This project instead predicts **Intraday Change** — a stationary, financially meaningful signal — using engineered technical features rather than raw OHLCV columns.

---

## Dataset

| Property | Detail |
|---|---|
| Companies | AMZN, GOOGL, IBM, MSFT |
| Date Range | January 2006 – January 2018 |
| Records | 3,019 per company · 12,077 combined |
| Source | NYSE / NASDAQ historical data |
| Files | `../RNN_Stocks_Data/<TICKER>_stocks_data.csv` |

**Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Name`

---

## Project Structure

```
RNN_Stock_Price_Prediction/
│
├── RNN_Assg_Stock_Price_Prediction_Starter.ipynb   # Main notebook
├── tuning_worker.py                                 # Model factory + parallel grid search worker
├── conclusion.md                                    # Conclusion and insights
├── evalutaion_rubric.xlsx                           # Assignment rubric
└── README.md
```

```
RNN_Stocks_Data/          # (sibling directory)
├── AMZN_stocks_data.csv
├── GOOGL_stocks_data.csv
├── IBM_stocks_data.csv
└── MSFT_stocks_data.csv
```

---

## Approach

### 1. Data Preparation

- **Missing value imputation**: One missing row (IBM, 2017-07-31) was handled with time-series-aware logic — `Open` filled from previous day's `Close`, `Low` filled from `min(Open, Close)` — preserving sequence continuity.
- **Vertical stacking**: All four stock timelines are stacked row-wise. Windows are generated per-stock independently, preventing false cross-company relationships.

### 2. Feature Engineering

Six stationary features were derived from raw OHLCV data. A two-pass correlation analysis reduced these to the final five used for training:

| Feature | Description | Kept |
|---|---|---|
| `Overnight_Returns` | Log return from prev Close to today Open | Yes |
| `Daily_Range` | (High - Low) / Close — intraday volatility | Yes |
| `RSI` | 14-period Relative Strength Index | Yes |
| `Volume_Share` | Stock's volume as fraction of combined daily volume | Yes |
| `MACD_Histogram` | MACD line minus Signal line | Yes |
| `Returns` | Log close-to-close return | Dropped (0.79 corr with target) |
| `Log_Volume` | Log-transformed volume | Dropped (0.90 corr with Volume_Share) |

**Target:** `Intraday_Chg` = (Close - Open) / Open

### 3. Windowing & Scaling

- **Window size: 21 trading days** (1 calendar month) — selected by comparing rolling averages at 10, 21, and 50 days
- **Step size: 1 day** — maximum data utilization
- **Scaling: per-window Z-score** — each window is normalized by its own mean and std, avoiding data leakage from future windows
- **Train/test split: 80/20 chronological**, stratified per stock (no shuffling)

Final tensor shapes:

| Split | Shape |
|---|---|
| X_train | (9552, 21, 5) |
| X_test | (2389, 21, 5) |
| y_train | (9552,) |
| y_test | (2389,) |

### 4. Hyperparameter Tuning

Both models were tuned using a **parallel grid search** with `ProcessPoolExecutor` and **3-fold `TimeSeriesSplit`** cross-validation. Model selection was based on **Directional Hit Rate** (not MSE) — the commercially meaningful metric.

| Model | Permutations | Search Space |
|---|---|---|
| SimpleRNN | 864 | units, dense_units, dropout, activation, optimizer, lr, batch_size |
| LSTM | 288 | + internal_activation (tanh / sigmoid / hard_sigmoid) |

---

## Model Results

| Model | Architecture | Epochs | Optimizer | Test MSE | Test RMSE | Directional Hit Rate |
|---|---|---|---|---|---|---|
| **SimpleRNN** | SimpleRNN(64) → Dense(32, LeakyReLU) → Dropout(0.3) → Dense(1) | 20 | Adam + ReduceLROnPlateau | 1.3058 | 1.1427 | 51.95% |
| **LSTM** | LSTM(64) → Dense(16, tanh) → Dropout(0.3) → Dense(1) | 60 | AdamW + ReduceLROnPlateau | 1.3024 | 1.1412 | **54.04%** |

> A random direction guess = 50.00%. Both models consistently exceed this baseline. In algorithmic trading, a sustained 51–54% directional edge is a viable alpha signal.

---

## Key Findings

- **LSTM outperforms SimpleRNN** due to its gating mechanism, which retains multi-week patterns without suffering from vanishing gradients over the 21-step sequence.
- **Feature engineering was the primary performance driver** — introducing RSI, MACD, and Volume Share raised the hit rate ceiling from 52.26% to 54.04%.
- **Conservative predictions are correct behavior** — the model predicts near-zero values rather than chasing extreme daily spikes. Those spikes are news-driven and unpredictable; chasing them would be overfitting.
- **LeakyReLU eliminated upward prediction bias** observed in earlier trials with standard ReLU activations.
- **ReduceLROnPlateau was essential** — the LSTM required multiple learning rate reductions before stabilizing, confirming a fixed LR would have stalled.

---

## Requirements

```
tensorflow >= 2.18
keras >= 3.14
scikit-learn
pandas
numpy
matplotlib
seaborn
openpyxl
```

Install dependencies:

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn openpyxl
```

> The notebook detects and utilizes Apple Metal GPU (MPS) automatically on Apple Silicon machines.

---

## How to Run

1. Ensure the `RNN_Stocks_Data/` directory is at `../RNN_Stocks_Data/` relative to the notebook.
2. Open `RNN_Assg_Stock_Price_Prediction_Starter.ipynb` in Jupyter.
3. Run all cells top to bottom.

> Hyperparameter grid search cells use `multiprocessing` and will launch parallel worker processes. These are safe to re-run but take several minutes depending on hardware.
