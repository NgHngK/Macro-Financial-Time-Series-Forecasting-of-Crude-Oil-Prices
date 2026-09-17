# Motor Vehicle Fuel Price Regression with Deep Learning

## About this project

Fuel prices affect transportation, production costs, logistics, and many parts of the wider economy. This project studies whether deep learning models can learn the time patterns behind fuel prices and use market and macroeconomic information to predict future price movements.

The main prediction target is:

```text
cushing_crude_oil_price
```

We focus on three models with clear evaluation results:

- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Attention-based / Transformer model

Our goal is not only to obtain low prediction error, but also to understand how feature selection, lookback length, model architecture, and hyperparameter tuning affect time-series forecasting performance.

[Detailed project documentation](Report.pdf)

## Data

The dataset combines energy, financial-market, exchange-rate, interest-rate, and inflation information from sources such as the **U.S. Energy Information Administration (EIA)** and **Federal Reserve Economic Data (FRED)**.

The `compiled_dataset.csv` file contains **5,609 rows from January 16, 2003 to May 5, 2025**. It includes the crude-oil target together with variables such as the Dow Jones, Nasdaq, S&P 500, exchange rates, Federal Funds Rate, Bank Prime Loan Rate, Treasury yields, breakeven inflation rates, Henry Hub natural-gas price, and several moving-average and momentum features.

These variables were selected because oil prices are influenced by more than their own past values. Equity indices can reflect economic activity and market expectations, interest rates describe monetary and financing conditions, exchange rates affect the purchasing power of international buyers, and inflation indicators provide information about broader price pressure.

## Data preprocessing

The raw time series contains missing values because financial markets, government statistics, and energy-price series are not always updated on the same days. Holidays and different reporting schedules can therefore create small gaps between variables.

We use **linear interpolation** to fill these gaps instead of removing the affected rows. This keeps the timeline continuous and avoids losing a large amount of data. The features are then normalized with **Min-Max Scaling**, which puts variables with different units and ranges onto a comparable scale before model training.

The forecasting problem is converted into sequences using sliding windows. Instead of treating every day independently, the models receive information from previous time steps and learn how those values relate to the next target value.

# Model architecture

## 1. Long Short-Term Memory

The LSTM model is designed to learn longer temporal dependencies. Its hidden dimensions decrease through the network:

```text
128 -> 64 -> 32
```

The model uses three LSTM layers with dropout between them and a final dense regression layer. A 30-day lookback window is used. The final configuration contains **132,513 trainable parameters** and is trained with AdamW, Huber loss, a learning rate of 0.001, batch size 32, dropout 0.2, and early stopping.

The decreasing hidden dimensions act like a funnel. The first layers learn a larger set of temporal patterns, while later layers compress that information before producing the final prediction.

## 2. Gated Recurrent Unit

The GRU provides a simpler recurrent structure than LSTM while still keeping useful information from previous time steps. We first tested different feature groups and then performed a large hyperparameter search over hidden size, number of GRU layers, and lookback length.

The search covered hidden sizes from 8 to 256, one to five GRU layers, and lookback windows from 1 to 30 days, for a total of **490 experiments**.

The final GRU configuration was:

```text
hidden size = 32
GRU layers = 2
lookback = 1 day
learning rate = 0.007586
```

The learning rate was selected with an LR Finder.

## 3. Attention-based model

The attention-based model uses Transformer encoder blocks to learn relationships across the input sequence. The model contains two Transformer encoders followed by an output layer that maps the final sequence representation to one predicted value.

Each attention block uses four heads with a model dimension of 64. The feed-forward layer uses 128 dimensions, and dropout is set to 0.2. The model is trained with Adam and uses callbacks to reduce the learning rate when performance stops improving and to stop training when no further improvement is found.

We also tested smaller and larger Transformer configurations. Smaller models tended to smooth too much information, while larger models could fit the data more closely but required more computation. The two-encoder design provided a practical balance between model capacity and training cost.

# Results

## 1. LSTM performance

| Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Train | 0.0275 | 0.0404 | 0.9867 |
| Test | 0.0541 | 0.0643 | 0.8199 |

The LSTM learned the training sequence very well and still kept useful prediction ability on unseen data. However, there is a clear gap between training and testing performance. Test MAE is almost twice the training MAE, while R² decreases from 0.9867 to 0.8199.

The model follows the main price trend, but short-term peaks and rapid movements are harder to reproduce. This suggests that the LSTM captures the overall temporal structure well, but its ability to generalize to unseen periods is weaker than its training fit.

## 2. GRU feature-selection experiments

Feature selection had a large effect on GRU performance.

| Feature setup | MAE | RMSE | R² |
|---|---:|---:|---:|
| High-correlation features | 0.0321 | 0.0425 | 0.9407 |
| All features with positive correlation | 0.0338 | 0.0432 | 0.9385 |
| Selected macro/market feature group | 0.0337 | **0.0415** | **0.9433** |

Using every positively correlated feature did not improve the model. In fact, the result became slightly worse. This shows that adding more variables does not automatically produce better forecasts.

The selected feature group focused on major equity indices and macroeconomic variables, including the Dow Jones, Nasdaq, S&P 500, USD exchange rate, Treasury rate, breakeven inflation, Bank Prime Loan Rate, and Federal Funds Rate. Together, these features give the model information about economic demand, liquidity, monetary conditions, inflation, and currency movements.

The key point is that same-day correlation is not enough to decide whether a feature is useful for time-series forecasting. A variable can have a weak direct correlation with oil price but still provide useful information when combined with previous time steps.

## 3. GRU hyperparameter tuning

After feature selection, we performed **490 hyperparameter experiments**. The final tuned GRU achieved:

```text
MAE  = 0.018011
RMSE = 0.024176
R²   = 0.981945
```

Compared with the selected feature setup before final tuning, MAE decreased from 0.0337 to 0.018011, which is about a **46.6% reduction**. RMSE decreased from 0.0415 to 0.024176, which is about a **41.7% reduction**.

This is one of the clearest results in the project. The improvement shows that architecture choice alone is not enough. Hidden size, number of layers, lookback length, learning rate, and feature selection all have a strong effect on final performance.

Another interesting result is that the best GRU lookback was only **one day**. This suggests that the most recent oil-price information carries a strong short-term signal, while the macroeconomic variables provide additional context for the prediction.

## 4. Attention-based model

| Split | RMSE |
|---|---:|
| Train | 0.03255 |
| Test | **0.0233** |

The attention-based model follows the main direction of the target series well, especially during moderate price movements. Its main weakness is that some of the largest peaks are smoothed rather than fully reproduced.

The architecture experiments also show a clear trade-off between model size and efficiency. Smaller Transformer configurations lose too much information and underfit, while larger versions can fit the data more closely but require more computation. The two-encoder model keeps enough capacity to learn the main sequence pattern without making the architecture unnecessarily large.

# What the results tell us

The experiments show that **feature design and model tuning are just as important as model architecture**. The GRU provides the clearest example: changing the feature set alone affected performance, and the final tuning stage reduced MAE by about 46.6% compared with the selected pre-tuning setup.

The results also show that simply using more input variables is not always helpful. The GRU experiment with all positively correlated features performed worse than a smaller, more carefully selected macroeconomic feature group. This suggests that useful forecasting information depends on how variables interact through time, not only on their direct correlation with the target.

The LSTM results show strong learning ability but also a clear train-test gap. The attention model follows the overall trend well but smooths some extreme movements. The tuned GRU achieves the strongest complete set of MAE, RMSE, and R² results in the current experiments.

# Project contribution

This project builds a deep learning pipeline for crude-oil price forecasting using both historical energy prices and broader financial and macroeconomic information. Instead of using only past oil prices, we combine market indices, exchange rates, interest rates, inflation indicators, and other time-series variables to give the models more economic context.

A major part of the project is the experimental process. We compare different recurrent and attention-based approaches, test different feature groups, perform a large GRU hyperparameter search, use sequence-based preprocessing, and study how Transformer size affects forecasting behavior.

The project highlights several practical lessons for financial time-series modeling. More features do not always improve performance, the best lookback window can be surprisingly short, strong training performance does not guarantee equally strong test performance, and larger neural networks are only useful when the extra complexity produces enough improvement to justify the computational cost.

# Limitations

The project uses U.S. data because long and detailed fuel-price and macroeconomic time series are more readily available. This means the results should not be directly treated as a model of the Vietnamese fuel market.

The three evaluated architectures also use different experiment settings, so their metrics should be interpreted in the context of each model rather than as a perfectly controlled leaderboard. A stronger future comparison would use one fixed train/validation/test split, the same input feature set, the same evaluation metrics, and repeated runs across multiple random seeds.

Sudden market shocks also remain difficult to predict. Both recurrent and attention-based models can follow the overall price trend while still missing short, extreme movements. Future work could focus more directly on these high-volatility periods.

# Repository structure

```text
.
├── compiled_dataset.csv
├── oil_data.csv
├── gru.py
├── gru.ipynb
├── main.ipynb
├── Report.pdf
├── last_updated.txt
├── README.md
└── model/
    ├── RNN.keras
    ├── LSTM.keras
    ├── GRU.keras
    ├── LSTM_GRU.keras
    └── Transformer.keras
```

`compiled_dataset.csv` contains the combined modeling dataset. `gru.ipynb` and `gru.py` contain the GRU feature-selection and tuning experiments, while `main.ipynb` contains additional data-processing and model experiments. The `model/` folder stores exported model files.

# How to run

The project requires the following main libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
tensorflow
keras
yfinance
fredapi
```

To run the project, open:

```text
main.ipynb
```

and execute the notebook from top to bottom.

# Team

This project was developed by:

- Hoang Trung Dung
- Le Minh Kiet
- Nguyen Tat Hung
- Long Nguyen
- Van Thang

Hanoi University of Science and Technology

# Final note

This project shows that deep learning can learn useful structure from crude-oil and macroeconomic time series, but good forecasting performance depends heavily on feature selection, sequence design, and model tuning. The strongest results come from careful experimentation rather than simply choosing a more complex neural network.
