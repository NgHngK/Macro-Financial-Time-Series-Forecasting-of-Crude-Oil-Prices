# Motor Vehicle Fuel Price Regression with Deep Learning

## About this project

Fuel prices affect transportation, production costs, logistics, and many parts of the wider economy. This project studies whether deep learning models can learn the time patterns behind fuel prices and use market and macroeconomic information to predict future price movements.

The project was developed as a Deep Learning course project at **Hanoi University of Science and Technology**. Five different time-series architectures were explored:

- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- LSTM + GRU hybrid
- Attention-based / Transformer model

Although the project discusses fuel and gas prices in a broad sense, the main prediction target used in the experiments is:

```text
cushing_crude_oil_price
```

The main goal is not only to train several neural networks, but also to understand how architecture choice, feature selection, lookback length, and hyperparameter tuning affect time-series forecasting performance.

[Read the full project report](Report.pdf)

## Data

The dataset combines energy, financial-market, exchange-rate, interest-rate, and inflation information. The report mainly uses data collected from the **U.S. Energy Information Administration (EIA)** and **Federal Reserve Economic Data (FRED)** because these sources provide long and consistent historical time series.

The report describes a dataset covering the period from 2003 to 2025 with roughly 5.8 thousand observations. The `compiled_dataset.csv` included in this repository snapshot currently contains **5,609 rows from January 16, 2003 to May 5, 2025**, so the exact row count may differ from the report snapshot as the dataset is designed to be updated over time.

The data contains the crude-oil target together with variables such as the Dow Jones, Nasdaq, S&P 500, exchange rates, Federal Funds Rate, Bank Prime Loan Rate, Treasury yields, breakeven inflation rates, Henry Hub natural-gas price, and several moving-average and momentum features.

These variables were chosen because oil prices are affected by more than their own past values. Equity indices can reflect economic activity and market expectations, interest rates can describe monetary and financing conditions, exchange rates affect the purchasing power of international buyers, and inflation indicators provide information about broader price pressure.

## Data preprocessing

The raw time series contains missing values because financial markets, government statistics, and energy-price series are not always updated on the same days. Holidays and different reporting schedules can therefore create small gaps between variables.

The project uses **linear interpolation** to fill these gaps instead of removing the affected rows. This keeps the timeline continuous and avoids losing large parts of the dataset. The features are then normalized with **Min-Max Scaling**, which puts variables with very different units and ranges onto a comparable scale before model training.

The forecasting problem is converted into sequences using sliding windows. Instead of treating every day independently, the models receive information from previous time steps and learn how those values relate to the next target value.

## Model architecture

### 1. Recurrent Neural Network

The RNN is used as the basic recurrent model. It contains two stacked recurrent layers with a hidden size of 64 and a final linear regression layer. The model uses a 30-day sequence window, Adam with a learning rate of 0.001, MSE loss, a batch size of 32, dropout of 0.2, and 50 training epochs.

This model provides a simple baseline for learning temporal dependencies. It can follow short-term sequences, but standard RNNs have difficulty keeping information over long periods because gradients can vanish or explode during training.

### 2. Long Short-Term Memory

The LSTM model is designed to handle longer dependencies more effectively. Its hidden dimensions gradually decrease from:

```text
128 -> 64 -> 32
```

The model uses three LSTM layers with dropout between them and a final dense regression layer. A 30-day lookback window is used. The final configuration contains **132,513 trainable parameters** and is trained with AdamW, Huber loss, a learning rate of 0.001, batch size 32, dropout 0.2, and early stopping.

The decreasing hidden dimensions act like a funnel: the first layers learn a large set of temporal patterns, while later layers compress them into a smaller representation before producing the final price prediction.

### 3. Gated Recurrent Unit

The GRU offers a simpler alternative to LSTM. It uses reset and update gates instead of the larger LSTM gating structure, which reduces the number of parameters while still allowing the network to keep useful information from previous time steps.

An important part of the GRU experiment is that the team did not use only one fixed configuration. Different feature groups were tested first, followed by a large hyperparameter search. The final search covered hidden sizes from 8 to 256, one to five GRU layers, and lookback windows from 1 to 30 days, for a reported total of **490 experiments**.

The final configuration found by this process used:

```text
hidden size = 32
GRU layers = 2
lookback = 1 day
learning rate = 0.007586
```

The learning rate was selected with an LR Finder.

### 4. LSTM + GRU hybrid

The hybrid architecture is one of the more interesting parts of the project. Instead of simply stacking an LSTM and GRU, the model uses the LSTM as an encoder and transfers its learned memory into a GRU decoder.

The LSTM first learns long-term temporal information. Its final cell state is passed through a fully connected transformation and a `tanh` activation so that it can initialize the hidden state of the GRU. At the same time, the sequence of LSTM hidden states is passed to the GRU as its input sequence.

In simple terms, the model tries to let the LSTM answer:

> What long-term information should be remembered?

and then lets the GRU use that information to refine the final prediction.

The reported hybrid model contains **148,225 parameters**. It uses dropout of 0.1, Adam with a learning rate of 0.001, MAE loss, early stopping, and a batch size of 15. The report experiments with both daily and weekly forecasting, using 5-day and 20-day lookback windows.

### 5. Attention-based model

The final architecture uses the attention mechanism through Transformer encoder blocks. The model contains two Transformer encoders followed by a simple output stage that selects the final time step and maps it to one predicted value.

Each attention block uses four heads with a model dimension of 64. The feed-forward layer uses 128 dimensions, and dropout is set to 0.2. The model is trained with Adam and includes callbacks that reduce the learning rate when performance stops improving and stop training when no further improvement is found.

The appendix also studies larger and smaller Transformer configurations. Increasing the number of layers or dimensions can produce a tighter fit, but it also makes training more expensive. The final two-encoder design was kept as a practical balance between model capacity and computational cost.

# Results

The report does not provide exactly the same set of metrics for every model, so the values below should be read as the **reported results of each experiment**, rather than as a perfectly controlled leaderboard.

| Model | MAE | RMSE | R² | Notes |
|---|---:|---:|---:|---|
| RNN | — | — | — | Quantitative metrics are not stated in the report |
| LSTM - Train | 0.0275 | 0.0404 | 0.9867 | Strong fit on training data |
| LSTM - Test | 0.0541 | 0.0643 | 0.8199 | Lower performance than train, but still follows the main trend |
| GRU - Final tuned model | **0.018011** | **0.024176** | **0.981945** | Result after feature selection and 490 hyperparameter experiments |
| LSTM + GRU | — | — | — | Reported mainly through daily and weekly prediction plots |
| Attention - Test | — | **0.0233** | — | Captures the main trend but smooths some large peaks |
| Attention - Train | — | 0.03255 | — | Reported training RMSE |

## RNN result

The RNN prediction follows much of the long-term movement of the crude-oil series, which shows that even a basic recurrent network can learn useful time structure. However, the plot also shows larger errors around sudden and extreme movements. The model follows normal trends more easily than rare shocks, which is a common limitation of simple recurrent models.

The report does not provide MAE, RMSE, or R² values for the RNN result, so it is better to treat the RNN as a visual baseline rather than make a numerical comparison that the report does not support.

## LSTM result

The LSTM achieved a training MAE of **0.0275**, RMSE of **0.0404**, and R² of **0.9867**. On the test set, MAE increased to **0.0541**, RMSE increased to **0.0643**, and R² decreased to **0.8199**.

This shows that the LSTM learned the training sequence very well and still kept useful prediction ability on unseen data, but there is a clear gap between training and testing performance. The test prediction follows the general price movement, while some short-term peaks and rapid changes are harder to reproduce. In other words, the LSTM captures the overall temporal structure, but its generalization is weaker than its training fit.

## GRU feature-selection experiments

Feature selection had a large effect on the GRU model. The first experiment used only features with high same-day correlation to the target and obtained:

| Feature setup | MAE | RMSE | R² |
|---|---:|---:|---:|
| High-correlation features | 0.0321 | 0.0425 | 0.9407 |
| All features with positive correlation | 0.0338 | 0.0432 | 0.9385 |
| Selected macro/market feature group | 0.0337 | **0.0415** | **0.9433** |

Simply adding more positively correlated variables did not improve the model. This is an important result because same-day correlation does not necessarily tell us which features are useful across time. A variable can have a weak same-day relationship with oil price but still provide useful information about the future when it is combined with previous time steps.

The final feature group focused on major equity indices and macroeconomic variables, including the Dow Jones, Nasdaq, S&P 500, USD exchange rate, Treasury rate, breakeven inflation, Bank Prime Loan Rate, and Federal Funds Rate. The report argues that these variables give the model a broader view of economic demand, liquidity, monetary conditions, inflation, and currency movements.

## GRU hyperparameter tuning

After choosing the feature group, the project performed **490 hyperparameter experiments**. The final tuned GRU reached:

```text
MAE  = 0.018011
RMSE = 0.024176
R²   = 0.981945
```

Compared with the selected feature group before final tuning, MAE fell from 0.0337 to 0.018011, which is about a **46.6% reduction**. RMSE fell from 0.0415 to 0.024176, a reduction of about **41.7%**.

This improvement shows that the strong GRU result did not come from architecture choice alone. Feature selection, hidden size, number of layers, lookback length, and learning rate all had an important effect on the final performance.

Another interesting result is that the best GRU lookback was only **one day**. The report interprets this as evidence that the most recent oil-price information provides a strong short-term signal, while the selected macroeconomic variables give extra context that helps adjust the prediction.

## LSTM + GRU hybrid result

The hybrid model fits the training data closely and follows both gradual trends and many short-term movements. The test result is more mixed. According to the report, predictions track the ground truth well during the earlier part of the test set, but after roughly index 500 the model increasingly underestimates the true values. The difference becomes clearer in the later part of the sequence.

This suggests that the hybrid architecture is capable of learning the temporal structure, but its errors can grow over a long test period. The report points to possible cumulative error or overfitting as areas that need further investigation. The project also experiments with weekly prediction, showing that the same LSTM-to-GRU memory-transfer idea can be applied at a different forecasting frequency.

Because the Results section does not provide one final MAE, RMSE, and R² summary for this model, the README keeps the hybrid result qualitative rather than inventing a direct numerical comparison.

## Attention-based model result

The attention-based model reports an RMSE of **0.0233 on the test set** and **0.03255 on the training set**. Its prediction follows the main direction of the target series well, especially through moderate price movements.

The main weakness visible in the result is that attention tends to smooth some of the highest peaks. In other words, it learns the overall shape of the time series more easily than rare extreme values.

The appendix supports this interpretation. A smaller model with reduced dimensions underfits and smooths too much information, while a much larger model can produce a tighter fit but costs more to train. Increasing the number of encoder layers can also improve fitting, but the authors chose two encoders because the extra computational cost of the deeper versions was not considered worthwhile for this dataset.

# What the results tell us

The experiments show that architecture matters, but **model tuning and feature design matter just as much**. GRU performance improved strongly after the feature group and hyperparameters were changed, while using a larger set of simply correlated variables did not automatically improve forecasting.

The results also show that different architectures fail in different ways. The RNN has difficulty with extreme movements, the LSTM shows a noticeable train-test gap, the hybrid model begins to drift during the later test period, and the attention model tends to smooth large peaks. These differences are useful because they show that a low error score alone does not explain how a model behaves across different market conditions.

The GRU experiment gives the clearest quantitative example of successful tuning, while the hybrid model provides the most custom architectural idea in the project. The attention experiments add another useful perspective by showing the trade-off between model size, fitting ability, and computational cost.

# Project contribution

The main contribution of this project is the comparison of several deep learning approaches on one real-world fuel-price forecasting problem. Instead of using only historical oil prices, the project builds a broader dataset that includes financial markets, monetary variables, exchange rates, and inflation information.

The work also goes beyond simply training five default neural networks. It includes missing-value treatment, normalization, sequence construction, feature-selection experiments, a large GRU hyperparameter search, a custom LSTM-to-GRU state-transfer architecture, Transformer-size experiments, daily forecasting, and weekly forecasting.

From a learning and research perspective, the project demonstrates three main ideas: financial time-series forecasting depends heavily on good preprocessing, more features do not always produce a better model, and more complex architectures are only useful when their extra capacity produces enough improvement to justify their cost.

# Limitations

There are several limitations to keep in mind when reading the results. The project uses U.S. data because equivalent Vietnamese data was not available with enough history and detail. The report also does not present the same metrics and exactly the same experimental setup for all five architectures, so the reported values should not be treated as a strict apples-to-apples ranking without further controlled testing.

Sudden market shocks remain difficult for several models, and the hybrid and attention results show that a model can follow the general trend while still missing extreme values or slowly drifting away from the target. Future work could use one fixed train/validation/test split for every architecture, report the same MAE/RMSE/R² metrics for all models, and repeat experiments across multiple random seeds.

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

`compiled_dataset.csv` contains the combined modeling dataset. `gru.ipynb` and `gru.py` contain the GRU experiments, while `main.ipynb` contains additional data-processing and model experiments. The `model/` folder stores exported model files.

# How to run

This repository does not currently include a pinned `requirements.txt`. The main libraries used by the code are:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch tensorflow keras yfinance fredapi
```

Then open the notebooks and run the required experiment:

```text
gru.ipynb   -> GRU experiments and feature selection
main.ipynb  -> additional data collection / model experiments
```

If the data-collection cells are used, configure your own API credentials before running them.

# Team

This project was developed by:

- Hoang Trung Dung
- Le Minh Kiet
- Nguyen Tat Hung
- Long Nguyen
- Van Thang

Hanoi University of Science and Technology

## Final note

This project is a forecasting experiment and a Deep Learning course project. It shows that recurrent and attention-based models can learn useful structure from crude-oil and macroeconomic time series, but the reported results should be interpreted together with the model-specific evaluation setup and the limitations above.
