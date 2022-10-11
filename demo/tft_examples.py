#!/usr/bin/env python
# coding: utf-8

# # Temporal Fusion Transformer
# In this notebook, we show two examples of how two use Darts' `TFTModel`.
# If you are new to darts, we recommend you first follow the `darts-intro.ipynb` notebook.

# In[1]:


# fix python path if working locally
from examples.utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


# ## Temporal Fusion Transformer (TFT)
# 
# Darts' `TFTModel` incorporates the following main components from the original Temporal Fusion Transformer (TFT) architecture as outlined in [this paper](https://arxiv.org/pdf/1912.09363.pdf):
# 
# - gating mechanisms: skip over unused components of the model architecture
# - variable selection networks: select relevant input variables at each time step.
# - temporal processing of past and future input with LSTMs (long short-term memory)
# - multi-head attention: captures long-term temporal dependencies
# - prediction intervals: per default, produces quantile forecasts instead of deterministic values
# 
# ### Training
# `TFTModel` can be trained with past and future covariates. It is trained sequentially on fixed-size chunks consisting of an encoder and a decoder part:
# 
# - encoder: past input with `input_chunk_length`
#   - **past target**: mandatory
#   - **past covariates**: optional
# - decoder: future known input with `output_chunk_length`
#   - **future covariates**: mandatory (if none are available, consider `TFTModel`'s optional arguments `add_encoders` or `add_relative_index` from [here](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html?highlight=tftmodel#temporal-fusion-transformer-tft))
# 
# In each iteration, the model produces a quantile prediction of shape `(output_chunk_length, n_quantiles)` on the decoder part.
# 
# ### Forecast
# #### Probabilistic Forecast
# Per default, `TFTModel` produces probabilistic quantile forecasts using `QuantileRegression`.
# This gives the range of likely target values at each prediction step.
# Most deep learning models in Darts'  - including `TFTModel` -  support `QuantileRegression` and 16 other likelihoods to produce probabilistic forecasts by setting `likelihood=MyLikelihood()` at model creation.
# 
# To produce meaningful results, set `num_samples >> 1` when predicting. For example:
# 
# ```
# model.predict(*args, **kwargs, num_samples=200)
# ```
# 
# Predictions with forecasting horizon `n` are generated auto-regressively using encoder-decoder chunks of the same size as during training.
# 
# If `n > output_chunk_length`, you have to supply additional future values for the covariates you passed to `model.train()`.
# 
# #### Deterministic Forecast
# To produce deterministic rather than probabilistic forecasts, set parameter `likelihood` to `None` and `loss_fn` to a PyTorch loss function at model creation. For example:
# 
# ```
# model = TFTModel(*args, **kwargs, likelihood=None, loss_fn=torch.nn.MSELoss())
# ...
# model.predict(*args, **kwargs, num_samples=1)
# ```

# In[ ]:


# before starting, we define some constants
num_samples = 200

figsize = (9, 6)
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"


# ## Air Passenger Example
# This data set that is highly dependent on covariates. Knowing the month tells us a lot about the seasonal component, whereas the year determines the effect of the trend component.
# 
# Additionally, let's convert the time index to integer values and use them as covariates as well. 
# 
# All of the three covariates are known in the future, and can be used as `future_covariates` with the `TFTModel`.

# In[ ]:


# Read data
series = AirPassengersDataset().load()

# we convert monthly number of passengers to average daily number of passengers per month
series = series / TimeSeries.from_series(series.time_index.days_in_month)
series = series.astype(np.float32)

# Create training and validation sets:
training_cutoff = pd.Timestamp("19571201")
train, val = series.split_after(training_cutoff)

# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

# create year, month and integer index covariate series
covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
covariates = covariates.stack(
    datetime_attribute_timeseries(series, attribute="month", one_hot=False)
)
covariates = covariates.stack(
    TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.arange(len(series)),
        columns=["linear_increase"],
    )
)
covariates = covariates.astype(np.float32)

# transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
scaler_covs = Scaler()
cov_train, cov_val = covariates.split_after(training_cutoff)
scaler_covs.fit(cov_train)
covariates_transformed = scaler_covs.transform(covariates)


# ### Create a model
# If you want to produce deterministic forecasts rather than quantile forecasts, you can use a PyTorch loss function (i.e., set `loss_fn=torch.nn.MSELoss()` and `likelihood=None`).
# 
# The `TFTModel` can only be used if some future input is given. Optional parameters `add_encoders` and `add_relative_index` can be useful, especially if we don't have
# any future input available.<br>
# They generate endoded temporal data is used as future covariates.
# 
# Since we already have future covariates defined in our example they are commented out.

# In[ ]:


# default quantiles for QuantileRegression
quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]
input_chunk_length = 24
forecast_horizon = 12
my_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=300,
    add_relative_index=False,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
)


# ### Train the TFT
# 
# In what follows, we can just provide the whole `covariates` series as `future_covariates` argument to the model; the model will slice these covariates and use only what it needs in order to train on forecasting the target `train_transformed`:

# In[ ]:


my_model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True,epochs=-1)


# ### Look at predictions on the validation set
# We perform a one-shot prediction of 24 months using the "current" model - i.e., the model at the end of the training procedure:

# In[ ]:


def eval_model(model, n, actual_series, val_series):
    pred_series = model.predict(n=n, num_samples=num_samples)

    # plot actual series
    plt.figure(figsize=figsize)
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()


eval_model(my_model, 24, series_transformed, val_transformed)


# ### Backtesting
# Let's backtest our `TFTModel` model, to see how it performs with a forecast horizon of 12 months over the last 3 years:

# In[ ]:


backtest_series = my_model.historical_forecasts(
    series_transformed,
    future_covariates=covariates_transformed,
    start=train.end_time() + train.freq,
    num_samples=num_samples,
    forecast_horizon=forecast_horizon,
    stride=forecast_horizon,
    last_points_only=False,
    retrain=False,
    verbose=True,
)


# In[ ]:


def eval_backtest(backtest_series, actual_series, horizon, start, transformer):
    plt.figure(figsize=figsize)
    actual_series.plot(label="actual")
    backtest_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    backtest_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    plt.legend()
    plt.title(f"Backtest, starting {start}, {horizon}-months horizon")
    print(
        "MAPE: {:.2f}%".format(
            mape(
                transformer.inverse_transform(actual_series),
                transformer.inverse_transform(backtest_series),
            )
        )
    )


eval_backtest(
    backtest_series=concatenate(backtest_series),
    actual_series=series_transformed,
    horizon=forecast_horizon,
    start=training_cutoff,
    transformer=transformer,
)


# ## Monthly ice cream sales
# Let's try a another data set. That of monthly ice cream and heater sales since 2004. Our target is to predict future ice cream sales.
# First, we build the time series from the data, and check its periodicity.

# In[ ]:


series_ice_heater = IceCreamHeaterDataset().load()

plt.figure(figsize=figsize)
series_ice_heater.plot()

print(check_seasonality(series_ice_heater["ice cream"], max_lag=36))
print(check_seasonality(series_ice_heater["heater"], max_lag=36))

plt.figure(figsize=figsize)
plot_acf(series_ice_heater["ice cream"], 12, max_lag=36)  # ~1 year seasonality


# ### Process the data
# We again have a 12-month seasonality. This time we will not define monthly future covariates -> we let the model handle this itself!
# 
# Let's define past covariates instead. What if we used past data of heater sales to predict ice cream sales?

# In[ ]:


# convert monthly sales to average daily sales per month
converted_series = []
for col in ["ice cream", "heater"]:
    converted_series.append(
        series_ice_heater[col]
        / TimeSeries.from_series(series_ice_heater.time_index.days_in_month)
    )
converted_series = concatenate(converted_series, axis=1)
converted_series = converted_series[pd.Timestamp("20100101") :]

# define train/validation cutoff time
forecast_horizon_ice = 12
training_cutoff_ice = converted_series.time_index[-(2 * forecast_horizon_ice)]

# use ice cream sales as target, create train and validation sets and transform data
series_ice = converted_series["ice cream"]
train_ice, val_ice = series_ice.split_before(training_cutoff_ice)
transformer_ice = Scaler()
train_ice_transformed = transformer_ice.fit_transform(train_ice)
val_ice_transformed = transformer_ice.transform(val_ice)
series_ice_transformed = transformer_ice.transform(series_ice)

# use heater sales as past covariates and transform data
covariates_heat = converted_series["heater"]
cov_heat_train, cov_heat_val = covariates_heat.split_before(training_cutoff_ice)
transformer_heat = Scaler()
transformer_heat.fit(cov_heat_train)
covariates_heat_transformed = transformer_heat.transform(covariates_heat)


# ### Create a model with automatically generated future covariates and train it
# 
# Since we don't have future covariates defined, we need tell the model to generate future covariates itself.
# 
# - `add_encoders`: can add multiple encodings as past and / or future covariates from datetime attributes, cyclic repeating temporal patterns, index positions and customm functions for index encodings. You can even add a transformer that handles proper scaling of the training, validation and prediction data! Read more about it in the `TFTModel` docs [from here](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html#temporal-fusion-transformer-tft)
# - `add_relative_index`: adds a scaled integer position relative to the prediction point for each encoder-decoder chunk (this might be useful if you really don't want to use any future covariates. The position values remain constant over all chunks and do not add additional information).
# 
# We use `add_encoders={'cyclic': {'future': ['month']}}` to account for the 12-month seasonality as a future covariate..

# In[ ]:


# use the last 3 years as past input data
input_chunk_length_ice = 36

# use `add_encoders` as we don't have future covariates
my_model_ice = TFTModel(
    input_chunk_length=input_chunk_length_ice,
    output_chunk_length=forecast_horizon_ice,
    hidden_size=32,
    lstm_layers=1,
    batch_size=16,
    n_epochs=300,
    dropout=0.1,
    add_encoders={"cyclic": {"future": ["month"]}},
    add_relative_index=False,
    optimizer_kwargs={"lr": 1e-3},
    random_state=42,
)

# fit the model with past covariates
my_model_ice.fit(
    train_ice_transformed, past_covariates=covariates_heat_transformed, verbose=True
)


# ### Look at predictions on the validation set
# Again, we perform a one-shot prediction of 24 months using the "current" model - i.e., the model at the end of the training procedure:

# In[ ]:


n = 24
eval_model(
    model=my_model_ice,
    n=n,
    actual_series=series_ice_transformed[
        train_ice.end_time() - (2 * n - 1) * train_ice.freq :
    ],
    val_series=val_ice_transformed,
)


# ### Backtesting
# Let's backtest our `TFTModel` model, to see how it performs with a forecast horizon of 12 months over the last 2 years:

# In[ ]:


# Compute the backtest predictions with the two models
last_points_only = False
backtest_series_ice = my_model_ice.historical_forecasts(
    series_ice_transformed,
    num_samples=num_samples,
    start=training_cutoff_ice,
    forecast_horizon=forecast_horizon_ice,
    stride=1 if last_points_only else forecast_horizon_ice,
    retrain=False,
    last_points_only=last_points_only,
    overlap_end=True,
    verbose=True,
)

backtest_series_ice = (
    concatenate(backtest_series_ice)
    if isinstance(backtest_series_ice, list)
    else backtest_series_ice
)


# In[ ]:


eval_backtest(
    backtest_series=backtest_series_ice,
    actual_series=series_ice_transformed[
        train_ice.start_time() - 2 * forecast_horizon_ice * train_ice.freq :
    ],
    horizon=forecast_horizon_ice,
    start=training_cutoff_ice,
    transformer=transformer_ice,
)

