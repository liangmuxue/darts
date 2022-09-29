from examples.utils import fix_pythonpath_if_working_locally

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

figsize = (9, 6)
num_samples = 200
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

training_cutoff = 1200
transformer = Scaler()
forecast_horizon = 12
input_chunk_length = 20

def create_model():
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
    # 设置"day of week" 为动态离散变量
    categorical_embedding_sizes = {"dayofweek": 5}
    my_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=1024,
        n_epochs=300,
        add_relative_index=False,
        add_encoders=None,
        categorical_embedding_sizes=categorical_embedding_sizes,
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
    )
    return my_model

def eval_model(model, n, actual_series, val_series):
    pred_series = model.predict(n=n, num_samples=200)

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

def build_future_covariates(series):
    # 使用"day of week"作为未来已知协变量
    df = series.pd_dataframe()
    future_covariates = TimeSeries.from_times_and_values(
        times=df.index,
        values=df["dayofweek"].values,
        columns=["dayofweek"],
    )     
    return future_covariates  

def build_past_covariates(series,past_columns):
    """生成过去协变量系列"""
    
    past_covariates = []
    # 逐个生成每个列的协变量系列
    for column in past_columns:
        past = series.univariate_component(column)  
        past_covariates.append(past)   
    # 整合为一个系列
    past_covariates = concatenate(past_covariates, axis=1)
    return past_covariates  

def data_prepare():
    file_path = "/home/qdata/project/qlib/custom/data/aug/test_100_timeidx.pkl"
    df = pd.read_pickle(file_path)
    # group需要转换为数值型
    df['instrument'] = df['instrument'].apply(pd.to_numeric,errors='coerce')
    
    # dataframe转timeseries
    value_cols = ['dayofweek','CORD5', 'VSTD5', 'WVMA5', 'label','ori_label']
    series = TimeSeries.from_group_dataframe(df,
                                            time_col="time_idx",
                                             group_cols="instrument",
                                             freq='D',
                                             fill_missing_dates=True,
                                             # static_cols="instrument",
                                             value_cols=value_cols)
    # Create training and validation sets:
    series_transformed = []
    trains_transformed = []
    vals_transformed = []
    future_covariates = []
    past_covariates = []
    val_future_covariates = []
    val_past_covariates = []
    for s in series:
        try:
            train,val = s.split_after(training_cutoff)
        except Exception as e:
            # print("err",e)
            continue
        # Normalize the time series (note: we avoid fitting the transformer on the validation set)
        train_transformed = transformer.fit_transform(train)
        val_transformed = transformer.transform(val)
        s_transformed = transformer.transform(s)
        # 生成未来协变量
        future = build_future_covariates(train_transformed)     
        future_covariates.append(future)  
        val_future = build_future_covariates(val_transformed)     
        val_future_covariates.append(val_future)          
        # 生成过去协变量
        past = build_past_covariates(train_transformed,past_columns=['CORD5', 'VSTD5', 'WVMA5']) 
        past_covariates.append(past)
        val_past = build_past_covariates(val_transformed,past_columns=['CORD5', 'VSTD5', 'WVMA5']) 
        val_past_covariates.append(val_past)       
        # 删除其他的目标数据，target只保留一个数据 
        ignore_cols = ['dayofweek','CORD5', 'VSTD5', 'WVMA5', 'label']
        train_transformed = train_transformed.drop_columns(ignore_cols)
        val_transformed = val_transformed.drop_columns(ignore_cols)
        s_transformed = s_transformed.drop_columns(ignore_cols)
        trains_transformed.append(train_transformed)
        vals_transformed.append(val_transformed)
        series_transformed.append(s_transformed)
        
    return trains_transformed,vals_transformed,series_transformed,past_covariates,future_covariates,val_past_covariates,val_future_covariates
    

def process():
    # 取得训练数据及测试数据
    train_transformed,val_transformed,series_transformed,past_covariates,future_covariates,val_past_covariates,val_future_covariates = data_prepare()
    for item in train_transformed:
        find_nan(item)
    my_model = create_model()
    my_model.fit(train_transformed, past_covariates=past_covariates, future_covariates=future_covariates,
                 val_series=val_transformed,val_past_covariates=val_past_covariates,val_future_covariates=val_future_covariates,
                 verbose=True)
    eval_model(my_model, input_chunk_length, series_transformed, val_transformed)
    
    backtest_series = my_model.historical_forecasts(
        series_transformed, 
        start=train_transformed.end_time() + train_transformed.freq,
        num_samples=num_samples,
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    

    eval_backtest(
        backtest_series=concatenate(backtest_series),
        actual_series=series_transformed,
        horizon=forecast_horizon,
        start=training_cutoff,
        transformer=transformer,
    )
    
  

def find_nan(series):
   df = series.pd_dataframe()
   print(df.isnull().T.any())


def test():
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
    
if __name__ == "__main__":
    process()
    # test()
