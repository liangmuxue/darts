from examples.utils import fix_pythonpath_if_working_locally

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings
from darts.utils import data
from darts.models.forecasting.block_rnn_model import BlockRNNModel
# from akshare.bond.bond_bank import df
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

from demo.cus_utils.tensor_viz import TensorViz


figsize = (9, 6)
num_samples = 200
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

training_cutoff = 900
transformer = Scaler()
forecast_horizon = 5
input_chunk_length = 15
emb_size = 955
value_cols = ['dayofweek','STD5', 'VSTD5', 'label','ori_label']
past_columns = ['STD5', 'VSTD5','ori_label']

viz_target = TensorViz(env="data_target") 
viz_input = TensorViz(env="data_input") 

def create_model():
    my_model = BlockRNNModel(
        model="LSTM",
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_dim=20,
        dropout=0,
        batch_size=4096,
        n_epochs=300,

        # loss_fn=MSELoss(),
        random_state=42,
        model_name="lstm_2",
        log_tensorboard=True,
        save_checkpoints=True,
        work_dir="demo/darts_log",
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
    )
    # model_name = "2022-10-05_21.20.35.159335_torch_model_run_13604"
    # my_model = TFTModel.load_from_checkpoint(model_name,work_dir="demo/darts_log")
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
    file_path = "/home/qdata/project/qlib/custom/data/aug/test_all_timeidx.pkl"
    df = pd.read_pickle(file_path)
    # 清洗数据
    df = data_clean(df)
    vis_target_ser(df)
    # 使用移动平均值作为目标数值
    # df['ori_label'] = df['ori_label'].rolling(window=5,min_periods=1).mean()
    vis_target_ser(df,label="target_ser_rolling")
    # group需要转换为数值型
    df['instrument'] = df['instrument'].apply(pd.to_numeric,errors='coerce')
    
    # dataframe转timeseries,使用group模式，每个股票生成一个序列,
    series = TimeSeries.from_group_dataframe(df,
                                            time_col="time_idx",
                                             group_cols="instrument",# group_cols会自动成为静态协变量
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
            print("err",e)
            continue
        # Normalize the time series (note: we avoid fitting the transformer on the validation set)
        train_transformed = transformer.fit_transform(train)
        val_transformed = transformer.transform(val)
        # 剔除训练接transform后的极值--不需要
        # val_transformed = val_data_clean(val_transformed,columns=past_columns)
        # 保留原序列用于回测
        s_transformed = transformer.transform(s)
        
        # 生成未来协变量
        future = build_future_covariates(train_transformed)     
        future_covariates.append(future)  
        val_future = build_future_covariates(val_transformed)     
        val_future_covariates.append(val_future)          
        # 生成过去协变量
        past = build_past_covariates(train_transformed,past_columns=past_columns) 
        past_covariates.append(past)
        val_past = build_past_covariates(val_transformed,past_columns=past_columns) 
        val_past_covariates.append(val_past)       
        # 删除其他的目标数据，target只保留一个数据 
        ignore_cols = ['dayofweek','STD5', 'VSTD5', 'label']
        train_transformed = train_transformed.drop_columns(ignore_cols)
        val_transformed = val_transformed.drop_columns(ignore_cols)
        s_transformed = s_transformed.drop_columns(ignore_cols)
        trains_transformed.append(train_transformed)
        vals_transformed.append(val_transformed)
        series_transformed.append(s_transformed)
        
    # 可视化部分   
    vis_target(series_transformed,type="ser")
    vis_target(trains_transformed,type="train")
    vis_input(past_covariates,type="train",columns=past_columns)
    vis_target(vals_transformed,type="val")
    vis_input(val_past_covariates,type="val",columns=past_columns)    
    return trains_transformed,vals_transformed,series_transformed,past_covariates,future_covariates,val_past_covariates,val_future_covariates
 
def data_clean(data):
    # 清除序列长度不够的股票,需要多余训练数据的一定比例
    thr_number = int(training_cutoff * 1.2)
    gcnt = data.groupby("instrument").count()
    index = gcnt[gcnt['time_idx']>thr_number].index
    data = data[data['instrument'].isin(index)]
    return data

def val_data_clean(data_transformed,columns=[]):
    # 清理验证集数据，剔除极值
    df = data_transformed.pd_dataframe()
    for column in columns:
        # 只保留合适的数值，范围为-3到3
        df = df[(df[column]<3)  & (df[column]>-3)]
    df["time_idx"] = df.index
    
    data_transformed = TimeSeries.from_dataframe(df,time_col="time_idx",value_cols=value_cols,static_covariates=df["instrument"])
    return data_transformed
    
def vis_target(data_sers,type="train"):
    """查看目标数据"""
    data = None
    for item in data_sers:
        item_df = item.pd_dataframe()
        # 对于测试集，查看0-1之间的数据
        if type=="val":
            item_df = item_df[(item_df["ori_label"]>0)&(item_df["ori_label"]<1)] 
        values = item_df.values.reshape(-1)
        if data is None:
            data = values
        else:
            data = np.concatenate((data,values),axis=0) 
    label = "target_{}".format(type)
    viz_target.viz_data_hist(data,numbins=20,win=label,title=label)
    
def vis_target_ser(df,label="target_ser"):
    """查看目标序列数据"""
    data = df[df['instrument']<9]
    data = data.pivot(index="instrument",columns="time_idx",values="ori_label")
    data = data.values.transpose(1,0)
    viz_target.viz_matrix_var(data,win=label,title=label)
    
def vis_input(data_sers,type="train",columns=[]):
    """查看输入数据"""
    df_list = []
    for item in data_sers:
        df_list.append(item.pd_dataframe())
    for index,column in enumerate(columns):
        data = None
        for item in df_list:
            values = item[column].values.reshape(-1)
            if data is None:
                data = values
            else:
                data = np.concatenate((data,values),axis=0) 
        label = "input_{}_{}".format(type,column)
        viz_input.viz_data_hist(data,numbins=20,win=label,title=label)
      
def process():
    # 取得训练数据及测试数据
    train_transformed,val_transformed,series_transformed,past_covariates,future_covariates,val_past_covariates,val_future_covariates = data_prepare()
    # 过滤空值
    for item in train_transformed:
        find_nan(item)
    
    my_model = create_model()
    my_model.fit(train_transformed, past_covariates=past_covariates,
                 val_series=val_transformed,val_past_covariates=val_past_covariates,
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
