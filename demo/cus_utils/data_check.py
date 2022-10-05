import numpy as np
import pandas as pd

def check_time_ser_data(file_path):
    """检查原数据的时间长度"""
        
    df = pd.read_pickle(file_path)
    cnt = df.groupby("instrument").count()
    print(cnt)

if __name__ == "__main__":
    file_path = "/home/qdata/project/qlib/custom/data/aug/test_all_timeidx.pkl"
    check_time_ser_data(file_path)