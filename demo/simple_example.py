import matplotlib.pyplot as plt
import numpy as np
from darts.utils import timeseries_generation as tg
np.random.seed(42)

LENGTH = 3 * 365  # 3年的逐日数据

# Melting: 具有年周期性和加性白噪声的正弦
melting = 0.9 * tg.sine_timeseries(
            length=LENGTH, value_frequency=(1 / 365), freq="D", column_name="melting"
            ) + 0.1 * tg.gaussian_timeseries(length=LENGTH, freq="D")

# Rainfalls: 具有双周周期和加性白噪声的正弦信号
rainfalls = 0.5 * tg.sine_timeseries(
            length=LENGTH, value_frequency=(1 / 14), freq="D", column_name="rainfall"
            ) + 0.5 * tg.gaussian_timeseries(length=LENGTH, freq="D")

# 我们将`Melting`提前 5 天
melting_contribution = 0.5 * melting.shift(5)

# 我们为rainfalls计算的类似分布
all_contributions = [melting_contribution] + [
            0.1 * rainfalls.shift(lag) for lag in range(5)
            ]

# 我们将最终流量计算为所有内容的总和；
# 修剪series，使它们都具有相同的开始时间
flow = sum(
            [
                        series[melting_contribution.start_time() :][: melting.end_time()]
                                for series in all_contributions
                                    ]
            ).with_columns_renamed("melting", "flow")

# 加入一些白噪声
flow += 0.1 * tg.gaussian_timeseries(length=len(flow))

plt.figure(figsize=(12, 5))
melting.plot()
rainfalls.plot()
flow.plot(lw=4)
plt.show()
print("flow plot")


