import pandas as pd
import matplotlib.pyplot as plt


def load_and_plot(file_path, label):
    df = pd.read_csv(file_path)
    start_time_ns = df['timestamp_ns'].iloc[0]
    df['time_s'] = (df['timestamp_ns'] - start_time_ns) / 1_000_000_000
    df_filtered = df[(df['time_s'] >= start_time_s) & (df['time_s'] <= end_time_s)]
    plt.plot(df_filtered['time_s'], df_filtered['tx_mbps'] + df_filtered['rx_mbps'], label=label, alpha=0.7)


file1 = "../logs/consist_samples_dp_1ns.csv"
file2 = "../logs/consist_samples_dp_5ns.csv"
file3 = "../logs/consist_samples_dp_20ns.csv"
# file1 = "../logs/consist_samples_tp_1ns.csv"
# file2 = "../logs/consist_samples_tp_5ns.csv"
# file3 = "../logs/consist_samples_tp_20ns.csv"

start_time_s =50
end_time_s = 65;
# start_time_s = 110
# end_time_s = 112
#
plt.figure(figsize=(10, 5))
load_and_plot(file1, "Dp1_1ns")
load_and_plot(file2, "Dp2_5ns")
load_and_plot(file3, "Dp3_20ns")
# load_and_plot(file1, "tp1")
# load_and_plot(file2, "tp2")
# load_and_plot(file3, "tp3")

plt.xlabel("Time (s)")
plt.ylabel("Bandwidth (Mbps)")
plt.legend(loc="upper right")
plt.title(f"Comparison of Multiple Datasets (depth=12, batch_size=2, {start_time_s:.2f}-{end_time_s:.2f} s)")
plt.show()
