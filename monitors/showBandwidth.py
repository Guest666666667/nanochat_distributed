import pandas as pd
import matplotlib.pyplot as plt


def load_and_plot(file_path, label):
    df = pd.read_csv(file_path)
    start_time_ns = df['timestamp_ns'].iloc[0]
    df['time_s'] = (df['timestamp_ns'] - start_time_ns) / 1_000_000_000
    df_filtered = df[(df['time_s'] >= start_time_s) & (df['time_s'] <= end_time_s)]
    plt.plot(df_filtered['time_s'], df_filtered['tx_mbps'] + df_filtered['rx_mbps'], label=label, alpha=0.7)


# file1 = "../logs/consist_samples_dp1.csv"
# file2 = "../logs/consist_samples_dp2.csv"
file3 = "../logs/consist_samples_tp1.csv"
file4 = "../logs/consist_samples_tp2.csv"

# start_time_s = 20
# end_time_s = 70
start_time_s = 150
end_time_s = 200

plt.figure(figsize=(10, 5))
# load_and_plot(file1, "dp1")
# load_and_plot(file2, "dp2")
load_and_plot(file3, "tp1")
load_and_plot(file4, "tp2")

plt.xlabel("Time (s)")
plt.ylabel("Bandwidth (Mbps)")
plt.legend(loc="upper right")
plt.title(f"Comparison of Multiple Datasets ({start_time_s:.2f}-{end_time_s:.2f} s)")
plt.show()
