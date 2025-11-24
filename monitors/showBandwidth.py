import pandas as pd
import matplotlib.pyplot as plt


def load_and_plot(file_path, label):
    df = pd.read_csv(file_path)
    start_time_ns = df['timestamp_ns'].iloc[0]
    df['time_s'] = (df['timestamp_ns'] - start_time_ns) / 1_000_000_000
    df_filtered = df[(df['time_s'] >= start_time_s) & (df['time_s'] <= end_time_s)]
    plt.plot(df_filtered['time_s'], df_filtered['tx_mbps'] + df_filtered['rx_mbps'], label=label, alpha=0.7)


# file2 = "../save_logs/remote_2jobs_competition_mul/approach2/multiple_3_5_d300/consist_samples_tp1.csv"
# file2 = "../logs/consist_samples_tp1.csv"
file2 = "../logs/consist_samples_dp1.csv"
# file3 = "../save_logs/remote_2jobs_competition_mul/approach1/single_3_5/dynamic_samples.csv"
# file4 = "../save_logs/remote_2jobs_competition_mul/approach1/single_3_5_2/dynamic_samples.csv"

start_time_s = 0
end_time_s = 70

plt.figure(figsize=(10, 5))
# load_and_plot(file1, "Node3&5_multiple")
load_and_plot(file2, "Node3&5_single")

plt.xlabel("Time (s)")
plt.ylabel("Bandwidth (Mbps)")
plt.legend(loc="upper right")
plt.title(f"Comparison of Multiple Datasets ({start_time_s:.2f}-{end_time_s:.2f} s)")
plt.show()
