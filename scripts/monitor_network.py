# monitor_network.py
import time
import psutil
import csv
import numpy as np
import threading
from scipy.fft import fft, fftfreq


class NetworkMonitor:
    def __init__(self, interface="ib0", sample_interval=10,
                 enable_file_logging=True,
                 dynamic_mode=False,
                 dynamic_start=10000,
                 dynamic_length=1000,
                 dynamic_interval=15000,
                 request_queue=None,
                 response_queue=None):
        self.interface = interface
        self.sample_interval = sample_interval / 1000
        self.enable_file_logging = enable_file_logging
        self.dynamic_mode = dynamic_mode
        self.dynamic_start = dynamic_start / 1000
        self.dynamic_length = dynamic_length
        self.dynamic_interval = dynamic_interval / 1000

        self.consist_data_file = "logs/consist_samples_tp1.csv"
        self.dynamic_data_file = "logs/dynamic_samples.csv"
        self.periods_results_file = "logs/periods_results.csv"

        if enable_file_logging:
            with open(self.consist_data_file, 'w') as f:
                csv.writer(f).writerow(["timestamp_ns", "tx_mbps", "rx_mbps"])
            if dynamic_mode:
                with open(self.dynamic_data_file, 'w') as f:
                    csv.writer(f).writerow(["timestamp_ns", "tx_mbps", "rx_mbps"])
                with open(self.periods_results_file, 'w') as f:
                    csv.writer(f).writerow(["capture_id", "start_ns", "end_ns", "period_ms"])

        self.capture_count = 0
        self.dynamic_buffer = []
        self.monitor_start_time = None
        self.in_capture_window = False
        self.last_tx = 0
        self.last_rx = 0
        self.last_time = time.time_ns()

        self.request_queue = [] if request_queue is None else request_queue
        self.response_queue = [] if response_queue is None else response_queue

    def _get_bytes(self):
        stats = psutil.net_io_counters(pernic=True).get(self.interface)
        return (stats.bytes_sent, stats.bytes_recv) if stats else (0, 0)

    def _calculate_mbps(self, current_tx, current_rx, time_elapsed_ns):
        time_elapsed_sec = time_elapsed_ns / 1e9
        tx_bps = (current_tx - self.last_tx) * 8
        rx_bps = (current_rx - self.last_rx) * 8
        return (
            tx_bps / (time_elapsed_sec * 1e6),
            rx_bps / (time_elapsed_sec * 1e6)
        )

    def _process_dynamic_data(self):
        if len(self.dynamic_buffer) < self.dynamic_length:
            return

        timestamps = np.array([x[0] for x in self.dynamic_buffer])
        tx = np.array([x[1] for x in self.dynamic_buffer])
        rx = np.array([x[2] for x in self.dynamic_buffer])

        relative_time = (timestamps - timestamps[0]) / 1e6
        bandwidth = tx + rx
        period = compute_fft_period(bandwidth, relative_time)

        if self.enable_file_logging:
            with open(self.periods_results_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.capture_count,
                    timestamps[0],
                    timestamps[-1],
                    f"{period:.2f}" if period else "N/A"
                ])

            self.dynamic_buffer.insert(0, (self.dynamic_buffer[0][0] - 1, 0.0, 0.0))
            self.dynamic_buffer.append((self.dynamic_buffer[-1][0] + 1, 0.0, 0.0))
            with open(self.dynamic_data_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(self.dynamic_buffer)

        self.capture_count += 1
        self.dynamic_buffer.clear()

    def _handle_dynamic_mode(self, timestamp, tx, rx):
        current_time = time.time()

        if self.monitor_start_time is None:
            self.monitor_start_time = current_time + self.dynamic_start
            return

        if current_time < self.monitor_start_time:
            return

        if not self.in_capture_window:
            self.in_capture_window = True

        self.dynamic_buffer.append((timestamp, tx, rx))

        if len(self.dynamic_buffer) >= self.dynamic_length:
            self._process_dynamic_data()
            self.monitor_start_time = current_time + self.dynamic_interval
            self.in_capture_window = False

    def _process_queue(self):
        last_timestamp = None
        while True:
            timestamps = []
            if last_timestamp is not None:
                timestamps.append(last_timestamp)

            while not self.request_queue.empty():
                job_id, timestamp = self.request_queue.get_nowait()
                timestamps.append(timestamp)

            if len(timestamps) > 1:
                intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
                avg_interval = sum(intervals) / len(intervals)
                self.response_queue.put(avg_interval)
                time.sleep(self.dynamic_interval)

            if len(timestamps) > 0:
                last_timestamp = timestamps[-1]

    def run(self):
        queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        queue_thread.start()
        while True:
            current_time_ns = time.time_ns()
            current_tx, current_rx = self._get_bytes()

            if self.last_time != 0:
                elapsed_ns = current_time_ns - self.last_time
                tx_mbps, rx_mbps = self._calculate_mbps(current_tx, current_rx, elapsed_ns)

                # Basic file log
                if self.enable_file_logging:
                    with open(self.consist_data_file, 'a') as f:
                        csv.writer(f).writerow([current_time_ns, tx_mbps, rx_mbps])

                # Dynamic mode
                if self.dynamic_mode:
                    self._handle_dynamic_mode(current_time_ns, tx_mbps, rx_mbps)

            self.last_tx, self.last_rx = current_tx, current_rx
            self.last_time = current_time_ns
            time.sleep(self.sample_interval)


def compute_fft_period(bandwidth_series, time_stamps):
    sample_interval_ms = np.mean(np.diff(time_stamps))
    sample_interval = sample_interval_ms / 1000
    n = len(bandwidth_series)

    detrended = bandwidth_series - np.mean(bandwidth_series)
    windowed = detrended * np.hanning(n)

    fft_vals = fft(windowed)
    freqs = fftfreq(n, d=sample_interval)
    magnitudes = np.abs(fft_vals)

    valid_mask = (freqs > 0) & (freqs < 1 / (2 * sample_interval))
    if not np.any(valid_mask):
        return 0

    main_freq_hz = freqs[valid_mask][np.argmax(magnitudes[valid_mask])]
    MIN_FREQ_THRESHOLD = 1e-6
    return 1000 / main_freq_hz if main_freq_hz > MIN_FREQ_THRESHOLD else 0


if __name__ == "__main__":
    monitor = NetworkMonitor(
        interface="enp0s31f6",
        sample_interval=10,
        enable_file_logging=True,
        dynamic_mode=True,
        dynamic_start=20000,
        dynamic_length=1000,
        dynamic_interval=10000
    )
    monitor.run()
