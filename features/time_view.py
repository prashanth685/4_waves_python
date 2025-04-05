from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QTextEdit, QScrollArea
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeViewFeature:
    def __init__(self, parent, db, project_name):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.widget = QWidget()
        self.mqtt_tag = None
        self.initial_buffer_size = 4096
        self.time_view_buffers = [deque(maxlen=self.initial_buffer_size) for _ in range(4)]
        self.time_view_timestamps = deque(maxlen=self.initial_buffer_size * 4)
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self.update_time_view_plot)
        self.figure = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.dragging = False
        self.press_x = None
        self.last_data_time = None
        self.data_rate = 1.0
        self.annotations = []  # For hover value display
        self.initUI()
        self.connect_events()

    def initUI(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"TIME VIEW FOR {self.project_name.upper()}")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        self.time_widget = QWidget()
        self.time_layout = QVBoxLayout()
        self.time_widget.setLayout(self.time_layout)
        self.time_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")

        tag_layout = QHBoxLayout()
        tag_label = QLabel("Select Tag:")
        tag_label.setStyleSheet("color: white; font-size: 16px;")
        self.tag_combo = QComboBox()
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        if not tags_data:
            self.tag_combo.addItem("No Tags Available")
        else:
            for tag in tags_data:
                self.tag_combo.addItem(tag["tag_name"])
        self.tag_combo.setStyleSheet("background-color: #34495e; color: white; border: 1px solid #1a73e8; padding: 15px")
        self.tag_combo.currentTextChanged.connect(self.setup_time_view_plot)

        tag_layout.addWidget(tag_label)
        tag_layout.addWidget(self.tag_combo)
        tag_layout.addStretch()
        self.time_layout.addLayout(tag_layout)

        self.time_layout.addWidget(self.canvas)

        self.time_result = QTextEdit()
        self.time_result.setReadOnly(True)
        self.time_result.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px; padding: 10px;")
        self.time_result.setMinimumHeight(50)
        self.time_result.setText(
            f"Time View for {self.project_name}: Select a tag to start real-time plotting.\n"
            "Drag to pan, scroll to zoom, hover for values."
        )
        self.time_layout.addWidget(self.time_result)
        self.time_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.time_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #2c3e50; border: none;")
        scroll_area.setMinimumHeight(400)
        layout.addWidget(scroll_area)

        if tags_data:
            self.tag_combo.setCurrentIndex(0)
            self.setup_time_view_plot(self.tag_combo.currentText())

    def connect_events(self):
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.inaxes in self.axes and event.button == 1:  # Left mouse button
            self.dragging = True
            self.press_x = event.xdata
            self.active_ax = event.inaxes

    def on_release(self, event):
        self.dragging = False
        self.press_x = None
        self.active_ax = None

    def on_motion(self, event):
        if self.dragging and event.inaxes == self.active_ax and event.xdata is not None:
            dx = self.press_x - event.xdata
            xlim = self.active_ax.get_xlim()
            new_xlim = (xlim[0] + dx, xlim[1] + dx)
            self.active_ax.set_xlim(new_xlim)
            self.canvas.draw_idle()
            self.press_x = event.xdata

        # Hover value display
        for ann in self.annotations:
            ann.remove()
        self.annotations.clear()

        if event.inaxes in self.axes and event.xdata is not None and event.ydata is not None:
            ax_idx = self.axes.index(event.inaxes)
            window_values = list(self.time_view_buffers[ax_idx])[-self.samples_per_window:]
            time_points = np.linspace(self.axes[ax_idx].get_xlim()[0], self.axes[ax_idx].get_xlim()[1], len(window_values))
            if window_values:
                idx = np.argmin(np.abs(time_points - event.xdata))
                value = window_values[idx]
                ann = event.inaxes.annotate(f"{value:.2f}", xy=(event.xdata, event.ydata),
                                            xytext=(5, 5), textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
                self.annotations.append(ann)
                self.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes in self.axes:
            ax = event.inaxes
            xlim = ax.get_xlim()
            center = (xlim[0] + xlim[1]) / 2
            width = xlim[1] - xlim[0]
            zoom_factor = 1.2 if event.step < 0 else 0.8  # Zoom out or in
            new_width = width * zoom_factor
            new_xlim = (center - new_width / 2, center + new_width / 2)
            ax.set_xlim(new_xlim)
            self.canvas.draw_idle()

    def setup_time_view_plot(self, tag_name):
        if not self.project_name or not tag_name or tag_name == "No Tags Available":
            logging.warning("No project or valid tag selected for Time View!")
            return

        self.mqtt_tag = tag_name
        self.timer.stop()
        self.timer.setInterval(100)
        for buf in self.time_view_buffers:
            buf.clear()
        self.time_view_timestamps.clear()
        self.last_data_time = None
        self.data_rate = 1.0

        data = self.db.get_tag_values(self.project_name, self.mqtt_tag)
        if data:
            for entry in data[-2:]:
                self.split_and_store_values(entry["values"], entry["timestamp"])

        self.figure.clear()
        self.axes = [self.figure.add_subplot(4, 1, i+1) for i in range(4)]
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], f'C{i}-', linewidth=1.5)
            self.lines.append(line)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylabel(f"Values {i+1}", rotation=90, labelpad=10)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_xlabel("Time (HH:MM:SSS)")
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 10))

        self.figure.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.15, hspace=0.4)
        self.canvas.setMinimumSize(1000, 800)
        self.time_widget.setMinimumSize(1000, 850)
        self.canvas.draw()
        self.timer.start()

    def split_and_store_values(self, values, timestamp):
        for i in range(10, min(len(values), 4096 * 4), 4):
            if i + 3 < len(values):
                self.time_view_buffers[0].append(values[i])
                self.time_view_buffers[1].append(values[i + 1])
                self.time_view_buffers[2].append(values[i + 2])
                self.time_view_buffers[3].append(values[i + 3])
                self.time_view_timestamps.extend([timestamp] * 4)
        logging.debug(values)

    def adjust_buffer_size(self):
        xlim = self.axes[0].get_xlim()
        window_size = xlim[1] - xlim[0]
        if self.data_rate > 0:
            new_buffer_size = max(int(self.data_rate * window_size * 2), 100)
            if new_buffer_size != self.time_view_buffers[0].maxlen:
                for i in range(4):
                    self.time_view_buffers[i] = deque(self.time_view_buffers[i], maxlen=new_buffer_size)
                self.time_view_timestamps = deque(self.time_view_timestamps, maxlen=new_buffer_size * 4)
                logging.debug(f"Adjusted buffer size to {new_buffer_size} based on data rate {self.data_rate:.2f} samples/s")

    def generate_y_ticks(self, values):
        if not values or not all(np.isfinite(v) for v in values):
            return np.arange(16390, 46538, 5000)
        y_max = max(values)
        y_min = min(values)
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
        y_max += padding
        y_min -= padding
        range_val = y_max - y_min
        step = max(range_val / 10, 1)
        step = np.ceil(step / 500) * 500
        ticks = []
        current = np.floor(y_min / step) * step
        while current <= y_max:
            ticks.append(current)
            current += step
        return ticks

    def update_time_view_plot(self):
        if not self.project_name or not self.mqtt_tag:
            self.time_result.setText("No project or tag selected for Time View.")
            return

        current_buffer_size = len(self.time_view_buffers[0])
        if current_buffer_size < 2:
            self.time_result.setText(
                f"Waiting for sufficient data for {self.mqtt_tag} (Current buffer: {current_buffer_size}/{self.time_view_buffers[0].maxlen})."
            )
            return

        self.adjust_buffer_size()

        for i, (ax, line) in enumerate(zip(self.axes, self.lines)):
            xlim = ax.get_xlim()
            window_size = xlim[1] - xlim[0]
            self.samples_per_window = min(current_buffer_size, int(self.data_rate * window_size))
            if self.samples_per_window < 2:
                self.samples_per_window = 2

            window_values = list(self.time_view_buffers[i])[-self.samples_per_window:]
            window_timestamps = list(self.time_view_timestamps)[-self.samples_per_window * 4:][::4]

            if not window_values or not all(np.isfinite(v) for v in window_values):
                self.time_result.setText(f"Invalid data received for {self.mqtt_tag}. Buffer: {current_buffer_size}")
                ax.set_ylim(16390, 46537)
                ax.set_yticks(self.generate_y_ticks([]))
                line.set_data([], [])
                continue

            time_points = np.linspace(xlim[0], xlim[1], self.samples_per_window)
            line.set_data(time_points, window_values)
            y_max = max(window_values)
            y_min = min(window_values)
            padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.set_yticks(self.generate_y_ticks(window_values))

            if window_timestamps:
                latest_dt = datetime.strptime(window_timestamps[-1], "%Y-%m-%dT%H:%M:%S.%f")
                time_labels = []
                tick_positions = np.linspace(xlim[0], xlim[1], 10)
                for tick in tick_positions:
                    delta_seconds = tick * window_size - window_size
                    tick_dt = latest_dt + timedelta(seconds=delta_seconds)
                    milliseconds = tick_dt.microsecond // 1000
                    time_labels.append(f"{tick_dt.strftime('%H:%M:%S:')}{milliseconds:03d}")
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(time_labels, rotation=0)  # Rotate for better readability

        self.canvas.draw_idle()
        latest_values = [list(buf)[-1] for buf in self.time_view_buffers if buf]
        self.time_result.setText(
            f"Time View Data for {self.mqtt_tag}, Latest values: {[f'{v:.2f}' for v in latest_values]}, "
            f"Window: {window_size:.2f}s, Buffer: {current_buffer_size}/{self.time_view_buffers[0].maxlen}, "
            f"Data rate: {self.data_rate:.2f} samples/s"
        )

    def on_data_received(self, tag_name, values):
        if tag_name == self.mqtt_tag:
            current_time = datetime.now()
            if self.last_data_time:
                time_delta = (current_time - self.last_data_time).total_seconds()
                if time_delta > 0:
                    self.data_rate = len(values) / time_delta / 4
            self.last_data_time = current_time
            self.split_and_store_values(values, current_time.isoformat())
            logging.debug(f"Time View - Received {len(values)} values for {tag_name}, Data rate: {self.data_rate:.2f} samples/s")

    def get_widget(self):
        return self.widget