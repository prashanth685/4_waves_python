from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QDateTimeEdit, QListWidget, 
                             QListWidgetItem, QPushButton, QTextEdit, QSizePolicy)
from PyQt5.QtCore import Qt, QDateTime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import logging
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeReportFeature:
    def __init__(self, parent, db, project_name):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.widget = QWidget()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.dragging = False
        self.press_x = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"TIME REPORT FOR {self.project_name.upper()}")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        self.time_report_widget = QWidget()
        self.time_report_layout = QVBoxLayout()
        self.time_report_widget.setLayout(self.time_report_layout)
        self.time_report_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")

        filter_layout = QHBoxLayout()
        
        from_label = QLabel("From:")
        from_label.setStyleSheet("color: white; font-size: 16px;")
        self.time_from_date = QDateTimeEdit()
        self.time_from_date.setCalendarPopup(True)
        self.time_from_date.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        current_time = QDateTime.currentDateTime()
        self.time_from_date.setDateTime(current_time.addSecs(-2 * 3600))
        # self.time_from_date.setDateTime(current_time)  # Default to current time, no automatic 1-hour offset
        self.time_from_date.setStyleSheet("""
            background-color: #34495e;
            color: white;
            border: 1px solid #1a73e8;
            padding: 5px;
            border-radius: 8px;
            font-size: 14px;
            font-family: Arial, sans-serif;
            transition: all 0.3s ease;
        """)
        
        to_label = QLabel("To:")
        to_label.setStyleSheet("color: white; font-size: 16px;")
        self.time_to_date = QDateTimeEdit()
        current_time = QDateTime.currentDateTime()

        self.time_to_date.setCalendarPopup(True)
        self.time_to_date.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.time_to_date.setDateTime(current_time)
        self.time_to_date.setStyleSheet("""background-color: #34495e;
            color: white;
            border: 1px solid #1a73e8;
            padding: 5px;
            border-radius: 8px;
            font-size: 14px;
            font-family: Arial, sans-serif;
            transition: all 0.3s ease;
        """)
        
        tag_label = QLabel("Select Tags:")
        tag_label.setStyleSheet("color: white; font-size: 16px;")
        self.time_report_tag_list = QListWidget()
        self.time_report_tag_list.setSelectionMode(QListWidget.MultiSelection)
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        if not tags_data:
            self.time_report_tag_list.addItem("No Tags Available")
        else:
            for tag in tags_data:
                self.time_report_tag_list.addItem(QListWidgetItem(tag["tag_name"]))
        self.time_report_tag_list.setStyleSheet("background-color: #34495e; color: white; border: 1px solid #1a73e8; padding: 5px; font-size:20px")

        # Add Fetch button
        fetch_btn = QPushButton("Fetch Data")
        fetch_btn.setStyleSheet("""
            QPushButton { background-color: #2ecc71; color: white; border: none; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #27ae60; }
        """)
        fetch_btn.clicked.connect(self.update_plot)

        filter_layout.addWidget(from_label)
        filter_layout.addWidget(self.time_from_date)
        filter_layout.addWidget(to_label)
        filter_layout.addWidget(self.time_to_date)
        filter_layout.addWidget(tag_label)
        filter_layout.addWidget(self.time_report_tag_list)
        filter_layout.addWidget(fetch_btn)
        filter_layout.addStretch()
        self.time_report_layout.addLayout(filter_layout)

        self.time_report_layout.addWidget(self.canvas)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)

        button_layout = QHBoxLayout()
        pdf_btn = QPushButton("Export to PDF")
        pdf_btn.setStyleSheet("""
            QPushButton { background-color: #3498db; color: white; border: none; padding: 5px; border-radius: 5px; }
            QPushButton:hover { background-color: #2980b9; }
        """)
        pdf_btn.clicked.connect(lambda: self.export_time_report_to_pdf(self.project_name))

        reset_btn = QPushButton("Reset View")
        reset_btn.setStyleSheet("""
            QPushButton { background-color: #f39c12; color: white; border: none; padding: 5px; border-radius: 5px; }
            QPushButton:hover { background-color: #e67e22; }
        """)
        reset_btn.clicked.connect(self.reset_view)

        button_layout.addWidget(pdf_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        self.time_report_layout.addLayout(button_layout)

        self.time_report_result = QTextEdit()
        self.time_report_result.setReadOnly(True)
        self.time_report_result.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px; padding: 10px;")
        self.time_report_result.setMinimumHeight(100)
        self.time_report_result.setText(f"Time Report for {self.project_name}: Select tags and time range, then click 'Fetch Data' to plot.\nUse mouse wheel to zoom, drag to pan.")
        self.time_report_layout.addWidget(self.time_report_result)
        self.time_report_layout.addStretch()

        layout.addWidget(self.time_report_widget)

        # Removed automatic update_plot call here

    def update_plot(self):
        selected_tags = [item.text() for item in self.time_report_tag_list.selectedItems()]
        if not selected_tags or "No Tags Available" in selected_tags:
            self.time_report_result.setText("No valid tags selected.")
            self.figure.clear()
            self.canvas.draw()
            return

        from_dt = self.time_from_date.dateTime().toPyDateTime()
        to_dt = self.time_to_date.dateTime().toPyDateTime()
        window_size = (to_dt - from_dt).total_seconds()

        if window_size <= 0:
            self.time_report_result.setText("Error: 'To' time must be after 'From' time.")
            self.figure.clear()
            self.canvas.draw()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = ['b', 'r', 'g', 'y', 'm', 'c']

        report = f"Time Report for {self.project_name} ({from_dt.strftime('%Y-%m-%d %H:%M:%S')} to {to_dt.strftime('%Y-%m-%d %H:%M:%S')}):\n"
        report += f"Selected Tags: {', '.join(selected_tags)}\n\n"

        for i, tag in enumerate(selected_tags):
            try:
                data = self.db.get_tag_values(self.project_name, tag)
                filtered_data = [entry for entry in data if from_dt <= datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S.%f") <= to_dt]
                logging.debug(f"Fetched {len(filtered_data)} entries for tag {tag} from {from_dt} to {to_dt}")
            except Exception as e:
                logging.error(f"Error fetching data for tag {tag}: {e}")
                report += f"Tag: {tag}\n  Error fetching data: {str(e)}\n"
                continue

            if filtered_data:
                timestamps = []
                values = []
                for entry in filtered_data:
                    try:
                        dt = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
                        timestamps.extend([dt] * len(entry["values"]))
                        values.extend(entry["values"])
                    except (ValueError, KeyError) as e:
                        logging.error(f"Error processing entry for tag {tag}: {e}")
                        continue

                if timestamps and values:
                    timestamps = np.array(timestamps)
                    values = np.array(values)
                    time_points = np.linspace(0, window_size, len(values))
                    ax.plot(time_points, values, f'{colors[i % len(colors)]}-', 
                            label=tag, linewidth=1.5, markersize=5, marker='o', color='darkblue')
                    logging.debug(f"Plotted {len(timestamps)} points for tag {tag}")

                    tick_positions = np.linspace(0, window_size, 10)
                    time_labels = []
                    for tick in tick_positions:
                        delta_seconds = tick - window_size
                        tick_dt = to_dt + timedelta(seconds=delta_seconds)
                        time_labels.append(tick_dt.strftime('%H:%M:%S') + f":{tick_dt.microsecond // 1000:03d}")
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(time_labels, rotation=0)

                    y_max = max(values)
                    y_min = min(values)
                    padding = (y_max - y_min) * 0.1 if y_max != y_min else 5000
                    ax.set_ylim(y_min - padding, y_max + padding)
                    ax.set_yticks(self.generate_y_ticks(values))

                    report += f"Tag: {tag}\n"
                    report += f"  Messages in Range: {len(filtered_data)}\n"
                    report += f"  Latest Value: {values[-1]:.2f}\n"
                    report += f"  Sample Data (last 5 entries):\n"
                    for entry in filtered_data[-5:]:
                        report += f"    {entry['timestamp']}: {entry['values'][:5]}\n"
                else:
                    report += f"Tag: {tag}\n  No valid data points after filtering.\n"
            else:
                report += f"Tag: {tag}\n  No data in selected time range.\n"
                logging.debug(f"No data found for tag {tag} in range {from_dt} to {to_dt}")

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("Time (HH:MM:SS)")
        ax.set_ylabel("Values", rotation=90, labelpad=10)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xlim(0, window_size)
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.time_report_result.setText(report)
        logging.debug(f"Time report and exact plot updated for tags: {selected_tags}")

    def generate_y_ticks(self, values):
        if values.size == 0 or not np.isfinite(values).all():
            return np.arange(16390, 46538, 5000)
        y_max = np.max(values)
        y_min = np.min(values)
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

    def reset_view(self):
        self.update_plot()
        logging.debug("Time report view reset")

    def on_scroll(self, event):
        if event.inaxes:
            ax = event.inaxes
            xlim = ax.get_xlim()
            x_range = xlim[1] - xlim[0]
            center = event.xdata if event.xdata is not None else xlim[0] + x_range / 2
            scale = 1.1 if event.button == 'down' else 0.9
            new_range = x_range * scale
            new_left = center - new_range / 2
            new_right = center + new_range / 2
            ax.set_xlim(new_left, new_right)
            tick_positions = np.linspace(new_left, new_right, 10)
            from_dt = self.time_from_date.dateTime().toPyDateTime()
            to_dt = self.time_to_date.dateTime().toPyDateTime()
            window_size = (to_dt - from_dt).total_seconds()
            time_labels = []
            for tick in tick_positions:
                delta_seconds = tick - window_size
                tick_dt = to_dt + timedelta(seconds=delta_seconds)
                time_labels.append(tick_dt.strftime('%H:%M:%S') + f":{tick_dt.microsecond // 1000:03d}")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(time_labels, rotation=0)
            self.canvas.draw_idle()
            logging.debug(f"Zoomed: new window size {new_range:.2f} seconds")

    def on_press(self, event):
        if event.inaxes and event.button == 1:
            self.dragging = True
            self.press_x = event.xdata

    def on_release(self, event):
        self.dragging = False

    def on_drag(self, event):
        if self.dragging and event.inaxes:
            ax = event.inaxes
            if self.press_x is not None and event.xdata is not None:
                dx = self.press_x - event.xdata
                xlim = ax.get_xlim()
                new_left = xlim[0] + dx
                new_right = xlim[1] + dx
                ax.set_xlim(new_left, new_right)
                from_dt = self.time_from_date.dateTime().toPyDateTime()
                to_dt = self.time_to_date.dateTime().toPyDateTime()
                window_size = (to_dt - from_dt).total_seconds()
                tick_positions = np.linspace(new_left, new_right, 10)
                time_labels = []
                for tick in tick_positions:
                    delta_seconds = tick - window_size
                    tick_dt = to_dt + timedelta(seconds=delta_seconds)
                    time_labels.append(tick_dt.strftime('%H:%M:%S') + f":{tick_dt.microsecond // 1000:03d}")
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(time_labels, rotation=0)
                self.press_x = event.xdata
                self.canvas.draw_idle()
                logging.debug(f"Panned: new xlim [{new_left}, {new_right}]")

    def export_time_report_to_pdf(self, project_name):
        try:
            report_text = self.time_report_result.toPlainText()
            logging.info(f"Exporting time report for {project_name} to PDF:\n{report_text[:100]}...")
            self.time_report_result.setText(f"{report_text}\n\n[Export to PDF functionality not fully implemented yet.]")
        except Exception as e:
            logging.error(f"Failed to export time report to PDF: {str(e)}")
            self.time_report_result.setText(f"Error exporting to PDF: {str(e)}")

    def get_widget(self):
        return self.widget

# Example usage with a dummy database
# if __name__ == "__main__":
#     from PyQt5.QtWidgets import QApplication
#     import sys
#     app = QApplication(sys.argv)
#     class DummyDB:
#         tags_collection = {"find": lambda x: [{"tag_name": "sarayu/tag2/topic2|m/s"}]}
#         def get_tag_values(self, project, tag):
#             if tag == "sarayu/tag2/topic2|m/s" and project == "sispl 10":
#                 return [
#                     {"timestamp": "2025-04-02T10:11:15.000", "values": [100, 102, 104]},
#                     {"timestamp": "2025-04-02T10:11:16.000", "values": [105, 107, 109]}
#                 ]
#             return []
#     window = TimeReportFeature(None, DummyDB(), "sispl 10")
#     window.get_widget().show()
#     sys.exit(app.exec_())