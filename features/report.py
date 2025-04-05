from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QTextEdit,QMessageBox
from PyQt5.QtCore import Qt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportFeature:
    def __init__(self, parent, db, project_name):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.widget = QWidget()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"REPORT FOR {self.project_name.upper()}")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        self.feature_widget = QWidget()
        self.feature_layout = QVBoxLayout()
        self.feature_widget.setLayout(self.feature_layout)
        self.feature_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")

        button_layout = QHBoxLayout()
        report_btn = QPushButton("Generate Project Report")
        report_btn.setStyleSheet("""
            QPushButton { background-color: #f39c12; color: white; border: none; padding: 5px; border-radius: 5px; }
            QPushButton:hover { background-color: #e67e22; }
        """)
        report_btn.clicked.connect(self.generate_report)
        button_layout.addWidget(report_btn)
        button_layout.addStretch()
        self.feature_layout.addLayout(button_layout)

        self.feature_result = QTextEdit()
        self.feature_result.setReadOnly(True)
        self.feature_result.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px; padding: 10px;")
        self.feature_result.setText(f"Project Report for {self.project_name}: Click 'Generate Project Report' to view details.")
        self.feature_layout.addWidget(self.feature_result)

        layout.addWidget(self.feature_widget)

    def generate_report(self):
        if not self.project_name:
            QMessageBox.warning(self.parent, "Error", "No project selected for Report!")
            return

        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        report = f"Project Report for {self.project_name}:\n"
        report += f"Total Tags: {len(tags_data)}\n"
        for tag in tags_data:
            tag_name = tag["tag_name"]
            data = self.db.get_tag_values(self.project_name, tag_name)
            report += f"\nTag: {tag_name}\n"
            report += f"  Total Messages: {len(data)}\n"
            if data:
                report += f"  Latest Timestamp: {data[-1]['timestamp']}\n"
                report += f"  Latest Values: {data[-1]['values'][-5:]}\n"
            else:
                report += "  No data available.\n"
        self.feature_result.setText(report)

    def on_data_received(self, tag_name, values):
        pass  # No real-time update needed for report

    def get_widget(self):
        return self.widget