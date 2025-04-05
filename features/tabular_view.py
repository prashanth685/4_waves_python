from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QTableWidget, QTableWidgetItem,QHeaderView
from PyQt5.QtCore import Qt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TabularViewFeature:
    def __init__(self, parent, db, project_name):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.widget = QWidget()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"TABULAR VIEW FOR {self.project_name.upper()}")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        tags_widget = QWidget()
        tags_layout = QVBoxLayout()
        tags_widget.setLayout(tags_layout)
        tags_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")

        filter_layout = QHBoxLayout()
        filter_label = QLabel("Select Tags:")
        filter_label.setStyleSheet("color: white; font-size: 14px;")
        self.tag_combo = QComboBox()
        self.tag_combo.addItem("All Tags")
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        for tag in tags_data:
            self.tag_combo.addItem(tag["tag_name"])
        self.tag_combo.setStyleSheet("background-color: #34495e; color: white; border: 1px solid #1a73e8; padding: 5px;")
        self.tag_combo.currentTextChanged.connect(self.update_tabular_view)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.tag_combo)
        filter_layout.addStretch()
        tags_layout.addLayout(filter_layout)

        self.tabular_table = QTableWidget()
        self.tabular_table.setColumnCount(3)
        self.tabular_table.setHorizontalHeaderLabels(["FULL TAG", "TIMESTAMP", "VALUE"])
        self.tabular_table.setStyleSheet("""
            QTableWidget { background-color: #34495e; color: white; border: none; gridline-color: #2c3e50; }
            QTableWidget::item { padding: 5px; border: none; }
            QHeaderView::section { background-color: #1a73e8; color: white; border: none; padding: 10px; font-size: 14px; }
        """)
        self.tabular_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabular_table.verticalHeader().setVisible(False)
        self.update_tabular_view()
        tags_layout.addWidget(self.tabular_table)

        layout.addWidget(tags_widget)

    def update_tabular_view(self):
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        selected_tag = self.tag_combo.currentText()

        filtered_tags = tags_data if selected_tag == "All Tags" else [tag for tag in tags_data if tag["tag_name"] == selected_tag]
        self.tabular_table.setRowCount(len(filtered_tags))
        for row, tag in enumerate(filtered_tags):
            self.tabular_table.setItem(row, 0, QTableWidgetItem(tag["tag_name"]))
            latest_data = self.db.get_tag_values(self.project_name, tag["tag_name"])
            timestamp = latest_data[-1]["timestamp"] if latest_data else "N/A"
            value = latest_data[-1]["values"][-1] if latest_data else "N/A"
            self.tabular_table.setItem(row, 1, QTableWidgetItem(timestamp))
            self.tabular_table.setItem(row, 2, QTableWidgetItem(str(value)))

    def on_data_received(self, tag_name, values):
        self.update_tabular_view()

    def get_widget(self):
        return self.widget