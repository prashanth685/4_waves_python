from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QHeaderView,QInputDialog,QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import Qt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CreateTagsFeature:
    def __init__(self, parent, db, project_name):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.widget = QWidget()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"MANAGE TAGS FOR {self.project_name.upper()}")
        header.setStyleSheet("color: white; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        tags_widget = QWidget()
        tags_layout = QVBoxLayout()
        tags_widget.setLayout(tags_layout)
        tags_widget.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")

        add_tag_form = QHBoxLayout()
        self.tag_name_input = QLineEdit()
        self.tag_name_input.setPlaceholderText("Enter full tag (e.g., sarayu/tag1/topic1|m/s)")
        self.tag_name_input.setStyleSheet("background-color: #34495e; color: white; border: 1px solid #1a73e8; padding: 10px; height: 30px;width:50%;font-size:15px")

        add_tag_btn = QPushButton("Add Tag")
        add_tag_btn.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; border: none; padding: 10px; border-radius: 5px; height: 30px; }
            QPushButton:hover { background-color: #218838; }
        """)
        add_tag_btn.clicked.connect(self.add_tag)

        add_tag_form.addWidget(self.tag_name_input)
        add_tag_form.addWidget(add_tag_btn)
        add_tag_form.addStretch()
        tags_layout.addLayout(add_tag_form)

        self.tags_table = QTableWidget()
        self.tags_table.setColumnCount(3)
        self.tags_table.setHorizontalHeaderLabels(["FULL TAG", "VALUE", "ACTIONS"])
        self.tags_table.setStyleSheet("""
            QTableWidget { background-color: #34495e; color: white; border: none; gridline-color: #2c3e50;padding:10px 10px }
            QTableWidget::item { padding: 5px 5px; border: none; }
    QHeaderView::section {
        background-color: #1a73e8;
        color: white;
        border: none;
        padding: 15px 15px;
        font-size: 12px;
        font-weight: bold;
    }
    QTableWidget QHeaderView::section:horizontal {
        border-right: 1px solid #2c3e50;
    }
        """)
        self.tags_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tags_table.verticalHeader().setVisible(False)
        self.update_table()
        tags_layout.addWidget(self.tags_table)

        layout.addWidget(tags_widget)

    def update_table(self):
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        self.tags_table.setRowCount(len(tags_data))
        for row, tag in enumerate(tags_data):
            self.tags_table.setItem(row, 0, QTableWidgetItem(tag["tag_name"]))
            latest_data = self.db.get_tag_values(self.project_name, tag["tag_name"])
            value = latest_data[-1]["values"][-1] if latest_data else "N/A"
            self.tags_table.setItem(row, 1, QTableWidgetItem(str(value)))

            actions_widget = QWidget()
            actions_layout = QHBoxLayout()
            actions_widget.setLayout(actions_layout)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(5)
            actions_layout.setAlignment(Qt.AlignCenter)

            edit_btn = QPushButton("Edit")
            edit_btn.setFixedSize(60, 30)
            edit_btn.setStyleSheet("""
                QPushButton { background-color: #3498db; color: white; border: none; border-radius: 5px; padding: 5px; }
                QPushButton:hover { background-color: #2980b9; }
            """)
            edit_btn.clicked.connect(lambda checked, r=row: self.edit_tag(r))

            delete_btn = QPushButton("Delete")
            delete_btn.setFixedSize(60, 30)
            delete_btn.setStyleSheet("""
                QPushButton { background-color: #e74c3c; color: white; border: none; border-radius: 5px; padding: 5px; }
                QPushButton:hover { background-color: #c0392b; }
            """)
            delete_btn.clicked.connect(lambda checked, r=row: self.delete_tag(r))

            actions_layout.addWidget(edit_btn)
            actions_layout.addWidget(delete_btn)
            self.tags_table.setCellWidget(row, 2, actions_widget)

    def add_tag(self):
        tag_string = self.tag_name_input.text().strip()
        tag_data = self.db.parse_tag_string(tag_string)
        if tag_data is None:
            return

        success, message = self.db.add_tag(self.project_name, tag_data)
        if success:
            self.tag_name_input.clear()
            if self.parent.mqtt_handler:
                self.parent.mqtt_handler.client.subscribe(tag_data["tag_name"])
            self.update_table()
        else:
            QMessageBox.warning(self.parent, "Error", message)

    def edit_tag(self, row):
        tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
        if row >= len(tags_data):
            return
        tag = tags_data[row]
        old_tag_string = tag["tag_name"]
        new_tag_string, ok = QInputDialog.getText(self.parent, "Edit Tag", "Enter new tag (e.g., sarayu/tag1/topic1|m/s):", text=old_tag_string)
        if ok and new_tag_string:
            new_tag_data = self.db.parse_tag_string(new_tag_string)
            if new_tag_data is None:
                return
            if self.parent.mqtt_handler:
                self.parent.mqtt_handler.client.unsubscribe(tag["tag_name"])
                self.parent.mqtt_handler.client.subscribe(new_tag_data["tag_name"])
            success, message = self.db.edit_tag(self.project_name, row, new_tag_data)
            if success:
                self.update_table()
            else:
                QMessageBox.warning(self.parent, "Error", message)

    def delete_tag(self, row):
        reply = QMessageBox.question(self.parent, "Confirm Delete", "Are you sure you want to delete this tag?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            tags_data = list(self.db.tags_collection.find({"project_name": self.project_name}))
            tag = tags_data[row]
            if self.parent.mqtt_handler:
                self.parent.mqtt_handler.client.unsubscribe(tag["tag_name"])
            success, message = self.db.delete_tag(self.project_name, row)
            if success:
                self.update_table()
            else:
                QMessageBox.warning(self.parent, "Error", message)

    def on_data_received(self, tag_name, values):
        self.update_table()

    def get_widget(self):
        return self.widget