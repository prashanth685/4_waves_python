import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSplitter,
                             QToolBar, QAction, QTreeWidget, QTreeWidgetItem, QInputDialog, QMessageBox,QSizePolicy,QApplication)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon
import os
from mqtthandler import MQTTHandler
from features.create_tags import CreateTagsFeature
from features.tabular_view import TabularViewFeature
from features.time_view import TimeViewFeature
from features.fft_view import FFTViewFeature
from features.waterfall import WaterfallFeature
from features.orbit import OrbitFeature
from features.trend_view import TrendViewFeature
from features.multi_trend import MultiTrendFeature
from features.bode_plot import BodePlotFeature
from features.history_plot import HistoryPlotFeature
from features.time_report import TimeReportFeature
from features.report import ReportFeature
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DashboardWindow(QWidget):
    def __init__(self, db, email):
        super().__init__()
        self.db = db
        self.email = email
        self.current_project = None
        self.current_feature = None
        self.mqtt_handler = None
        self.feature_instances = {}
        self.timer = QTimer(self)
        
        self.initUI()
        self.setup_mqtt()

    def setup_mqtt(self):
        if self.current_project:
            if self.mqtt_handler:
                self.mqtt_handler.stop()
            self.mqtt_handler = MQTTHandler(self.db, self.current_project)
            self.mqtt_handler.data_received.connect(self.on_data_received)
            self.mqtt_handler.start()
            logging.info(f"MQTT setup for project: {self.current_project}")

    def on_data_received(self, tag_name, values):
        if self.current_feature and self.current_project:
            feature_instance = self.feature_instances.get(self.current_feature)
            if feature_instance:
                feature_instance.on_data_received(tag_name, values)

    def initUI(self):
        self.setWindowTitle('Sarayu Dashboard')
        self.showMaximized()  # Maximize the window when it's shown

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.file_bar = QToolBar("File")
        self.file_bar.setStyleSheet("""
            QToolBar { background-color: #c3cb9b; border: none; padding: 5px; spacing: 10px; }
            QToolBar QToolButton { font-size: 16px; font-weight: bold; padding: 5px; }
            QToolBar QToolButton:hover { background-color: #a9b37e; }
        """)
        self.file_bar.setFixedHeight(40)
        self.file_bar.setMovable(False)
        self.file_bar.setFloatable(False)

        actions = [
            ("Home", self.display_dashboard),
            ("New", self.create_project),
            ("Open", self.open_project_dialog),
            ("Save", self.save_action),
            ("Settings", self.settings_action),
            ("Refresh", self.refresh_action),
            ("Exit", self.close)
        ]
        for text, func in actions:
            action = QAction(text, self)
            action.triggered.connect(func)
            self.file_bar.addAction(action)
        main_layout.addWidget(self.file_bar)

        self.toolbar = QToolBar("Navigation")
        self.update_toolbar()
        main_layout.addWidget(self.toolbar)

        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Projects")
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #2c3e50; color: white; border: none;}
            QTreeWidget::item { padding: 5px; text-align: center; }
            QTreeWidget::item:hover { background-color: #4a6077; }
            QTreeWidget::item:selected { background-color: #1a73e8; }
        """)
        self.tree.setFixedWidth(300)
        self.tree.itemClicked.connect(self.on_tree_item_clicked)
        main_splitter.addWidget(self.tree)

        content_container = QWidget()
        self.content_layout = QVBoxLayout()
        content_container.setLayout(self.content_layout)
        content_container.setStyleSheet("background-color: #34495e;")
        main_splitter.addWidget(content_container)
        main_splitter.setSizes([300, 900])
        main_splitter.setHandleWidth(0)

        self.load_projects()
        self.display_dashboard()

    def update_toolbar(self):
        self.toolbar.clear()
        self.toolbar.setStyleSheet("""
            QToolBar { background-color: #83afa5; border: none; padding: 5px; spacing: 5px; margin: 0; }
            QToolBar::separator { width: 1px; margin: 0; }
            QToolButton { border: none; padding: 8px; border: 1px solid black; margin: 0; border-radius: 5px; background-color: #1e2937; }
            QToolButton:hover { background-color: #e0e0e0; }
            QToolButton:pressed { background-color: #d0d0d0; }
            QToolButton:focus { outline: none; border: 1px solid #0078d7; }
        """)
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)

        def add_action(text, icon_path, callback, tooltip=None):
            icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
            action = QAction(icon, text, self)
            action.triggered.connect(callback)
            if tooltip:
                action.setToolTip(tooltip)
            self.toolbar.addAction(action)

        add_action("New", "icons/new.png", self.create_project, "Create a New Project")
        add_action("Open", "icons/open.png", self.open_project_dialog, "Open an Existing Project")
        add_action("", "icons/save.png", self.save_action, "Save Project")
        add_action("", "icons/refresh.png", self.refresh_action, "Refresh View")
        add_action("", "icons/edit.png", self.edit_project_dialog, "Edit Project Name")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        add_action("Settings", "icons/settings.png", self.settings_action, "Settings")

    def close_project(self):
        if self.mqtt_handler:
            self.mqtt_handler.stop()
            self.mqtt_handler = None
        self.current_project = None
        self.current_feature = None
        self.timer.stop()
        self.update_toolbar()
        self.display_dashboard()

    def open_project_dialog(self):
        project_name, ok = QInputDialog.getItem(self, "Open Project", "Select a project:", self.db.projects, 0, False)
        if ok and project_name:
            self.current_project = project_name
            self.current_feature = None
            self.timer.stop()
            self.update_toolbar()
            self.setup_mqtt()
            self.display_feature_content("Create Tags", project_name)
        

    def display_dashboard(self):
        if self.mqtt_handler:
            self.mqtt_handler.stop()
            self.mqtt_handler = None
        self.current_project = None
        self.current_feature = None
        self.timer.stop()
        self.update_toolbar()

        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        header = QLabel("Welcome to Sarayu Application")
        header.setStyleSheet("color: white; font-size: 24px; font-weight: bold; padding: 10px;")
        self.content_layout.addWidget(header, alignment=Qt.AlignCenter)

    def load_projects(self):
        self.db.load_projects()
        self.tree.clear()
        for project_name in self.db.projects:
            self.add_project_to_tree(project_name)

    def add_project_to_tree(self, project_name):
        project_item = QTreeWidgetItem(self.tree)
        project_item.setText(0, project_name)
        project_item.setIcon(0, QIcon("icons/folder.png") if os.path.exists("icons/folder.png") else QIcon())
        project_item.setData(0, Qt.UserRole, {"type": "project", "name": project_name})

        features = [
            ("Create Tags", "icons/tag.png"),
            ("Time View", "icons/time.png"),
            ("Tabular View", "icons/table.png"),
            ("FFT", "icons/fft.png"),
            ("Waterfall", "icons/waterfall.png"),
            ("Orbit", "icons/orbit.png"),
            ("Trend View", "icons/trend.png"),
            ("Multiple Trend View", "icons/multitrend.png"),
            ("Bode Plot", "icons/bode.png"),
            ("History Plot", "icons/history.png"),
            ("Time Report", "icons/report.png"),
            ("Report", "icons/report.png")
        ]

        for feature, icon_path in features:
            feature_item = QTreeWidgetItem(project_item)
            feature_item.setText(0, feature)
            feature_item.setIcon(0, QIcon(icon_path) if os.path.exists(icon_path) else QIcon())
            feature_item.setData(0, Qt.UserRole, {"type": "feature", "name": feature, "project": project_name})

    def on_tree_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data["type"] == "project":
            self.current_project = data["name"]
            self.current_feature = None
            self.setup_mqtt()
            self.display_dashboard()
        elif data["type"] == "feature":
            self.current_project = data["project"]
            self.current_feature = data["name"]
            self.setup_mqtt()
            self.display_feature_content(data["name"], data["project"])

    def create_project(self):
        project_name, ok = QInputDialog.getText(self, "Create Project", "Enter project name:")
        self.setStyleSheet("""
                        QMessageBox {
                background-color: #34495e;
                color: white;
                font-size: 14px;
                border-radius: 8px;
            }
            QMessageBox QLabel {
                font-size: 16px;
                color: white;
                padding: 20px 20px;
            }
            QMessageBox QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 12px 20px;
                font-size: 14px;
                min-width: 100px;
            }
            QMessageBox QPushButton:hover {
                background-color: #2980b9;
            }
            QMessageBox QPushButton:pressed {
                background-color: #1abc9c;
            }
            QMessageBox QPushButton:focus {
                border: none;
            }
                        QInputDialog {
                background-color: #34495e;
                color: white;
                font-size: 14px;
                border-radius: 8px;
            }
            QInputDialog QLabel {
                font-size: 16px;
                color: white;
                padding: 20px 20px;
            }
            QInputDialog QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 12px 20px;
                font-size: 14px;
                min-width: 100px;
            }
            QInputDialog QPushButton:hover {
                background-color: #2980b9;
            }
            QInputDialog QPushButton:pressed {
                background-color: #1abc9c;
            }
            QInputDialog QPushButton:focus {
                border: none;
            }
            """)
        if ok and project_name:
            success, message = self.db.create_project(project_name)
            if success:
                self.add_project_to_tree(project_name)
                QMessageBox.information(self, "Success", message)
                self.current_project = project_name
                self.current_feature = None
                self.update_toolbar()
                self.setup_mqtt()
                self.display_feature_content("Create Tags", project_name)
            else:
                QMessageBox.warning(self, "Error", message)

    def edit_project_dialog(self):
        if not self.current_project:
            QMessageBox.warning(self, "Error", "No project selected to edit!")
            return

        old_project_name = self.current_project
        new_project_name, ok = QInputDialog.getText(self, "Edit Project", "Enter new project name:", text=old_project_name)
        if not ok or not new_project_name or new_project_name == old_project_name:
            return

        success, message = self.db.edit_project(old_project_name, new_project_name)
        if success:
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                if item.text(0) == old_project_name:
                    item.setText(0, new_project_name)
                    item.setData(0, Qt.UserRole, {"type": "project", "name": new_project_name})
                    for j in range(item.childCount()):
                        child = item.child(j)
                        child_data = child.data(0, Qt.UserRole)
                        child_data["project"] = new_project_name
                        child.setData(0, Qt.UserRole, child_data)
                    break
            
            self.current_project = new_project_name
            self.setup_mqtt()
            self.update_toolbar()
            self.display_feature_content(self.current_feature or "Create Tags", self.current_project)
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Error", message)

    def delete_project(self, project_name):
        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete {project_name}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            success, message = self.db.delete_project(project_name)
            if success:
                for i in range(self.tree.topLevelItemCount()):
                    if self.tree.topLevelItem(i).text(0) == project_name:
                        self.tree.takeTopLevelItem(i)
                        break
                if self.current_project == project_name:
                    self.close_project()
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.warning(self, "Error", message)

    def display_feature_content(self, feature_name, project_name):
        self.current_project = project_name
        self.current_feature = feature_name
        self.update_toolbar()
        self.timer.stop()

        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        feature_classes = {
            "Create Tags": CreateTagsFeature,
            "Tabular View": TabularViewFeature,
            "Time View": TimeViewFeature,
            # "FFT": FFTViewFeature,
            # "Waterfall": WaterfallFeature,
            # "Orbit": OrbitFeature,
            # "Trend View": TrendViewFeature,
            # "Multiple Trend View": MultiTrendFeature,
            # "Bode Plot": BodePlotFeature,
            # "History Plot": HistoryPlotFeature,
            "Time Report": TimeReportFeature,
            # "Report": ReportFeature
        }

        if feature_name in feature_classes:
            feature_instance = feature_classes[feature_name](self, self.db, project_name)
            self.feature_instances[feature_name] = feature_instance
            self.content_layout.addWidget(feature_instance.get_widget())

    def save_action(self):
        if self.current_project and self.db.get_project_data(self.current_project):
            QMessageBox.information(self, "Save", f"Data for project '{self.current_project}' saved successfully!")
        else:
            QMessageBox.warning(self, "Save Error", "No project selected to save!")

    def refresh_action(self):
        if self.current_project and self.current_feature:
            self.display_feature_content(self.current_feature, self.current_project)
            QMessageBox.information(self, "Refresh", f"Refreshed view for '{self.current_feature}'!")
        else:
            self.display_dashboard()
            QMessageBox.information(self, "Refresh", "Refreshed dashboard view!")

    def settings_action(self):
        QMessageBox.information(self, "Settings", "Settings functionality not implemented yet.")

    def closeEvent(self, event):
        self.timer.stop()
        if self.mqtt_handler:
            self.mqtt_handler.stop()
        self.db.close_connection()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    from database import Database
    db = Database(email="user@example.com")
    window = DashboardWindow(db=db, email="user@example.com")
    window.show()
    sys.exit(app.exec_())