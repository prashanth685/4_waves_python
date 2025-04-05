# from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
# from dashboard import DashboardWindow
# from database import Database


# class AuthWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Authentication")
#         self.setGeometry(200, 200, 300, 100)
#         layout = QVBoxLayout()
#         layout.addWidget(QLabel("Authentication stub - proceeding to dashboard"))
#         self.setLayout(layout)
#         # Simulate login success
#         self.db = Database(email="user@example.com")
#         self.dashboard = DashboardWindow(self.db, "user@example.com")
#         self.dashboard.show()
#         self.hide()



import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QLabel, QLineEdit,
                             QPushButton, QMessageBox, QFormLayout, QApplication,
                             QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import bcrypt
import os

# Assuming Database class is defined elsewhere
from database import Database


class AuthWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.client = None
        self.db = None
        self.users_collection = None
        self.initUI()
        self.initDB()

    def initDB(self):
        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            self.db = self.client["sarayu_db"]
            self.users_collection = self.db["users"]
            print("Connected to MongoDB successfully!")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            QMessageBox.critical(self, "Database Error", "Failed to connect to the database.")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('Sarayu Infotech Solutions Pvt. Ltd.')
        # self.setGeometry(200, 200, 1800, 1600)
        self.showFullScreen()  # Fullscreen mode
        # self.showMaximized()


        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        self.setLayout(main_layout)

        logo_label = QLabel(self)
        logo_path = "logo.png" if os.path.exists("logo.png") else "icons/placeholder.png"
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            print(f"Warning: Could not load logo at {logo_path}")
            pixmap = QPixmap("icons/placeholder.png")  # Fallback
        logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        logo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(logo_label)

        company_label = QLabel('Sarayu Infotech Solutions Pvt. Ltd.')
        company_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007bff;")
        main_layout.addWidget(company_label, alignment=Qt.AlignCenter)

        welcome_label = QLabel('Welcome')
        welcome_label.setStyleSheet("font-size: 20px; color: #007bff;")
        main_layout.addWidget(welcome_label, alignment=Qt.AlignCenter)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_login_tab(), "Login")
        self.tabs.addTab(self.create_signup_tab(), "Signup")
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                min-width: 100px;
                margin-left: 5px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #d3d3d3;
            }
            QTabBar::tab:!selected {
                background-color: #d0d0d0;
            }
        """)
        main_layout.addWidget(self.tabs)

        self.setStyleSheet("background-color: white; border-radius: 10px; padding: 20px;")

    def create_input_field(self, placeholder):
        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 12px;
                width: 250px;
                background-color: #e9ecef;
            }
            QLineEdit:focus {
                border: 2px solid #007bff;
            }
        """)
        return input_field

    def create_shadow_effect(self):
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setOffset(0, 0)
        shadow_effect.setBlurRadius(15)
        shadow_effect.setColor(Qt.gray)
        return shadow_effect

    def create_container(self, layout):
        container = QWidget()
        container.setLayout(layout)
        container.setStyleSheet("background-color: white; border-radius: 10px; padding: 20px;")
        shadow = self.create_shadow_effect()
        container.setGraphicsEffect(shadow)
        return container

    def create_login_tab(self):
        login_tab = QWidget()
        layout = QFormLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        login_logo_label = QLabel(self)
        login_logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(login_logo_label)


        email_label = QLabel('Email')
        email_label.setStyleSheet("font-size: 18px; color: #333;font:bold")
        self.login_email_input = self.create_input_field('Enter your email')
        self.login_email_input.setStyleSheet("font-size: 16px; color: #333;border:2px solid black")
        self.login_email_input.setText('sarayu@gmail.com')
        layout.addRow(email_label, self.login_email_input)

        password_label = QLabel('Password')
        password_label.setStyleSheet("font-size: 18px; color: #333;font:bold")
        self.login_password_input = self.create_input_field('Enter your password')
        self.login_password_input.setStyleSheet("font-size: 16px; color: #333;border:2px solid black")
        self.login_password_input.setText('12345678')
        self.login_password_input.setEchoMode(QLineEdit.Password)
        layout.addRow(password_label, self.login_password_input)

        signin_button = QPushButton('Sign in')
        signin_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 5px;
                padding: 15px;
                width: 250px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        signin_button.clicked.connect(self.login)
        layout.addRow(signin_button)

        forgot_link = QLabel('<a href="#" style="color: #007bff; text-decoration: none;font-size:16px">Forgot Password?</a>')
        forgot_link.setOpenExternalLinks(True)
        layout.addRow(forgot_link)

        return self.create_container(layout)

    def create_signup_tab(self):
        signup_tab = QWidget()
        layout = QFormLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        signup_logo_label = QLabel(self)
        # logo_path = "logo.png" if os.path.exists("logo.png") else "icons/placeholder.png"
        # pixmap = QPixmap(logo_path)
        # signup_logo_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
        signup_logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(signup_logo_label)

        email_label = QLabel('Email')
        email_label.setStyleSheet("font-size: 18px; color: #333;font:bold")
        self.signup_email_input = self.create_input_field('Enter your email')
        self.signup_email_input.setStyleSheet("font-size: 16px; color: #333;border:2px solid black")

        layout.addRow(email_label, self.signup_email_input)

        password_label = QLabel('Password')
        password_label.setStyleSheet("font-size: 18px; color: #333;font:bold")
        self.signup_password_input = self.create_input_field('Enter your password')
        self.signup_password_input.setStyleSheet("font-size: 16px; color: #333;border:2px solid black")

        self.signup_password_input.setEchoMode(QLineEdit.Password)
        layout.addRow(password_label, self.signup_password_input)

        confirm_password_label = QLabel('Confirm Password')
        confirm_password_label.setStyleSheet("font-size: 18px; color: #333;font:bold")
        self.signup_confirm_password_input = self.create_input_field('Confirm your password')
        self.signup_confirm_password_input.setStyleSheet("font-size: 16px; color: #333;border:2px solid black")
        self.signup_confirm_password_input.setEchoMode(QLineEdit.Password)
        layout.addRow(confirm_password_label, self.signup_confirm_password_input)

        signup_button = QPushButton('Sign Up')
        signup_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border-radius: 5px;
                padding: 15px;
                width: 250px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        signup_button.clicked.connect(self.signup)
        layout.addRow(signup_button)

        return self.create_container(layout)

    def login(self):
        email = self.login_email_input.text().strip()
        password = self.login_password_input.text().strip()

        if not email or not password:
            QMessageBox.warning(self, "Input Error", "Please enter both email and password.")
            return

        user = self.users_collection.find_one({"email": email})

        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            try:
                from dashboard import DashboardWindow
                db = Database(connection_string="mongodb://localhost:27017/", email=email)
                self.dashboard = DashboardWindow(db, email)
                self.dashboard.show()
                self.close()
            except Exception as e:
                print(f"Error opening Dashboard: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open dashboard: {e}")
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect email or password.")

    def signup(self):
        email = self.signup_email_input.text().strip()
        password = self.signup_password_input.text().strip()
        confirm_password = self.signup_confirm_password_input.text().strip()

        if not email or not password or not confirm_password:
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "Input Error", "Passwords do not match.")
            return

        if self.users_collection.find_one({"email": email}):
            QMessageBox.warning(self, "Signup Failed", "User with this email already exists. Please log in.")
            return

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_data = {"email": email, "password": hashed_password}
        try:
            self.users_collection.insert_one(user_data)
            email_safe = email.replace('@', '_').replace('.', '_')
            self.db[f"tagcreated_{email_safe}"].insert_one({"init": True})
            self.db[f"mqttmessage_{email_safe}"].insert_one({"init": True})
            QMessageBox.information(self, "Success", "Signup successful! Please log in.")
            self.tabs.setCurrentIndex(0)
            self.signup_email_input.clear()
            self.signup_password_input.clear()
            self.signup_confirm_password_input.clear()
        except Exception as e:
            print(f"Error inserting user: {e}")
            QMessageBox.critical(self, "Database Error", "Failed to sign up.")

    def closeEvent(self, event):
        if self.client:
            self.client.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AuthWindow()
    window.show()
    sys.exit(app.exec_())