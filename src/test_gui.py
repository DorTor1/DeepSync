import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тестовое окно")
        self.setGeometry(100, 100, 400, 300)
        
        button = QPushButton("Тестовая кнопка", self)
        button.setGeometry(150, 120, 100, 30)

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 