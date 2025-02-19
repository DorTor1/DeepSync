import sys
import os
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QFileDialog, QLabel, QProgressBar,
                            QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import signal

# Создаем handler для отправки логов в GUI
class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)

class ProcessingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self._is_running = True
        
    def run(self):
        try:
            if not self._is_running:
                return
                
            # Импортируем процессоры только когда они нужны
            self.progress.emit("Загрузка компонентов...")
            
            self.progress.emit("Импорт VideoProcessor...")
            from video_processor import VideoProcessor
            
            if not self._is_running:
                return
                
            self.progress.emit("Импорт AudioProcessor...")
            from audio_processor import AudioProcessor
            
            if not self._is_running:
                return
                
            self.progress.emit("Импорт TTSProcessor...")
            from tts_processor import TTSProcessor
            
            if not self._is_running:
                return
            
            # Инициализация процессоров
            self.progress.emit("Инициализация VideoProcessor...")
            video_proc = VideoProcessor(self.video_path)
            
            if not self._is_running:
                return
                
            self.progress.emit("Инициализация AudioProcessor...")
            audio_proc = AudioProcessor()
            
            if not self._is_running:
                return
                
            self.progress.emit("Инициализация TTSProcessor...")
            tts_proc = TTSProcessor()
            
            if not self._is_running:
                return
            
            # Разделение видео и аудио
            self.progress.emit("Разделение видео и аудио...")
            video_only, audio = video_proc.split_video_audio()
            
            if not self._is_running:
                return
            
            # Разделение голоса и фона
            self.progress.emit("Отделение голоса от фоновой музыки...")
            vocals, background = audio_proc.separate_voice_background(audio)
            
            if not self._is_running:
                return
            
            # Транскрибация
            self.progress.emit("Транскрибация аудио...")
            text = audio_proc.transcribe_audio(vocals)
            
            if not self._is_running:
                return
            
            # Перевод
            self.progress.emit("Перевод текста...")
            translated_text = audio_proc.translate_text(text)
            
            if not self._is_running:
                return
            
            # Клонирование голоса и генерация речи
            self.progress.emit("Клонирование голоса и генерация речи...")
            output_dir = os.path.dirname(self.video_path)
            tts_output = os.path.join(output_dir, "generated_speech.wav")
            # Используем выделенный голос как референс
            tts_proc.clone_voice(vocals, "voice_model")
            tts_proc.generate_speech(translated_text, tts_output)
            
            if not self._is_running:
                return
            
            # Финальная сборка
            self.progress.emit("Сборка финального видео...")
            final_output = os.path.join(output_dir, "final_output.mp4")
            # Соединяем видео с новой озвучкой и фоновыми звуками
            video_proc.combine_video_audio_with_background(video_only, tts_output, background, final_output)
            
            if self._is_running:
                self.progress.emit("Готово!")
                self.finished.emit()
            
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
                
    def stop(self):
        self._is_running = False

class VideoDubbingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Дубляж видео")
        self.setGeometry(100, 100, 1000, 800)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем вертикальный layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Кнопка выбора видео
        self.select_video_btn = QPushButton("Выбрать видео")
        self.select_video_btn.clicked.connect(self.select_video)
        layout.addWidget(self.select_video_btn)
        
        # Метка для отображения выбранного видео файла
        self.video_label = QLabel("Видео файл не выбран")
        layout.addWidget(self.video_label)
        
        # Кнопка начала обработки
        self.process_btn = QPushButton("Начать обработку")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        # Кнопка отмены
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        layout.addWidget(self.cancel_btn)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Статус обработки
        self.status_label = QLabel("Статус: Ожидание")
        layout.addWidget(self.status_label)
        
        # Область для логов
        self.log_area = QTextEdit()
        self.log_area.setMinimumHeight(400)
        layout.addWidget(self.log_area)
        
        # Настраиваем логирование
        log_handler = QTextEditLogger(self.log_area)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        # Инициализация переменных
        self.video_path = None
        self.processing_thread = None
        
        # Обработка Ctrl+C
        signal.signal(signal.SIGINT, self.handle_sigint)
        
    def handle_sigint(self, signum, frame):
        self.cancel_processing()
        
    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео файл",
            "",
            "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_name:
            self.video_path = file_name
            self.video_label.setText(f"Выбран видео файл: {file_name}")
            self.check_ready()
            
    def check_ready(self):
        self.process_btn.setEnabled(self.video_path is not None)
            
    def process_video(self):
        self.process_btn.setEnabled(False)
        self.select_video_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        self.processing_thread = ProcessingThread(self.video_path)
        self.processing_thread.progress.connect(self.update_status)
        self.processing_thread.error.connect(self.handle_error)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
        
    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                'Подтверждение',
                'Вы уверены, что хотите отменить обработку?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.processing_thread.wait()
                self.status_label.setText("Статус: Отменено пользователем")
                self.processing_finished()
        
    def update_status(self, message):
        self.status_label.setText(f"Статус: {message}")
        logging.info(message)
        
    def handle_error(self, error_message):
        self.status_label.setText(f"Ошибка: {error_message}")
        logging.error(error_message)
        self.processing_finished()
        
    def processing_finished(self):
        self.process_btn.setEnabled(True)
        self.select_video_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    window = VideoDubbingApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 