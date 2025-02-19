import torch
import torchaudio
from TTS.api import TTS
import os

class TTSProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Используем Coqui TTS с моделью XTTS v2
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(self.device)
        
    def clone_voice(self, reference_audio: str, save_path: str) -> str:
        """Клонирует голос из референсного аудио"""
        self.reference_audio = reference_audio
        return reference_audio
    
    def generate_speech(self, text: str, output_path: str) -> str:
        """Генерирует речь с клонированным голосом"""
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=self.reference_audio,
            language="ru"
        )
        return output_path 