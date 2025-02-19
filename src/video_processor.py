import ffmpeg
import os
from typing import Tuple

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.output_dir = os.path.join(os.path.dirname(video_path), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def split_video_audio(self) -> Tuple[str, str]:
        """Разделяет видео на видеопоток без звука и аудиопоток"""
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        video_output = os.path.join(self.output_dir, f"{base_name}_video_only.mp4")
        audio_output = os.path.join(self.output_dir, f"{base_name}_audio.wav")
        
        # Извлекаем аудио
        stream = ffmpeg.input(self.video_path)
        audio = stream.audio
        stream = ffmpeg.output(audio, audio_output)
        ffmpeg.run(stream, overwrite_output=True)
        
        # Копируем видео без звука
        stream = ffmpeg.input(self.video_path)
        video = stream.video
        stream = ffmpeg.output(video, video_output, acodec='none')
        ffmpeg.run(stream, overwrite_output=True)
        
        return video_output, audio_output
        
    def combine_video_audio(self, video_path: str, audio_path: str, output_path: str):
        """Объединяет видео и аудио в финальный файл"""
        stream = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        stream = ffmpeg.output(stream.video, audio.audio, output_path)
        ffmpeg.run(stream, overwrite_output=True)
        
    def combine_video_audio_with_background(self, video_path: str, speech_path: str, background_path: str, output_path: str):
        """Объединяет видео с новой озвучкой и фоновыми звуками"""
        # Загружаем видео
        video = ffmpeg.input(video_path).video
        
        # Загружаем озвучку и фон
        speech = ffmpeg.input(speech_path).audio
        background = ffmpeg.input(background_path).audio
        
        # Микшируем аудио дорожки
        mixed_audio = ffmpeg.filter([speech, background], 'amix', inputs=2, duration='longest')
        
        # Соединяем видео и смикшированное аудио
        stream = ffmpeg.output(video, mixed_audio, output_path)
        ffmpeg.run(stream, overwrite_output=True) 