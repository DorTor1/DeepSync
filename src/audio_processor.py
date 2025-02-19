import torch
import torchaudio
import logging
import sys
import os
from pathlib import Path
from demucs.pretrained import get_model
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                         M2M100ForConditionalGeneration, M2M100Tokenizer)
from huggingface_hub import snapshot_download
import shutil
from tqdm import tqdm

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    CACHE_DIR = os.path.expanduser("~/.cache/huggingface/")
    MODELS = {
        "whisper-base": "openai/whisper-base",
        "m2m100": "facebook/m2m100_418M"
    }

    def __init__(self):
        logger.info("Начинаем инициализацию AudioProcessor...")
        
        # Устанавливаем устройство
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
        
        # Проверяем наличие кэш-директории
        logger.info(f"Проверка кэш-директории: {self.CACHE_DIR}")
        if not os.path.exists(self.CACHE_DIR):
            logger.info("Кэш-директория не найдена, создаем...")
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # Загружаем все модели
        logger.info("Начинаем загрузку всех моделей...")
        self._download_all_models()
        
        # Инициализируем модели
        logger.info("Инициализация моделей...")
        self._initialize_models()
        
        logger.info("Инициализация AudioProcessor завершена успешно")

    def _download_all_models(self):
        """Загружает все необходимые модели"""
        for model_name, model_path in self.MODELS.items():
            cache_path = os.path.join(self.CACHE_DIR, "models--" + model_path.replace("/", "--"))
            config_path = os.path.join(cache_path, "snapshots")
            
            # Проверяем наличие директории snapshots и config.json
            need_download = (
                not os.path.exists(cache_path) or 
                not os.path.exists(config_path) or 
                not any(f.endswith('config.json') for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f)))
            )
            
            if need_download:
                logger.info(f"Загрузка модели {model_name} ({model_path}) из Hugging Face...")
                try:
                    logger.info(f"Создание директории для модели: {cache_path}")
                    os.makedirs(cache_path, exist_ok=True)
                    
                    logger.info(f"Начинаем загрузку файлов модели {model_name}...")
                    snapshot_download(
                        repo_id=model_path,
                        local_files_only=False,
                        local_dir=cache_path,
                        tqdm_class=tqdm,
                        force_download=True  # Принудительно загружаем, если нет config.json
                    )
                    logger.info(f"Модель {model_name} успешно загружена в {cache_path}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {model_name}: {str(e)}")
                    if os.path.exists(cache_path):
                        logger.info(f"Удаление поврежденной директории: {cache_path}")
                        shutil.rmtree(cache_path, ignore_errors=True)
                    raise
            else:
                logger.info(f"Модель {model_name} уже есть в кэше: {cache_path}")
                try:
                    snapshot_path = os.path.join(cache_path, "snapshots")
                    files = [f for f in os.listdir(snapshot_path) if f.endswith('config.json')]
                    logger.info(f"Найдены файлы конфигурации: {', '.join(files)}")
                except Exception as e:
                    logger.warning(f"Проблема с кэшем модели {model_name}: {str(e)}")
                    logger.info("Попытка повторной загрузки...")
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path, ignore_errors=True)
                    self._download_model(model_name, model_path, force=True)

    def _download_model(self, model_name: str, model_path: str, force: bool = False):
        """Загружает конкретную модель"""
        cache_path = os.path.join(self.CACHE_DIR, "models--" + model_path.replace("/", "--"))
        
        logger.info(f"Загрузка модели {model_name} ({model_path}) из Hugging Face...")
        try:
            logger.info(f"Создание директории для модели: {cache_path}")
            os.makedirs(cache_path, exist_ok=True)
            
            logger.info(f"Начинаем загрузку файлов модели {model_name}...")
            snapshot_download(
                repo_id=model_path,
                local_files_only=False,
                local_dir=cache_path,
                tqdm_class=tqdm,
                force_download=force
            )
            logger.info(f"Модель {model_name} успешно загружена в {cache_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке {model_name}: {str(e)}")
            if os.path.exists(cache_path):
                logger.info(f"Удаление поврежденной директории: {cache_path}")
                shutil.rmtree(cache_path, ignore_errors=True)
            raise

    def _initialize_models(self):
        """Инициализирует все модели"""
        # Инициализация Demucs
        logger.info("Инициализация Demucs...")
        try:
            self.demucs_model = get_model('htdemucs')
            self.demucs_model.eval()
            self.demucs_model.to(self.device)
            logger.info("Demucs инициализирован успешно")
        except Exception as e:
            logger.error(f"Ошибка при инициализации Demucs: {str(e)}")
            raise

        # Инициализация Whisper
        logger.info("Инициализация Whisper...")
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.MODELS["whisper-base"],
                local_files_only=True,
                cache_dir=self.CACHE_DIR
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.MODELS["whisper-base"],
                local_files_only=True,
                cache_dir=self.CACHE_DIR
            ).to(self.device)
            logger.info("Whisper инициализирован успешно")
        except Exception as e:
            logger.error(f"Ошибка при инициализации Whisper: {str(e)}")
            raise

        # Инициализация переводчика
        logger.info("Инициализация переводчика...")
        try:
            self.translator_model = M2M100ForConditionalGeneration.from_pretrained(
                self.MODELS["m2m100"],
                local_files_only=True,
                cache_dir=self.CACHE_DIR
            ).to(self.device)
            self.translator_tokenizer = M2M100Tokenizer.from_pretrained(
                self.MODELS["m2m100"],
                local_files_only=True,
                cache_dir=self.CACHE_DIR
            )
            logger.info("Переводчик инициализирован успешно")
        except Exception as e:
            logger.error(f"Ошибка при инициализации переводчика: {str(e)}")
            raise

    def separate_voice_background(self, audio_path: str) -> tuple[str, str]:
        """Разделяет аудио на голос и фоновые звуки"""
        logger.info(f"Начинаем разделение аудио: {audio_path}")
        
        # Загружаем аудио
        logger.info("Загрузка аудио файла...")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        logger.info(f"Аудио загружено: {waveform.shape}, sample_rate: {sample_rate}")
        
        # Применяем Demucs
        logger.info("Применяем Demucs для разделения...")
        sources = self.demucs_model.separate(waveform)
        logger.info("Разделение выполнено успешно")
        
        # Получаем директорию для сохранения
        output_dir = os.path.dirname(audio_path)
        vocals_path = os.path.join(output_dir, "vocals.wav")
        background_path = os.path.join(output_dir, "background.wav")
        
        # Сохраняем результаты
        logger.info("Сохранение результатов разделения...")
        torchaudio.save(vocals_path, sources[0].cpu(), sample_rate)
        torchaudio.save(background_path, sources[1].cpu(), sample_rate)
        logger.info("Результаты сохранены успешно")
        
        return vocals_path, background_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибирует аудио в текст"""
        logger.info(f"Начинаем транскрибацию: {audio_path}")
        
        # Загружаем аудио
        logger.info("Загрузка аудио для транскрибации...")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        logger.info("Аудио загружено успешно")
        
        # Подготавливаем входные данные
        logger.info("Подготовка входных данных для Whisper...")
        input_features = self.whisper_processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        logger.info("Входные данные подготовлены")
        
        # Получаем транскрипцию
        logger.info("Выполнение транскрибации...")
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Транскрибация завершена: {transcription[:50]}...")
        
        return transcription
    
    def translate_text(self, text: str) -> str:
        """Переводит текст с английского на русский"""
        logger.info("Начинаем перевод текста...")
        
        # Токенизируем текст
        logger.info("Токенизация текста...")
        encoded = self.translator_tokenizer(
            text, 
            return_tensors="pt",
            src_lang="en",
            padding=True
        ).to(self.device)
        logger.info("Токенизация завершена")
        
        # Получаем перевод
        logger.info("Выполнение перевода...")
        translated = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.get_lang_id("ru")
        )
        translation = self.translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        logger.info(f"Перевод завершен: {translation[:50]}...")
        
        return translation 