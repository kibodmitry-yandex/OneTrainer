import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, TextIteratorStreamer
from langdetect import detect
import logging
import pynvml
from threading import Thread

logger = logging.getLogger(__name__)

# Global flag to prevent multiple GPU loads
gpu_in_use = False

class Translator:
    def __init__(self, force_cpu=False):
        global gpu_in_use
        self.model_name = 'facebook/m2m100_1.2B'
        
        # Check GPU availability and usage
        gpu_available = torch.cuda.is_available()
        gpu_busy = False
        if gpu_available and not force_cpu:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                if info.gpu > 0:  # GPU utilization > 0%
                    gpu_busy = True
                    logger.warning("GPU is busy (utilization > 0%), switching to CPU")
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Failed to check GPU utilization: {e}, assuming GPU available")
        
        if gpu_available and not gpu_busy and not gpu_in_use and not force_cpu:
            self.device = 'cuda'
            gpu_in_use = True
        else:
            self.device = 'cpu'
        
        logger.info(f"Loading M2M100 model on {self.device}")
        
        # Try to load with quantization
        try:
            from transformers import BitsAndBytesConfig
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            if self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.model = M2M100ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                logger.info("Model loaded with 4-bit quantization on GPU")
            else:
                self.model = M2M100ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
                logger.info("Model loaded on CPU")
        except Exception as e:
            logger.warning(f"Failed to load with 4-bit: {e}, trying 8-bit")
            try:
                if self.device == 'cuda':
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = M2M100ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16
                    )
                    logger.info("Model loaded with 8-bit quantization on GPU")
                else:
                    self.model = M2M100ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32
                    ).to(self.device)
                    logger.info("Model loaded on CPU")
            except Exception as e:
                logger.warning(f"Failed to load with 8-bit: {e}, loading without quantization")
                self.model = M2M100ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                logger.info("Model loaded without quantization")

    def __del__(self):
        global gpu_in_use
        if self.device == 'cuda':
            gpu_in_use = False

    def detect_language(self, text):
        try:
            lang = detect(text)
            # For short texts, langdetect may be unreliable, default to Russian to attempt translation
            if len(text.split()) < 3:
                return 'ru'
            return lang
        except Exception:
            return 'ru'  # Default to Russian if detection fails

    def translate_to_english(self, text, stream_callback=None):
        if not text.strip():
            return text
        
        # Detect language
        lang = self.detect_language(text)
        if lang == 'en':
            return text
        
        # Check if language is supported by M2M100
        supported_langs = set(self.tokenizer.lang_code_to_token.keys())
        if lang not in supported_langs:
            logger.warning(f"Language '{lang}' not supported by M2M100, skipping translation")
            return text
        
        # Chunk the text if too long - split by sentences more intelligently
        max_length = 512  # M2M100 max input length
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        translated_parts = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            self.tokenizer.src_lang = lang
            encoded = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
            if self.device == 'cuda':
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            
            generation_kwargs = {
                **encoded,
                "forced_bos_token_id": self.tokenizer.get_lang_id("en"),
                "max_length": max_length * 2,
                "streamer": streamer,
                "do_sample": False,
                "num_beams": 1,
                "early_stopping": False,
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            translated = ""
            for new_text in streamer:
                translated += new_text
                if stream_callback:
                    stream_callback(new_text)
                else:
                    print(new_text, end='', flush=True)
            
            thread.join()
            translated_parts.append(translated)
            if i < len(sentences) - 1:
                if stream_callback:
                    stream_callback(' ')
                else:
                    print(' ', end='', flush=True)
        
        print()  # Final newline
        return ' '.join(translated_parts)

    def translate(self, text, stream_callback=None):
        # Always attempt to translate to English
        return self.translate_to_english(text, stream_callback)


if __name__ == "__main__":
    # Test the translator
    import sys
    force_cpu = '--cpu' in sys.argv
    translator = Translator(force_cpu=force_cpu)
    
    # Long Russian text (~150 tokens/words)
    long_text = """
    В современном мире искусственный интеллект играет все более важную роль в различных сферах жизни. 
    От простых приложений, таких как распознавание речи и изображений, до сложных систем анализа данных и предсказания трендов. 
    Развитие ИИ открывает новые возможности для бизнеса, науки и общества в целом. 
    Однако с ростом возможностей возникают и этические вопросы, связанные с приватностью данных, безопасностью и справедливостью алгоритмов. 
    Важно развивать ИИ таким образом, чтобы он служил на благо человечества, а не наоборот. 
    В области машинного обучения используются различные методы, включая глубокое обучение, нейронные сети и обработку естественного языка. 
    Эти технологии позволяют создавать системы, способные понимать и генерировать человеческий язык, переводить между языками и даже создавать контент. 
    Будущее ИИ обещает быть захватывающим, но требует осторожного подхода к его внедрению и использованию. 
    Мы должны стремиться к тому, чтобы технологии помогали решать глобальные проблемы, такие как изменение климата, здравоохранение и образование. 
    Только так мы сможем построить более справедливое и процветающее общество для будущих поколений.
    """
    
    print(f"Original text length: {len(long_text.split())} words")
    lang = translator.detect_language(long_text)
    print(f"Detected language: {lang}")
    translated = translator.translate(long_text)
    print(f"Translated text length: {len(translated.split())} words")
    print("Translated text:")
    print(translated)