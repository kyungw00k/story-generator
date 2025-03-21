import logging
from llama_cpp import Llama
import sqlite3
from datetime import datetime
import os
import json
import feedparser
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import re
import random
from abc import ABC, abstractmethod
from openai import OpenAI
import argparse
import platform
import torch
import time
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM 추상 기본 클래스
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        pass

# GGUF 모델 구현
class GGUFModel(BaseLLM):
    def __init__(self, 
                 model_path: str = None,
                 repo_id: str = "Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M",
                 filename: str = "llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf"
                #  repo_id: str = "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
                #  filename: str = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
                 ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러 추가
        fh = logging.FileHandler('gguf_debug.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        try:
            n_gpu_layers = self._get_optimal_gpu_layers()
            self.logger.info(f"Using {n_gpu_layers} GPU layers")
            
            if model_path:
                self.logger.info(f"Loading model from path: {model_path}")
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=32768,
                    n_batch=2048,
                    n_threads=8,
                    f16_kv=True,
                    offload_kqv=True,
                    use_mmap=True,
                    use_mlock=True
                )
            else:
                self.logger.info(f"Loading model from Hugging Face: {repo_id}/{filename}")
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=32768,
                    n_batch=2048,
                    n_threads=8,
                    f16_kv=True,
                    offload_kqv=True,
                    use_mmap=True,
                    use_mlock=True
                )
                
        except Exception as e:
            self.logger.error(f"Error initializing GGUF model: {e}")
            raise

    def _get_optimal_gpu_layers(self) -> int:
        """시스템 리소스를 확인하여 최적의 GPU 레이어 수 결정"""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                is_arm = platform.machine() == "arm64"
                if is_arm:
                    self.logger.info("Apple Silicon (M1/M2) detected")
                    return 32
                else:
                    self.logger.info("Intel Mac detected")
                    return 1
            
            elif torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                self.logger.info(f"CUDA GPU detected with {gpu_memory_gb:.2f}GB memory")
                
                if gpu_memory_gb > 24:
                    return 32
                elif gpu_memory_gb > 16:
                    return 24
                elif gpu_memory_gb > 12:
                    return 16
                elif gpu_memory_gb > 8:
                    return 8
                elif gpu_memory_gb > 4:
                    return 4
                else:
                    return 1
            
            else:
                self.logger.warning("No GPU detected. Using CPU only.")
                return 0

        except Exception as e:
            self.logger.warning(f"Error detecting GPU resources: {e}")
            self.logger.warning("Defaulting to minimal GPU layers")
            return 1

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        self.logger.debug("=== Generation Request ===")
        self.logger.debug(f"Max tokens: {max_tokens}")
        
        # JSON 응답을 위한 시스템 프롬프트 추가
        system_prompt = (
            "You are a helpful AI assistant that always responds in valid JSON format. "
            "If the user's request requires a JSON response, you will:\n"
            "1. Only output valid JSON\n"
            "2. Do not include any explanatory text before or after the JSON\n"
            "3. Ensure all JSON keys and values are properly quoted\n"
            "4. Use proper JSON syntax with curly braces, square brackets, and commas\n"
        )
        
        # JSON 응답을 위한 포맷 추가
        formatted_prompt = (
            f"System: {system_prompt}\n\n"
            f"User: {prompt}\n\n"
            "Assistant: "
        )
        
        self.logger.debug("=== Sending Prompt ===")
        self.logger.debug(formatted_prompt)

        max_retries = 3
        retry_delay = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Generation attempt {attempt + 1}/{max_retries}")
                response = self.llm.create_completion(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    stop=["User:", "\n\n"],
                    echo=False
                )
                
                result = response["choices"][0]["text"].strip()
                self.logger.debug("=== Received Response ===")
                self.logger.debug(result)
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error("All retry attempts failed")
                    self.logger.exception("Detailed error information:")
                    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}") from last_error

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        self.logger.debug("=== Generation Request ===")
        self.logger.debug(f"Max tokens: {max_tokens}")
        
        # JSON 응답을 위한 시스템 프롬프트 추가
        system_prompt = (
            "You are a helpful AI assistant that always responds in valid JSON format. "
            "If the user's request requires a JSON response, you will:\n"
            "1. Only output valid JSON\n"
            "2. Do not include any explanatory text before or after the JSON\n"
            "3. Ensure all JSON keys and values are properly quoted\n"
            "4. Use proper JSON syntax with curly braces, square brackets, and commas\n"
        )
        
        # JSON 응답을 위한 포맷 추가
        formatted_prompt = (
            f"System: {system_prompt}\n\n"
            f"User: {prompt}\n\n"
            "Assistant: "
        )
        
        self.logger.debug("=== Sending Prompt ===")
        self.logger.debug(formatted_prompt)

        try:
            self.logger.debug("Starting generation...")
            response = self.llm.create_completion(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                stop=["User:", "\n\n"],
                echo=False
            )
            
            result = response["choices"][0]["text"].strip()
            self.logger.debug("=== Received Response ===")
            self.logger.debug(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            self.logger.exception("Detailed error information:")
            raise

# OpenAI 모델 구현
class OpenAIModel(BaseLLM):
    def __init__(self, api_key: str = None, cache_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러 추가
        fh = logging.FileHandler('openai_debug.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Initialize LRU cache for responses
        from functools import lru_cache
        self._generate_cached = lru_cache(maxsize=cache_size)(self._generate_uncached)
        
        if api_key is None:
            key_file = 'openai.key'
            if os.path.exists(key_file):
                try:
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
                    self.logger.info("API key loaded from openai.key file")
                except Exception as e:
                    self.logger.warning(f"Error reading openai.key file: {e}")
            
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.logger.info("API key loaded from environment variable")
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it via argument, openai.key file, or environment variable.")
        
        self.client = OpenAI(api_key=api_key)

    def _generate_uncached(self, prompt: str, max_tokens: int) -> str:
        max_retries = 3
        retry_delay = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a creative story writer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=max_tokens,
                    top_p=0.9,
                )
                
                content = response.choices[0].message.content.strip()
                self.logger.debug(f"Received response from OpenAI:\n{content}")
                
                return content
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error("All retry attempts failed")
                    self.logger.exception("Detailed error information:")
                    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}") from last_error

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        self.logger.debug(f"Sending prompt to OpenAI:\n{prompt}")
        return self._generate_cached(prompt, max_tokens)

class TransformerModel(BaseLLM):
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Memory optimization settings
        self.max_memory = {
            0: "24GB",  # Adjust based on available GPU memory
            "cpu": "48GB"  # Adjust based on available RAM
        }
        
        # 파일 핸들러 추가
        fh = logging.FileHandler('transformer_debug.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            n_gpu_layers = self._get_optimal_gpu_layers()
            self.logger.info(f"Using {n_gpu_layers} GPU layers")
            
            # Download model from Hugging Face if needed
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_name, filename="model.gguf")
            
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=32768,
                n_batch=2048,
                n_threads=8,
                f16_kv=True,
                offload_kqv=True,
                use_mmap=True,
                use_mlock=True
            )
            
            self.logger.info("Model loaded successfully with optimized memory settings")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _get_optimal_gpu_layers(self) -> int:
        """시스템 리소스를 확인하여 최적의 GPU 레이어 수 결정"""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                is_arm = platform.machine() == "arm64"
                if is_arm:
                    self.logger.info("Apple Silicon (M1/M2) detected")
                    return 32
                else:
                    self.logger.info("Intel Mac detected")
                    return 1
            
            elif torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                self.logger.info(f"CUDA GPU detected with {gpu_memory_gb:.2f}GB memory")
                
                if gpu_memory_gb > 24:
                    return 32
                elif gpu_memory_gb > 16:
                    return 24
                elif gpu_memory_gb > 12:
                    return 16
                elif gpu_memory_gb > 8:
                    return 8
                elif gpu_memory_gb > 4:
                    return 4
                else:
                    return 1
            
            else:
                self.logger.warning("No GPU detected. Using CPU only.")
                return 0

        except Exception as e:
            self.logger.warning(f"Error detecting GPU resources: {e}")
            self.logger.warning("Defaulting to minimal GPU layers")
            return 1

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        self.logger.debug("=== Generation Request ===")
        self.logger.debug(f"Max tokens: {max_tokens}")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            self.logger.debug("Starting generation...")
            with torch.no_grad():  # Disable gradient calculation for inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear CUDA cache before generation
                
                response = self.pipe(
                    messages,
                    max_length=max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            result = response[0]["generated_text"]
            self.logger.debug("=== Received Response ===")
            self.logger.debug(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            self.logger.exception("Detailed error information:")
            raise

# LMStudio 모델 구현
class LMStudioModel(BaseLLM):
    def __init__(self, api_base: str = "http://localhost:1234/v1"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러 추가
        fh = logging.FileHandler('lmstudio_debug.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.api_base = api_base
        self.client = OpenAI(base_url=api_base, api_key="not-needed")
        
        # 서버 연결 테스트
        try:
            self.client.models.list()
            self.logger.info(f"Successfully connected to LMStudio server at {api_base}")
        except Exception as e:
            self.logger.error(f"Failed to connect to LMStudio server: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        self.logger.debug("=== Generation Request ===")
        self.logger.debug(f"Max tokens: {max_tokens}")
        
        max_retries = 3
        retry_delay = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Generation attempt {attempt + 1}/{max_retries}")
                
                # JSON 응답을 위한 시스템 프롬프트 추가
                system_prompt = (
                    "You are a helpful AI assistant that always responds in valid JSON format. "
                    "If the user's request requires a JSON response, you will:\n"
                    "1. Only output valid JSON\n"
                    "2. Do not include any explanatory text before or after the JSON\n"
                    "3. Ensure all JSON keys and values are properly quoted\n"
                    "4. Use proper JSON syntax with curly braces, square brackets, and commas\n"
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model="local-model",  # LMStudio는 이 모델명을 사용
                    messages=messages,
                    temperature=0.8,
                    max_tokens=10240,
                    top_p=0.9
                )
                
                result = response.choices[0].message.content.strip()
                
                # JSON 응답 검증
                try:
                    json.loads(result)  # JSON 형식 검증
                except json.JSONDecodeError:
                    # JSON 형식이 아닌 경우, JSON 부분만 추출
                    start_idx = result.find('{')
                    end_idx = result.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        result = result[start_idx:end_idx]
                    else:
                        raise ValueError("No valid JSON found in response")
                
                self.logger.debug("=== Received Response ===")
                self.logger.debug(result)
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error("All retry attempts failed")
                    self.logger.exception("Detailed error information:")
                    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}") from last_error

    def _generate_structure(self, trending_topics: List[Dict[str, Any]], analysis: Dict, max_retries: int = 3) -> Dict:
        """소설의 전체 구조 생성"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"\n구조 생성 시도 {attempt + 1}/{max_retries}")
                
                structure_prompt = (
                    "System: 당신은 소설 구조를 생성하는 AI 어시스턴트입니다. "
                    "다음 규칙을 엄격히 따라주세요:\n"
                    "1. 응답은 반드시 유효한 JSON이어야 합니다\n"
                    "2. 최상위 레벨에 다음 키들이 반드시 포함되어야 합니다:\n"
                    "   - 제목: 소설 제목 (문자열)\n"
                    "   - 주제: 이야기의 핵심 주제 (문자열)\n"
                    "   - 주인공: 주인공 정보 (객체)\n"
                    "     * 이름: 주인공 이름 (문자열)\n"
                    "     * 나이: 주인공 나이 (문자열)\n"
                    "     * 직업: 주인공 직업 (문자열)\n"
                    "     * 설명: 주인공 성격과 배경 (문자열)\n"
                    "   - 조연: 조연 인물 목록 (배열)\n"
                    "     * 각 조연은 다음 키를 가져야 함:\n"
                    "       - 이름: 조연 이름 (문자열)\n"
                    "       - 역할: 조연의 역할 (문자열)\n"
                    "       - 설명: 조연 설명 (문자열)\n"
                    "   - 챕터: 8개의 챕터 정보 (배열)\n"
                    "     * 각 챕터는 다음 키를 가져야 함:\n"
                    "       - 제목: 챕터 제목 (문자열)\n"
                    "       - 사건: 주요 사건 목록 (문자열 배열)\n"
                    "       - 갈등: 갈등 요소 목록 (문자열 배열)\n"
                    "       - 요약: 챕터 요약 (문자열)\n"
                    "3. 설명이나 부가 텍스트를 포함하지 마세요\n"
                    "4. 중괄호로 시작하고 끝내세요\n"
                    "5. 모든 키는 한글로 작성해주세요\n"
                    "6. 위에서 명시한 모든 필수 키가 반드시 포함되어야 합니다\n"
                    "7. 각 키의 데이터 타입이 정확히 일치해야 합니다\n\n"
                    "User: 다음 요소들을 바탕으로 한국적 정서가 담긴 소설 구조를 만들어주세요. "
                    "보편적인 주제를 탐구하되 한국의 문화적 맥락을 반영해주세요.\n\n"
                    f"주제: {analysis['주제']}\n"
                    f"부주제: {', '.join(analysis['부주제'])}\n\n"
                    "인물 유형:\n"
                    + '\n'.join([
                        f"- {char['유형']}: {char['배경_설정']}"
                        for char in analysis['인물_제안']
                    ]) + "\n\n"
                    + ("갈등 요소:\n"
                    + '\n'.join([
                        f"- {conflict['유형']}: {conflict['구체적_설정']}"
                        for conflict in analysis.get('갈등_요소', [])
                    ]) + "\n\n") if '갈등_요소' in analysis else ""
                    + f"배경: {analysis.get('배경_제안', {}).get('장소', '현대 한국')}\n"
                    + f"분위기: {analysis.get('배경_제안', {}).get('분위기', '현대적')}\n\n"
                    "요구사항:\n"
                    "1. 8개의 챕터로 구성하여 명확한 이야기 전개를 보여주세요\n"
                    "2. 허구의 인물과 배경을 사용하세요\n"
                    "3. 보편적 주제를 한국적 맥락에서 다루세요\n"
                    "4. 감정선을 진정성 있게 표현하세요\n"
                    "5. 설득력 있는 인물 성장을 보여주세요\n\n"
                    "다음 JSON 구조로 출력해주세요:\n"
                    "{\n"
                    '  "제목": "소설 제목",\n'
                    '  "주제": "이야기의 핵심 주제",\n'
                    '  "주인공": {\n'
                    '    "이름": "이름",\n'
                    '    "나이": "나이",\n'
                    '    "직업": "직업",\n'
                    '    "설명": "성격과 배경"\n'
                    '  },\n'
                    '  "조연": [{\n'
                    '    "이름": "이름",\n'
                    '    "역할": "역할",\n'
                    '    "설명": "설명"\n'
                    '  }],\n'
                    '  "챕터": [{\n'
                    '    "제목": "챕터 제목",\n'
                    '    "사건": ["주요 사건1", "주요 사건2"],\n'
                    '    "갈등": ["갈등1", "갈등2"],\n'
                    '    "요약": "챕터 요약"\n'
                    '  }]\n'
                    "}\n\n"
                    "Assistant: "
                )
                
                response = self.llm.generate(structure_prompt, max_tokens=10240)
                
                try:
                    # JSON 파싱
                    structure = json.loads(response)
                    
                    # 중첩된 구조 처리
                    if "소설_구조" in structure:
                        structure = structure["소설_구조"]
                    
                    # 필수 키 확인
                    required_keys = ['제목', '주제', '주인공', '챕터']
                    missing_keys = [key for key in required_keys if not structure.get(key)]
                    
                    if missing_keys:
                        raise ValueError(f"필수 키가 누락되었습니다: {missing_keys}")
                    
                    # 주인공 필수 키 확인
                    required_main_char_keys = ['이름', '나이', '직업', '설명']
                    missing_main_char_keys = [key for key in required_main_char_keys if not structure['주인공'].get(key)]
                    if missing_main_char_keys:
                        raise ValueError(f"주인공 정보에 필수 키가 누락되었습니다: {missing_main_char_keys}")
                    
                    # 챕터 필수 키 확인
                    for i, chapter in enumerate(structure['챕터'], 1):
                        required_chapter_keys = ['제목', '사건', '갈등', '요약']
                        missing_chapter_keys = [key for key in required_chapter_keys if not chapter.get(key)]
                        if missing_chapter_keys:
                            raise ValueError(f"챕터 {i}에 필수 키가 누락되었습니다: {missing_chapter_keys}")
                    
                    self.logger.info("소설 구조 생성 및 검증 완료")
                    return structure
                    
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(f"시도 {attempt + 1}에서 유효하지 않은 구조 발생: {e}")
                    continue
                
            except Exception as e:
                self.logger.error(f"시도 {attempt + 1} 실패: {e}")
                if attempt < max_retries - 1:
                    self.logger.info("재시도 중...")
                    time.sleep(2)
                else:
                    raise
        
        raise ValueError("모든 시도 후에도 유효한 소설 구조 생성 실패")

class StoryGenerator:
    def __init__(self, llm_type: str = "gguf", db_path: str = "stories.db", **kwargs):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.init_db()
        
        # 장르 정의
        self.genres = {
            "로맨스": {
                "배경": ["현대", "사극"],
                "핵심_요소": ["연애", "로맨스", "관계"],
                "클리셰": [
                    "우연한 만남",
                    "계약 관계",
                    "신분 차이",
                    "오해와 화해",
                    "삼각관계",
                    "재회",
                    "첫사랑"
                ]
            },
            "로맨스_판타지": {
                "배경": ["이세계", "판타지 세계", "중세 유럽풍"],
                "핵심_요소": ["로맨스", "마법", "모험", "신분제"],
                "클리셰": [
                    "빙의/환생",
                    "악녀/빌런",
                    "황녀/공녀",
                    "마법 아카데미",
                    "숨겨진 신분",
                    "운명적 만남",
                    "계약 결혼"
                ]
            },
            "판타지": {
                "배경": ["중세 판타지", "이세계"],
                "핵심_요소": ["마법", "모험", "전투", "성장"],
                "클리셰": [
                    "용사와 마왕",
                    "마법 수행",
                    "던전 공략",
                    "파티 결성",
                    "전설의 무기",
                    "선택받은 자",
                    "길드 시스템"
                ]
            },
            "무협": {
                "배경": ["고대 중국", "중세 중국", "가상의 무협 세계"],
                "핵심_요소": ["무공", "협객", "문파", "의리"],
                "클리셰": [
                    "무공 비급",
                    "문파 갈등",
                    "고수와 제자",
                    "강호 정의",
                    "내공 수련",
                    "협객의 도",
                    "은원과 복수"
                ]
            },
            "현대_판타지": {
                "배경": ["현대", "근미래"],
                "핵심_요소": ["판타지", "현실", "성공", "성장"],
                "클리셰": [
                    "각성자",
                    "던전 출현",
                    "헌터/레이더",
                    "길드 시스템",
                    "스킬 창",
                    "랭킹 시스템",
                    "히든 퀘스트"
                ]
            },
            "게임_판타지": {
                "배경": ["근미래", "게임 세계"],
                "핵심_요소": ["VRMMORPG", "게임", "모험"],
                "클리셰": [
                    "게임 로그인",
                    "레벨업",
                    "아이템 수집",
                    "퀘스트",
                    "NPC와 교류",
                    "버그/시스템 오류",
                    "현실과 가상의 경계"
                ]
            }
        }
        
        # LLM 모델 초기화
        if llm_type == "gguf":
            gguf_params = {
                'model_path': kwargs.get('model_path'),
                'repo_id': kwargs.get('repo_id'),
                'filename': kwargs.get('filename')
            }
            gguf_params = {k: v for k, v in gguf_params.items() if v is not None}
            self.llm = GGUFModel(**gguf_params)
        elif llm_type == "openai":
            self.llm = OpenAIModel(**kwargs)
        elif llm_type == "transformer":
            self.llm = TransformerModel(**kwargs)
        elif llm_type == "lmstudio":
            self.llm = LMStudioModel(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

        self.current_dir = None  # 현재 작업 디렉토리 저장용
        self.selected_genre = None  # 선택된 장르 저장

    def init_db(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # 트렌드 테이블에서 traffic 제거
            c.execute('''CREATE TABLE IF NOT EXISTS trending_topics
                        (keyword TEXT PRIMARY KEY,
                         news_items TEXT,
                         collected_at TIMESTAMP)''')
            
            # 생성된 단편소설 테이블
            c.execute('''CREATE TABLE IF NOT EXISTS generated_stories
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         title TEXT,
                         content TEXT,
                         inspiration_topics TEXT,
                         generated_at TIMESTAMP)''')
            
            conn.commit()

    def collect_trending_topics(self) -> List[Dict[str, Any]]:
        """Google Trends에서 실시간 검색어와 관련 뉴스 수집"""
        try:
            logger.info("Collecting trending topics from Google Trends...")
            feed = feedparser.parse('https://trends.google.co.kr/trending/rss?geo=KR')
            trending_topics = []
            
            for entry in feed.entries:
                keyword = entry.title
                logger.info(f"Processing trending topic: {keyword}")
                
                topic_data = {
                    'keyword': keyword,
                    'news_items': [],
                    'collected_at': datetime.now().isoformat()
                }
                
                try:
                    # RSS 피드의 뉴스 정보 사용
                    news_item = {
                        'title': entry.get('ht_news_item_title', ''),
                        'snippet': entry.get('ht_news_item_snippet', ''),
                        'source': entry.get('ht_news_item_source', ''),
                        'url': entry.get('ht_news_item_url', '')
                    }
                    
                    if news_item['snippet']:  # snippet이 있는 경우만 추가
                        topic_data['news_items'].append(news_item)
                        logger.info(f"Added news item: {news_item['title']}")
                    
                except Exception as e:
                    logger.error(f"Error processing news for {keyword}: {e}")
                    continue
                
                trending_topics.append(topic_data)
                logger.info(f"Collected {len(topic_data['news_items'])} news items for: {keyword}")
            
            logger.info(f"Total trending topics collected: {len(trending_topics)}")
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error collecting trending topics: {e}")
            logger.exception("Detailed error information:")
            return []

    def _summarize_news(self, content: str) -> str:
        """LLM을 사용하여 뉴스 기사 요약"""
        try:
            prompt = (
                "User: 다음 뉴스 기사를 3줄로 요약해주세요:\n\n"
                f"{content}\n"
                "Assistant:"
            )
            
            response = self.llm.generate(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error summarizing news: {e}")
            return ""

    def store_trending_topics(self, topics: List[Dict[str, Any]]):
        """트렌드 데이터를 DB에 저장"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            for topic in topics:
                try:
                    c.execute('''INSERT OR REPLACE INTO trending_topics
                                (keyword, news_items, collected_at)
                                VALUES (?, ?, ?)''',
                             (topic['keyword'],
                              json.dumps(topic['news_items'], ensure_ascii=False),
                              topic['collected_at']))
                except sqlite3.Error as e:
                    logger.error(f"Error storing trend data for {topic['keyword']}: {e}")
                    continue
            
            conn.commit()

    def _format_trends(self, trending_topics: List[Dict[str, Any]]) -> str:
        """트렌드 데이터를 프롬프트용으로 포맷팅"""
        formatted = []
        for i, topic in enumerate(trending_topics, 1):
            topic_info = [f"{i}. {topic['keyword']}"]
            
            # 뉴스 요약 추가
            for news in topic.get('news_items', []):
                title = news.get('title', '')
                snippet = news.get('snippet', '')
                if title and snippet:
                    topic_info.append(f"   - Title: {title}")
                    topic_info.append(f"   - Summary: {snippet}")
            
            formatted.append('\n'.join(topic_info))
        
        return '\n\n'.join(formatted)

    def _select_genre(self, trending_topics: List[Dict[str, Any]]) -> str:
        """트렌드 데이터를 기반으로 적합한 장르 선택"""
        genre_prompt = (
            "System: 다음 트렌드 데이터를 분석하여 가장 적합한 웹소설 장르를 선택해주세요.\n\n"
            "선택 가능한 장르:\n"
            "- 로맨스: 현대/사극 배경의 연애 이야기\n"
            "- 로맨스_판타지: 판타지 세계관의 로맨스\n"
            "- 판타지: 중세 서양 판타지의 모험과 전투\n"
            "- 무협: 중국풍 세계관의 무공과 협객\n"
            "- 현대_판타지: 현대 배경의 판타지\n"
            "- 게임_판타지: VR게임 배경의 판타지\n\n"
            "트렌드 데이터:\n"
        )
        
        # 트렌드 데이터 추가
        for topic in trending_topics:
            genre_prompt += f"- {topic['keyword']}\n"
            for news in topic.get('news_items', []):
                genre_prompt += f"  * {news.get('title', '')}\n"
        
        genre_prompt += (
            "\n다음 JSON 형식으로 응답해주세요:\n"
            "{\n"
            '  "선택_장르": "장르명",\n'
            '  "선택_이유": "선택 이유 설명"\n'
            "}\n\n"
            "Assistant: "
        )
        
        try:
            response = self.llm.generate(genre_prompt)
            result = json.loads(response)
            
            selected_genre = result['선택_장르']
            if selected_genre not in self.genres:
                self.logger.warning(f"Invalid genre selected: {selected_genre}")
                selected_genre = random.choice(list(self.genres.keys()))
            
            self.logger.info(f"Selected genre: {selected_genre}")
            self.logger.info(f"Selection reason: {result['선택_이유']}")
            
            return selected_genre
            
        except Exception as e:
            self.logger.error(f"Genre selection failed: {e}")
            return random.choice(list(self.genres.keys()))

    def _analyze_trends(self, trending_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """트렌드 데이터 분석하여 소설 구조의 기초 생성"""
        # 장르 선택
        self.selected_genre = self._select_genre(trending_topics)
        genre_info = self.genres[self.selected_genre]
        
        self.logger.info(f"\n선택된 장르: {self.selected_genre}")
        self.logger.info(f"장르 특성:")
        self.logger.info(f"- 배경: {', '.join(genre_info['배경'])}")
        self.logger.info(f"- 핵심 요소: {', '.join(genre_info['핵심_요소'])}")
        self.logger.info(f"- 주요 클리셰: {', '.join(genre_info['클리셰'])}\n")
        
        # 트렌드 데이터 요약
        summarized_trends = []
        for i, topic in enumerate(trending_topics[:10], 1):  # 상위 10개만 사용
            trend_info = ["{0}. {1}".format(i, topic['keyword'])]
            if topic['news_items']:
                trend_info.append("   - {0}".format(topic['news_items'][0]['title']))
            summarized_trends.append('\n'.join(trend_info))
        
        analysis_prompt = (
            "System: JSON 형식으로만 응답하는 소설 구조 분석 시스템입니다. "
            "선택된 장르의 특성과 클리셰를 반영하여 분석해주세요.\n\n"
            f"선택된 장르: {self.selected_genre}\n"
            f"장르 특성:\n"
            f"- 배경: {', '.join(genre_info['배경'])}\n"
            f"- 핵심 요소: {', '.join(genre_info['핵심_요소'])}\n"
            f"- 주요 클리셰: {', '.join(genre_info['클리셰'])}\n\n"
            "응답은 반드시 다음 JSON 형식을 따라주세요:\n"
            "{\n"
            '  "주제": "문자열",\n'
            '  "부주제": ["문자열"],\n'
            '  "인물_제안": [{\n'
            '    "유형": "문자열",\n'
            '    "영감": "문자열",\n'
            '    "역할": "문자열",\n'
            '    "배경_설정": "문자열",\n'
            '    "관련_주제": ["문자열"]\n'
            '  }],\n'
            '  "갈등_요소": [{\n'
            '    "유형": "문자열",\n'
            '    "영감": "문자열",\n'
            '    "구체적_설정": "문자열",\n'
            '    "전개": "문자열",\n'
            '    "연관_주제": ["문자열"]\n'
            '  }],\n'
            '  "배경_제안": {\n'
            '    "시대": "문자열",\n'
            '    "장소": "문자열",\n'
            '    "분위기": "문자열",\n'
            '    "영감_요소": ["문자열"]\n'
            '  },\n'
            '  "장르_요소": {\n'
            '    "선택_이유": "문자열",\n'
            '    "활용_클리셰": ["문자열"],\n'
            '    "주요_특성": ["문자열"]\n'
            '  }\n'
            "}\n\n"
            "User: 다음 트렌드를 바탕으로 선택된 장르에 맞는 소설 구조를 분석해주세요:\n\n"
            f"{' '.join(summarized_trends)}\n\n"
            "Assistant: "
        )
        
        response = self.llm.generate(analysis_prompt, max_tokens=10240)
        return self._extract_json_from_response(response, validation_type='analysis')

    def _extract_json_from_response(self, response: str, validation_type: str = None) -> Dict:
        """응답에서 JSON 추출 및 검증"""
        self.logger.debug(f"=== Extracting JSON ({validation_type}) ===")
        self.logger.debug(f"Raw response:\n{response}")
        
        try:
            # JSON 시작과 끝 위치 찾기
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.error("No JSON object markers found in response")
                self.logger.debug(f"Response content:\n{response}")
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            self.logger.debug(f"Extracted JSON string:\n{json_str}")
            
            result = json.loads(json_str)
            self.logger.debug(f"Parsed JSON object:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.debug(f"Attempted to parse:\n{json_str}")
            raise ValueError(f"Failed to parse JSON: {e}")

    def _generate_structure(self, trending_topics: List[Dict[str, Any]], analysis: Dict, max_retries: int = 3) -> Dict:
        """소설의 전체 구조 생성"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"\n구조 생성 시도 {attempt + 1}/{max_retries}")
                
                structure_prompt = (
                    "System: 당신은 소설 구조를 생성하는 AI 어시스턴트입니다. "
                    "다음 규칙을 엄격히 따라주세요:\n"
                    "1. 응답은 반드시 유효한 JSON이어야 합니다\n"
                    "2. 최상위 레벨에 다음 키들이 반드시 포함되어야 합니다:\n"
                    "   - 제목: 소설 제목 (문자열)\n"
                    "   - 주제: 이야기의 핵심 주제 (문자열)\n"
                    "   - 주인공: 주인공 정보 (객체)\n"
                    "     * 이름: 주인공 이름 (문자열)\n"
                    "     * 나이: 주인공 나이 (문자열)\n"
                    "     * 직업: 주인공 직업 (문자열)\n"
                    "     * 설명: 주인공 성격과 배경 (문자열)\n"
                    "   - 조연: 조연 인물 목록 (배열)\n"
                    "     * 각 조연은 다음 키를 가져야 함:\n"
                    "       - 이름: 조연 이름 (문자열)\n"
                    "       - 역할: 조연의 역할 (문자열)\n"
                    "       - 설명: 조연 설명 (문자열)\n"
                    "   - 챕터: 8개의 챕터 정보 (배열)\n"
                    "     * 각 챕터는 다음 키를 가져야 함:\n"
                    "       - 제목: 챕터 제목 (문자열)\n"
                    "       - 사건: 주요 사건 목록 (문자열 배열)\n"
                    "       - 갈등: 갈등 요소 목록 (문자열 배열)\n"
                    "       - 요약: 챕터 요약 (문자열)\n"
                    "3. 설명이나 부가 텍스트를 포함하지 마세요\n"
                    "4. 중괄호로 시작하고 끝내세요\n"
                    "5. 모든 키는 한글로 작성해주세요\n"
                    "6. 위에서 명시한 모든 필수 키가 반드시 포함되어야 합니다\n"
                    "7. 각 키의 데이터 타입이 정확히 일치해야 합니다\n\n"
                    "User: 다음 요소들을 바탕으로 한국적 정서가 담긴 소설 구조를 만들어주세요. "
                    "보편적인 주제를 탐구하되 한국의 문화적 맥락을 반영해주세요.\n\n"
                    f"주제: {analysis['주제']}\n"
                    f"부주제: {', '.join(analysis['부주제'])}\n\n"
                    "인물 유형:\n"
                    + '\n'.join([
                        f"- {char['유형']}: {char['배경_설정']}"
                        for char in analysis['인물_제안']
                    ]) + "\n\n"
                    + ("갈등 요소:\n"
                    + '\n'.join([
                        f"- {conflict['유형']}: {conflict['구체적_설정']}"
                        for conflict in analysis.get('갈등_요소', [])
                    ]) + "\n\n") if '갈등_요소' in analysis else ""
                    + f"배경: {analysis.get('배경_제안', {}).get('장소', '현대 한국')}\n"
                    + f"분위기: {analysis.get('배경_제안', {}).get('분위기', '현대적')}\n\n"
                    "요구사항:\n"
                    "1. 8개의 챕터로 구성하여 명확한 이야기 전개를 보여주세요\n"
                    "2. 허구의 인물과 배경을 사용하세요\n"
                    "3. 보편적 주제를 한국적 맥락에서 다루세요\n"
                    "4. 감정선을 진정성 있게 표현하세요\n"
                    "5. 설득력 있는 인물 성장을 보여주세요\n\n"
                    "다음 JSON 구조로 출력해주세요:\n"
                    "{\n"
                    '  "제목": "소설 제목",\n'
                    '  "주제": "이야기의 핵심 주제",\n'
                    '  "주인공": {\n'
                    '    "이름": "이름",\n'
                    '    "나이": "나이",\n'
                    '    "직업": "직업",\n'
                    '    "설명": "성격과 배경"\n'
                    '  },\n'
                    '  "조연": [{\n'
                    '    "이름": "이름",\n'
                    '    "역할": "역할",\n'
                    '    "설명": "설명"\n'
                    '  }],\n'
                    '  "챕터": [{\n'
                    '    "제목": "챕터 제목",\n'
                    '    "사건": ["주요 사건1", "주요 사건2"],\n'
                    '    "갈등": ["갈등1", "갈등2"],\n'
                    '    "요약": "챕터 요약"\n'
                    '  }]\n'
                    "}\n\n"
                    "Assistant: "
                )
                
                response = self.llm.generate(structure_prompt, max_tokens=10240)
                
                try:
                    # JSON 파싱
                    structure = json.loads(response)
                    
                    # 중첩된 구조 처리
                    if "소설_구조" in structure:
                        structure = structure["소설_구조"]
                    
                    # 필수 키 확인
                    required_keys = ['제목', '주제', '주인공', '챕터']
                    missing_keys = [key for key in required_keys if not structure.get(key)]
                    
                    if missing_keys:
                        raise ValueError(f"필수 키가 누락되었습니다: {missing_keys}")
                    
                    # 주인공 필수 키 확인
                    required_main_char_keys = ['이름', '나이', '직업', '설명']
                    missing_main_char_keys = [key for key in required_main_char_keys if not structure['주인공'].get(key)]
                    if missing_main_char_keys:
                        raise ValueError(f"주인공 정보에 필수 키가 누락되었습니다: {missing_main_char_keys}")
                    
                    # 챕터 필수 키 확인
                    for i, chapter in enumerate(structure['챕터'], 1):
                        required_chapter_keys = ['제목', '사건', '갈등', '요약']
                        missing_chapter_keys = [key for key in required_chapter_keys if not chapter.get(key)]
                        if missing_chapter_keys:
                            raise ValueError(f"챕터 {i}에 필수 키가 누락되었습니다: {missing_chapter_keys}")
                    
                    self.logger.info("소설 구조 생성 및 검증 완료")
                    return structure
                    
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(f"시도 {attempt + 1}에서 유효하지 않은 구조 발생: {e}")
                    continue
                
            except Exception as e:
                self.logger.error(f"시도 {attempt + 1} 실패: {e}")
                if attempt < max_retries - 1:
                    self.logger.info("재시도 중...")
                    time.sleep(2)
                else:
                    raise
        
        raise ValueError("모든 시도 후에도 유효한 소설 구조 생성 실패")

    def _replace_real_names(self, keywords: List[str]) -> Dict[str, str]:
        """실제 인물 이름을 가명으로 대체"""
        name_mapping = {}
        korean_surnames = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
        korean_names = ["지원", "민준", "서연", "준호", "미래", "도윤", "하은", "지훈", "유진", "서준"]
        
        for keyword in keywords:
            # 한글 이름 패턴 확인 (2-4자 한글)
            if re.match(r'^[가-힣]{2,4}$', keyword):
                # 임의의 가명 생성
                surname = random.choice(korean_surnames)
                name = random.choice(korean_names)
                name_mapping[keyword] = f"{surname}{name}"
        
        return name_mapping

    def _create_emotion_map(self, structure: Dict) -> Dict:
        """전체 스토리의 감정선 맵을 생성합니다."""
        emotion_prompt = (
            "System: 당신은 스토리의 감정선을 설계하는 전문가입니다. "
            "다음 소설 구조를 바탕으로 각 챕터별 감정선과 긴장도를 설계해주세요.\n\n"
            "응답은 반드시 다음 JSON 형식을 따라주세요:\n"
            "{\n"
            '  "전체_감정_곡선": "전체 스토리의 감정 흐름 설명",\n'
            '  "챕터별_감정": [\n'
            '    {\n'
            '      "챕터_번호": 1,\n'
            '      "주요_감정": ["감정1", "감정2"],\n'
            '      "긴장도": 5,\n'
            '      "감정_변화": "이 챕터에서의 감정 변화 설명",\n'
            '      "연결_포인트": "다음 챕터와의 감정적 연결점"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            f"소설 제목: {structure['제목']}\n"
            f"주제: {structure['주제']}\n"
            "챕터 구조:\n"
        )
        
        # 챕터 정보 추가
        for i, chapter in enumerate(structure['챕터'], 1):
            emotion_prompt += (
                f"\n{i}. {chapter['제목']}\n"
                f"   사건: {', '.join(chapter['사건'])}\n"
                f"   갈등: {', '.join(chapter['갈등'])}\n"
                f"   요약: {chapter['요약']}"
            )
        
        emotion_prompt += (
            "\n\n요구사항:\n"
            "1. 각 챕터의 감정선이 전체 스토리의 흐름과 일치해야 합니다\n"
            "2. 긴장도는 1-10 사이의 정수여야 합니다\n"
            "3. 주요 감정은 2-3개의 핵심 감정을 포함해야 합니다\n"
            "4. 감정 변화는 구체적이고 명확해야 합니다\n"
            "5. 연결 포인트는 다음 챕터와의 자연스러운 감정적 전환을 설명해야 합니다\n"
            "6. 마지막 챕터의 연결 포인트는 전체 이야기의 마무리를 설명해야 합니다\n\n"
            "Assistant: "
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(emotion_prompt, max_tokens=4096)
                
                # JSON 시작과 끝 위치 찾기
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx == -1 or end_idx <= 0:
                    raise ValueError("JSON 구조를 찾을 수 없습니다")
                
                json_str = response[start_idx:end_idx]
                emotion_map = json.loads(json_str)
                
                # 필수 키 검증
                if '전체_감정_곡선' not in emotion_map:
                    raise ValueError("전체_감정_곡선이 누락되었습니다")
                if '챕터별_감정' not in emotion_map:
                    raise ValueError("챕터별_감정이 누락되었습니다")
                
                # 각 챕터별 감정 데이터 검증
                for chapter_emotion in emotion_map['챕터별_감정']:
                    required_keys = ['챕터_번호', '주요_감정', '긴장도', '감정_변화', '연결_포인트']
                    missing_keys = [key for key in required_keys if key not in chapter_emotion]
                    if missing_keys:
                        raise ValueError(f"챕터 {chapter_emotion.get('챕터_번호', '알 수 없음')}의 감정 데이터에서 다음 키가 누락됨: {missing_keys}")
                    
                    # 긴장도 값 검증
                    if not isinstance(chapter_emotion['긴장도'], (int, float)) or not 1 <= chapter_emotion['긴장도'] <= 10:
                        chapter_emotion['긴장도'] = max(1, min(10, int(float(str(chapter_emotion['긴장도'])))))
                    
                    # 주요_감정이 리스트인지 확인
                    if not isinstance(chapter_emotion['주요_감정'], list):
                        chapter_emotion['주요_감정'] = [str(chapter_emotion['주요_감정'])]
                
                # 로깅
                self.logger.info("\n=== 감정선 맵 생성 완료 ===")
                self.logger.info(f"전체 감정 곡선: {emotion_map['전체_감정_곡선']}")
                for chapter in emotion_map['챕터별_감정']:
                    self.logger.info(f"\n챕터 {chapter['챕터_번호']}:")
                    self.logger.info(f"주요 감정: {', '.join(chapter['주요_감정'])}")
                    self.logger.info(f"긴장도: {chapter['긴장도']}")
                    self.logger.info(f"감정 변화: {chapter['감정_변화']}")
                
                return emotion_map
                
            except Exception as e:
                self.logger.error(f"감정선 맵 생성 시도 {attempt + 1} 실패: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("모든 시도 실패. 기본 감정선 맵 생성")
                    # 기본 감정선 맵 생성
                    default_map = {
                        "전체_감정_곡선": "기본적인 기승전결 구조를 따르는 감정선",
                        "챕터별_감정": []
                    }
                    for i in range(len(structure['챕터'])):
                        default_map["챕터별_감정"].append({
                            "챕터_번호": i + 1,
                            "주요_감정": ["기대", "긴장"],
                            "긴장도": min(8, max(3, i + 3)),  # 3부터 시작해서 점진적 상승
                            "감정_변화": "기본적인 감정 변화",
                            "연결_포인트": "다음 챕터로의 자연스러운 전개"
                        })
                    return default_map
                time.sleep(2)  # 재시도 전 대기
        
        return None

    def _generate_chapter(self, structure: Dict, chapter: Dict, analysis: Dict, chapter_index: int = 1, previous_chapters: List[str] = None, emotion_map: Dict = None) -> str:
        """각 챕터의 상세 내용 생성"""
        
        # 이전 챕터들의 요약 생성
        previous_chapters_summary = ""
        if previous_chapters and chapter_index > 1:
            summary_prompt = (
                "System: 다음 챕터들의 핵심 내용을 간단히 요약해주세요. "
                "다음 챕터 작성에 필요한 핵심 정보만 포함해주세요.\n\n"
                "이전 챕터 내용:\n"
                f"{' '.join(previous_chapters)}\n\n"
                "Assistant: "
            )
            try:
                previous_chapters_summary = self.llm.generate(summary_prompt, max_tokens=1024)
            except Exception as e:
                self.logger.warning(f"이전 챕터 요약 생성 실패: {e}")
                previous_chapters_summary = "요약 생성 실패"
        
        # 전체 스토리 맥락 파악
        story_context = (
            f"주제: {structure['주제']}\n"
            f"주인공: {structure['주인공']['이름']} - {structure['주인공']['설명']}\n"
            "조연:\n" + '\n'.join([f"- {char['이름']}: {char['설명']}" for char in structure['조연']])
        )
        
        # 현재 챕터의 위치와 역할 파악
        chapter_position = (
            "이 챕터의 위치적 특성:\n"
            f"- 전체 {len(structure['챕터'])}개 챕터 중 {chapter_index}번째\n"
            f"- {'도입부' if chapter_index <= 2 else '전개부' if chapter_index <= 6 else '결말부'} 위치\n"
            f"- {'인물과 배경 소개' if chapter_index == 1 else '갈등 심화' if chapter_index <= 6 else '해결과 마무리'} 단계"
        )
        
        # 감정선 정보 추가
        emotion_context = ""
        if emotion_map and '챕터별_감정' in emotion_map:
            current_emotion = next((e for e in emotion_map['챕터별_감정'] if e['챕터_번호'] == chapter_index), None)
            if current_emotion:
                emotion_context = (
                    "\n감정선 가이드:\n"
                    f"- 주요 감정: {', '.join(current_emotion['주요_감정'])}\n"
                    f"- 긴장도: {current_emotion['긴장도']}/10\n"
                    f"- 감정 변화: {current_emotion['감정_변화']}\n"
                    f"- 다음 챕터 연결: {current_emotion['연결_포인트']}"
                )
        
        chapter_prompt = (
            "System: 당신은 소설 챕터를 생성하는 AI 어시스턴트입니다. "
            "다음 규칙을 엄격히 따라주세요:\n"
            "1. 응답은 반드시 유효한 JSON이어야 합니다\n"
            "2. 최상위 레벨에 다음 키가 반드시 포함되어야 합니다:\n"
            "   - 내용: 챕터 내용 (문자열)\n"
            "3. 마크다운 포맷을 활용해주세요:\n"
            "   - 섹션 제목은 H2 (##)\n"
            "   - 중요한 장면 전환은 수평선 (---)\n"
            "   - 대화는 인용구 (>)\n"
            "   - 강조가 필요한 부분은 이탤릭 (*) 또는 볼드 (**)\n"
            "   - 시간/장소 전환은 이탤릭 (*)\n"
            "4. 설명이나 부가 텍스트를 포함하지 마세요\n"
            "5. 모든 키는 한글로 작성해주세요\n\n"
            "User: 다음 요소들을 바탕으로 한국적 정서가 담긴 소설 챕터를 작성해주세요.\n"
            "특히 이전 챕터와의 자연스러운 연결과 전체 스토리의 흐름을 고려해주세요.\n\n"
            f"소설 제목: {structure['제목']}\n"
            f"챕터 제목: {chapter['제목']}\n"
            f"주요 사건: {', '.join(chapter['사건'])}\n"
            f"갈등 요소: {', '.join(chapter['갈등'])}\n\n"
            f"전체 스토리 맥락:\n{story_context}\n\n"
            f"챕터 위치와 역할:\n{chapter_position}\n"
            f"{emotion_context}\n\n"
        )
        
        # 이전 챕터 요약이 있는 경우에만 추가
        if previous_chapters_summary:
            chapter_prompt += (
                "이전 챕터들의 주요 내용:\n"
                f"{previous_chapters_summary}\n\n"
                "위 내용을 자연스럽게 이어받아 현재 챕터를 전개해주세요.\n\n"
            )
        
        chapter_prompt += (
            "요구사항:\n"
            "1. 자연스러운 한국어 대화체를 사용해주세요\n"
            "2. 생생한 묘사로 장면을 표현해주세요\n"
            "3. 한국적 정서와 문화적 맥락을 반영해주세요\n"
            "4. 약 2000자 분량으로 작성해주세요\n"
            "5. 이전 챕터와의 자연스러운 연결성을 유지해주세요\n"
            "6. 전체 스토리의 흐름에 맞는 전개를 해주세요\n"
            "7. 지정된 감정선과 긴장도를 자연스럽게 표현해주세요\n\n"
            "다음 JSON 구조로 출력해주세요:\n"
            "{\n"
            '  "내용": "챕터 내용"\n'
            "}\n\n"
            "Assistant: "
        )
        
        response = self.llm.generate(chapter_prompt, max_tokens=10240)
        
        try:
            # JSON 파싱
            chapter_data = json.loads(response)
            
            # 필수 키 확인
            if '내용' not in chapter_data:
                raise ValueError("필수 키 '내용'이 누락되었습니다")
            
            # 챕터 내용 반환
            return chapter_data['내용']
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"챕터 생성 중 오류 발생: {e}")
            raise

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """텍스트를 청크로 나누기"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # 공백 포함
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _clean_story_response(self, response: str) -> str:
        """응답 텍스트를 정리하여 소설 형식에 맞게 변환"""
        # 줄바꿈 정리
        response = response.replace('\n', ' ').replace('\r', '').replace('\t', '').strip()
        
        # 문장 부호 정리
        response = re.sub(r'[^\w\s]', '', response)
        
        # 긴 문장 분할
        sentences = re.split(r'(?<=[.!?])', response)
        
        # 최대 길이 제한
        max_length = 3000
        
        cleaned_content = []
        for sentence in sentences:
            if len(cleaned_content) + len(sentence) > max_length:
                cleaned_content.append(sentence)
            else:
                cleaned_content[-1] += sentence
        
        return '\n\n'.join(cleaned_content)

    def _translate_to_korean(self, text: str) -> str:
        """영어 텍스트를 한글로 번역"""
        prompt = (
            "System: You are a professional translator. Translate the following English text to Korean.\n"
            "Requirements:\n"
            "1. Maintain the original meaning and nuance\n"
            "2. Use natural, literary Korean\n"
            "3. Keep any proper nouns as is\n"
            "4. Do not add explanatory notes\n\n"
            f"English text: {text}\n\n"
            "Korean translation:"
        )
        
        response = self.llm.generate(prompt, max_tokens=8192)
        return response.strip()

    def _format_source_summary(self, trending_topics: List[Dict[str, Any]]) -> str:
        """소스 데이터 요약 포맷팅"""
        summary = []
        for i, topic in enumerate(trending_topics, 1):
            topic_summary = [f"Topic {i}: {topic['keyword']}"]
            
            # 뉴스 요약 추가
            news_items = []
            for news in topic.get('news_items', []):
                title = news.get('title', '')
                snippet = news.get('snippet', '')
                if title and snippet:
                    news_items.append(f"제목: {title}\n요약: {snippet}")
            
            if news_items:
                topic_summary.extend(news_items)
            
            summary.append('\n'.join(topic_summary))
        
        return '\n\n'.join(summary)

    def _load_progress(self, output_dir: str) -> tuple[Dict, Dict, List[int]]:
        """기존 진행 상황 로드"""
        self.logger.info(f"Checking progress in: {output_dir}")
        
        structure = None
        analysis = None
        completed_chapters = []
        
        # structure.json 확인
        structure_path = os.path.join(output_dir, 'structure.json')
        if os.path.exists(structure_path):
            with open(structure_path, 'r', encoding='utf-8') as f:
                structure = json.load(f)
            self.logger.info("Loaded existing structure")
        
        # analysis.json 확인
        analysis_path = os.path.join(output_dir, 'analysis.json')
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            self.logger.info("Loaded existing analysis")
        
        # 완성된 챕터 확인
        for i in range(1, 9):  # 8챕터
            ko_path = os.path.join(output_dir, f'chapter_{i:02d}.txt')
            en_path = os.path.join(output_dir, f'chapter_{i:02d}_en.txt')
            if os.path.exists(ko_path) and os.path.exists(en_path):
                completed_chapters.append(i)
        
        if completed_chapters:
            self.logger.info(f"Found completed chapters: {completed_chapters}")
        
        return structure, analysis, completed_chapters

    def _analyze_story_for_improvement(self, story_content: str, structure: Dict, chapter_contents: List[str]) -> Dict:
        """소설의 전체적인 개선점과 챕터별 상세 개선점을 분석합니다."""
        self.logger.info("\n=== 소설 분석 시작 ===")
        
        analysis_prompt = (
            "System: 당신은 소설 분석 전문가입니다. "
            "다음 소설을 분석하여 개선점을 제시해주세요.\n\n"
            "응답은 반드시 다음 JSON 형식을 따라주세요:\n"
            "{\n"
            '  "전체_개선점": {\n'
            '    "구조적_개선": ["개선점1", "개선점2"],\n'
            '    "문체_개선": ["개선점1", "개선점2"],\n'
            '    "캐릭터_개선": ["개선점1", "개선점2"],\n'
            '    "스토리_개선": ["개선점1", "개선점2"]\n'
            '  },\n'
            '  "챕터별_개선점": [{\n'
            '    "챕터_번호": 1,\n'
            '    "구조": ["개선점1", "개선점2"],\n'
            '    "문체": ["개선점1", "개선점2"],\n'
            '    "캐릭터": ["개선점1", "개선점2"],\n'
            '    "스토리": ["개선점1", "개선점2"],\n'
            '    "연결성": ["이전 챕터와의 연결성 개선점", "다음 챕터와의 연결성 개선점"]\n'
            '  }]\n'
            "}\n\n"
            f"소설 제목: {structure['제목']}\n"
            f"주제: {structure['주제']}\n\n"
            "챕터별 내용:\n"
        )
        
        # 챕터별 내용 추가
        for i, chapter in enumerate(chapter_contents, 1):
            analysis_prompt += f"\n=== 챕터 {i}: {structure['챕터'][i-1]['제목']} ===\n"
            analysis_prompt += chapter + "\n"
        
        try:
            response = self.llm.generate(analysis_prompt, max_tokens=10240)
            analysis = json.loads(response)
            
            # 분석 결과 로깅
            self.logger.info("\n=== 분석 결과 ===")
            self.logger.info("\n전체 개선점:")
            for category, points in analysis['전체_개선점'].items():
                self.logger.info(f"\n{category}:")
                for point in points:
                    self.logger.info(f"- {point}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"소설 분석 중 오류 발생: {e}")
            raise

    def _improve_chapter(self, chapter_content: str, chapter_number: int, analysis: Dict, structure: Dict) -> str:
        """개별 챕터를 개선합니다."""
        self.logger.info(f"\n=== 챕터 {chapter_number} 개선 시작 ===")
        
        chapter_analysis = next(
            (ch for ch in analysis['챕터별_개선점'] if ch['챕터_번호'] == chapter_number),
            None
        )
        
        if not chapter_analysis:
            self.logger.warning(f"챕터 {chapter_number}의 분석 정보를 찾을 수 없습니다.")
            return chapter_content
        
        improvement_prompt = (
            "System: 당신은 소설 개선 전문가입니다. "
            "다음 개선점들을 반영하여 챕터를 개선해주세요.\n\n"
            "개선이 필요한 부분:\n"
        )
        
        # 개선점 추가
        for category in ['구조', '문체', '캐릭터', '스토리', '연결성']:
            if category in chapter_analysis and chapter_analysis[category]:
                improvement_prompt += f"\n{category} 개선점:\n"
                for point in chapter_analysis[category]:
                    improvement_prompt += f"- {point}\n"
        
        improvement_prompt += (
            "\n전체 스토리 개선점:\n"
            + "\n".join([f"- {point}" for point in analysis['전체_개선점']['스토리_개선']])
            + "\n\n원본 챕터 내용:\n"
            + chapter_content
            + "\n\n"
            "개선된 챕터를 다음 JSON 형식으로 출력해주세요:\n"
            "{\n"
            '  "개선된_내용": "개선된 챕터 전문",\n'
            '  "개선_사항": ["실제로 적용된 개선 사항 리스트"]\n'
            "}\n\n"
            "Assistant: "
        )
        
        try:
            response = self.llm.generate(improvement_prompt, max_tokens=20480)
            result = json.loads(response)
            
            # 개선 사항 로깅
            self.logger.info(f"\n챕터 {chapter_number} 개선 사항:")
            for improvement in result['개선_사항']:
                self.logger.info(f"- {improvement}")
            
            return result['개선된_내용']
            
        except Exception as e:
            self.logger.error(f"챕터 {chapter_number} 개선 중 오류 발생: {e}")
            return chapter_content

    def _review_and_refine_story(self, story_content: str, structure: Dict) -> str:
        """소설 내용을 검토하고 개선하여 완성도를 높입니다."""
        self.logger.info("\n=== 소설 검토 및 개선 단계 시작 ===")
        
        # 챕터별 내용 로드
        chapter_contents = []
        for i in range(1, 9):
            chapter_path = os.path.join(self.current_dir, f'chapter_{i:02d}.txt')
            with open(chapter_path, 'r', encoding='utf-8') as f:
                chapter_contents.append(f.read())
        
        improved_content = story_content
        
        for iteration in range(3):  # 3회 반복
            self.logger.info(f"\n=== 개선 반복 {iteration + 1}/3 시작 ===")
            
            # 이전 개선 결과를 기반으로 챕터 내용 업데이트
            if iteration > 0:
                chapter_contents = []
                for i in range(1, 9):
                    improved_chapter_path = os.path.join(
                        self.current_dir,
                        f'chapter_{i:02d}_improved_{iteration}.txt'
                    )
                    if os.path.exists(improved_chapter_path):
                        with open(improved_chapter_path, 'r', encoding='utf-8') as f:
                            chapter_contents.append(f.read())
                    else:
                        # 이전 개선 파일이 없으면 원본 챕터 사용
                        original_chapter_path = os.path.join(self.current_dir, f'chapter_{i:02d}.txt')
                        with open(original_chapter_path, 'r', encoding='utf-8') as f:
                            chapter_contents.append(f.read())
            
            # Step 1: 전체 분석
            analysis = self._analyze_story_for_improvement(improved_content, structure, chapter_contents)
            
            # 분석 결과 저장
            analysis_path = os.path.join(
                self.current_dir,
                f'improvement_analysis_{iteration + 1}.json'
            )
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            # Step 2: 챕터별 개선
            improved_chapters = []
            for i, chapter_content in enumerate(chapter_contents, 1):
                improved_chapter = self._improve_chapter(
                    chapter_content, i, analysis, structure
                )
                improved_chapters.append(improved_chapter)
                
                # 개선된 챕터 저장
                improved_chapter_path = os.path.join(
                    self.current_dir,
                    f'chapter_{i:02d}_improved_{iteration + 1}.txt'
                )
                with open(improved_chapter_path, 'w', encoding='utf-8') as f:
                    f.write(improved_chapter)
            
            # Step 3: 전체 소설 재구성
            chapter_contents = improved_chapters
            improved_content = "\n\n".join(improved_chapters)
            
            # 개선된 전체 소설 저장
            full_story_path = os.path.join(
                self.current_dir,
                f'full_story_improved_{iteration + 1}.txt'
            )
            with open(full_story_path, 'w', encoding='utf-8') as f:
                f.write(improved_content)
            
            self.logger.info(f"반복 {iteration + 1} 완료 - 파일 저장됨: {full_story_path}")
        
        # 마크다운 메타데이터 추가
        metadata = (
            "---\n"
            f"title: {structure['제목']}\n"
            f"author: AI Story Generator\n"
            f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"theme: {structure['주제']}\n"
            "---\n\n"
        )
        
        # 목차 생성
        toc = "## 목차\n\n"
        for i, chapter in enumerate(structure['챕터'], 1):
            toc += f"{i}. [{chapter['제목']}](#{chapter['제목'].replace(' ', '-')})\n"
        toc += "\n---\n\n"
        
        return metadata + toc + improved_content

    def generate_story(self, trending_topics: List[Dict[str, Any]], output_dir: str = None) -> Dict[str, Any]:
        try:
            # 출력 디렉토리 설정
            if output_dir is None:
                today = datetime.now().strftime('%Y-%m-%d')
                safe_title = "temp_story"
                # 절대 경로로 output 디렉토리 설정
                base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                output_dir = os.path.join(base_output_dir, f"{today}_{safe_title}")
            elif not output_dir.startswith(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")):
                # output_dir이 지정되었지만 output 디렉토리 하위가 아닌 경우
                base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                output_dir = os.path.join(base_output_dir, os.path.basename(output_dir))
            
            # output 디렉토리가 없으면 생성
            base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(base_output_dir, exist_ok=True)
            
            # 디렉토리가 없으면 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.current_dir = output_dir
            
            # 기존 진행 상황 확인
            structure, analysis, completed_chapters = self._load_progress(output_dir)
            
            # 새로 시작할 때만 트렌드 데이터 필요
            if analysis is None and not trending_topics:
                raise ValueError("새로운 분석을 시작하려면 trending_topics가 필요합니다")
            
            # 1. 트렌드 분석 및 구조 생성
            if analysis is None:
                self.logger.info("\n=== 트렌드 분석 생성 중 ===")
                analysis = self._analyze_trends(trending_topics)
                
                # 분석 결과 저장
                analysis_path = os.path.join(output_dir, 'analysis.json')
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2)
                self.logger.info(f"분석 결과 저장됨: {analysis_path}")
            
            if structure is None:
                self.logger.info("\n=== 소설 구조 생성 중 ===")
                structure = self._generate_structure(trending_topics, analysis)
                
                # 구조가 생성되면 디렉토리 이름 업데이트
                if output_dir.endswith('temp_story'):
                    new_dir = f"{datetime.now().strftime('%Y-%m-%d')}_{structure['제목']}"
                    try:
                        os.rename(output_dir, new_dir)
                        self.current_dir = new_dir
                        output_dir = new_dir
                        self.logger.info(f"디렉토리 이름 변경됨: {new_dir}")
                    except OSError as e:
                        self.logger.warning(f"디렉토리 이름 변경 실패: {e}")
                
                # 구조 저장
                structure_path = os.path.join(output_dir, 'structure.json')
                with open(structure_path, 'w', encoding='utf-8') as f:
                    json.dump(structure, f, ensure_ascii=False, indent=2)
                self.logger.info(f"구조 저장됨: {structure_path}")
            
            # 감정선 맵 생성
            self.logger.info("\n=== 감정선 맵 생성 중 ===")
            emotion_map = self._create_emotion_map(structure)
            
            # 감정선 맵 저장
            emotion_map_path = os.path.join(output_dir, 'emotion_map.json')
            with open(emotion_map_path, 'w', encoding='utf-8') as f:
                json.dump(emotion_map, f, ensure_ascii=False, indent=2)
            self.logger.info(f"감정선 맵 저장됨: {emotion_map_path}")
            
            # 2. 챕터별 내용 생성
            self.logger.info("\n=== 챕터 생성 중 ===")
            full_content = []
            previous_chapters = []
            
            for i, chapter in enumerate(structure['챕터'], 1):
                if i in completed_chapters:
                    self.logger.info(f"\n완성된 챕터 {i} 건너뛰기")
                    # 기존 챕터 로드
                    with open(os.path.join(output_dir, f'chapter_{i:02d}.txt'), 'r', encoding='utf-8') as f:
                        chapter_content = f.read()
                else:
                    self.logger.info(f"\n챕터 {i}/{len(structure['챕터'])} 생성 중")
                    chapter_content = self._generate_chapter(
                        structure, 
                        chapter, 
                        analysis,
                        chapter_index=i,
                        previous_chapters=previous_chapters,
                        emotion_map=emotion_map
                    )
                    
                    # 챕터 저장
                    chapter_path = os.path.join(output_dir, f'chapter_{i:02d}.txt')
                    with open(chapter_path, 'w', encoding='utf-8') as f:
                        # 챕터 내용에 이미 제목이 포함되어 있는지 확인
                        if not chapter_content.startswith(f"# {chapter['제목']}"):
                            f.write(f"# {chapter['제목']}\n\n")
                        f.write(chapter_content)
                
                full_content.append(chapter_content)
                previous_chapters.append(chapter_content)
            
            # 3. 전체 소설 조합 및 저장
            story = {
                'title': structure['제목'],
                'content': '\n\n'.join(full_content)
            }
            
            # 4. 소설 검토 및 개선
            self.logger.info("\n=== 소설 검토 및 개선 시작 ===")
            improved_content = self._review_and_refine_story(story['content'], structure)
            story['content'] = improved_content
            
            # 개선된 버전 저장
            improved_story_path = os.path.join(output_dir, 'improved_story.txt')
            with open(improved_story_path, 'w', encoding='utf-8') as f:
                f.write(f"# {story['title']}\n\n{improved_content}")
            
            # 원본 소설도 저장
            full_story_path = os.path.join(output_dir, 'full_story.txt')
            with open(full_story_path, 'w', encoding='utf-8') as f:
                content = "\n\n".join(full_content)
                f.write(f"# {story['title']}\n\n{content}")
            
            # 로그 파일에 개선 정보 추가
            log_path = os.path.join(output_dir, 'generation.log')
            original_length = len("\n\n".join(full_content))
            improved_length = len(improved_content)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 소설 생성 로그 ===\n")
                f.write(f"생성 시간: {datetime.now().isoformat()}\n")
                f.write(f"제목: {story['title']}\n")
                f.write(f"원본 길이: {original_length} 글자\n")
                f.write(f"개선된 길이: {improved_length} 글자\n")
                f.write(f"총 챕터: {len(structure['챕터'])}\n")
                f.write("\n영감을 준 주제들:\n")
                for topic in trending_topics:
                    f.write(f"- {topic['keyword']}\n")
            
            self.logger.info("\n=== 소설 생성 완료 ===")
            self.logger.info(f"원본 길이: {original_length} 글자")
            self.logger.info(f"개선된 길이: {improved_length} 글자")
            self.logger.info(f"총 챕터: {len(structure['챕터'])}")
            self.logger.info(f"모든 파일이 저장됨: {output_dir}")
            self.logger.info("=== 프로세스 완료 ===\n")
            
            # DB에 저장
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO generated_stories 
                            (title, content, inspiration_topics, generated_at)
                            VALUES (?, ?, ?, ?)''',
                         (story['title'], improved_content,
                          json.dumps(trending_topics, ensure_ascii=False),
                          datetime.now().isoformat()))
                conn.commit()
            print("\n소설이 DB에 저장되었습니다.")
            
            return story
            
        except Exception as e:
            self.logger.error(f"소설 생성 중 오류 발생: {e}")
            self.logger.exception("상세 오류 정보:")
            if self.current_dir:
                self.logger.info(f"다음 디렉토리에서 이어서 작업할 수 있습니다: {self.current_dir}")
            return None

    def refine_existing_story(self, story_dir: str) -> Dict[str, Any]:
        """기존에 생성된 소설을 개선합니다."""
        try:
            self.logger.info(f"\n=== 기존 소설 개선 시작: {story_dir} ===")
            
            # 현재 작업 디렉토리 설정
            self.current_dir = story_dir
            
            # 필요한 파일들 확인
            required_files = [
                'structure.json',
                'analysis.json',
                'emotion_map.json',
                'full_story.txt'
            ]
            
            # 챕터 파일들도 확인
            chapter_files = [f'chapter_{i:02d}.txt' for i in range(1, 9)]
            
            # 모든 필요 파일 존재 확인
            missing_files = []
            for file in required_files + chapter_files:
                if not os.path.exists(os.path.join(story_dir, file)):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"다음 파일들을 찾을 수 없습니다: {', '.join(missing_files)}")
            
            # 필요한 파일들 로드
            with open(os.path.join(story_dir, 'structure.json'), 'r', encoding='utf-8') as f:
                structure = json.load(f)
            
            with open(os.path.join(story_dir, 'full_story.txt'), 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 개선 작업 수행
            self.logger.info("\n=== 소설 검토 및 개선 시작 ===")
            improved_content = self._review_and_refine_story(original_content, structure)
            
            # 개선된 버전 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            improved_story_path = os.path.join(story_dir, f'improved_story_{timestamp}.txt')
            with open(improved_story_path, 'w', encoding='utf-8') as f:
                f.write(improved_content)
            
            # 로그 파일에 개선 정보 추가
            log_path = os.path.join(story_dir, f'refinement_{timestamp}.log')
            original_length = len(original_content)
            improved_length = len(improved_content)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 소설 개선 로그 ===\n")
                f.write(f"개선 시간: {datetime.now().isoformat()}\n")
                f.write(f"제목: {structure['제목']}\n")
                f.write(f"원본 길이: {original_length} 글자\n")
                f.write(f"개선된 길이: {improved_length} 글자\n")
            
            self.logger.info("\n=== 소설 개선 완료 ===")
            self.logger.info(f"원본 길이: {original_length} 글자")
            self.logger.info(f"개선된 길이: {improved_length} 글자")
            self.logger.info(f"개선된 파일 저장됨: {improved_story_path}")
            self.logger.info("=== 프로세스 완료 ===\n")
            
            return {
                'title': structure['제목'],
                'original_content': original_content,
                'improved_content': improved_content,
                'improved_file': improved_story_path
            }
            
        except Exception as e:
            self.logger.error(f"소설 개선 중 오류 발생: {e}")
            self.logger.exception("상세 오류 정보:")
            return None

def main():
    parser = argparse.ArgumentParser(description='트렌드 기반 스토리 생성기')
    parser.add_argument('--llm-type', 
                       choices=['gguf', 'openai', 'transformer', 'lmstudio'],
                       default='gguf',
                       help='사용할 LLM 타입')
    parser.add_argument('--model-path', help='GGUF 모델 경로 (선택사항)')
    parser.add_argument('--repo-id', help='GGUF 모델의 Hugging Face repo ID (선택사항)')
    parser.add_argument('--filename', help='GGUF 모델 파일 이름 (선택사항)')
    parser.add_argument('--api-key', help='OpenAI API 키 (선택사항)')
    parser.add_argument('--model-name',
                       default="deepseek-ai/DeepSeek-R1",
                       help='Transformer 모델 이름')
    parser.add_argument('--lmstudio-url',
                       default="http://localhost:1234/v1",
                       help='LMStudio 서버 URL')
    parser.add_argument('--resume-from', 
                       help='기존 작업을 이어서 할 디렉토리 경로 (예: 2024-02-03_소설제목)')
    parser.add_argument('--refine-story',
                       help='기존 소설을 개선할 디렉토리 경로 (예: 2024-02-03_소설제목)')
    
    args = parser.parse_args()
    
    # 절대 경로로 output 디렉토리 설정
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    kwargs = {}
    if args.llm_type == 'gguf':
        if args.model_path:
            kwargs['model_path'] = args.model_path
        if args.repo_id:
            kwargs['repo_id'] = args.repo_id
        if args.filename:
            kwargs['filename'] = args.filename
    elif args.llm_type == 'openai':
        if args.api_key:
            kwargs['api_key'] = args.api_key
    elif args.llm_type == 'transformer':
        if args.model_name:
            kwargs['model_name'] = args.model_name
    elif args.llm_type == 'lmstudio':
        kwargs['api_base'] = args.lmstudio_url
    
    generator = StoryGenerator(llm_type=args.llm_type, **kwargs)
    
    if args.refine_story:
        # 개선 모드: 기존 소설 개선
        story_path = os.path.join(base_output_dir, args.refine_story) if not args.refine_story.startswith(base_output_dir) else args.refine_story
        print(f"\n기존 소설 개선 시작: {story_path}")
        result = generator.refine_existing_story(story_path)
        
        if result:
            print("\n소설 개선 완료:")
            print("-" * 50)
            print(f"제목: {result['title']}")
            print(f"개선된 파일: {result['improved_file']}")
        else:
            print("\n소설 개선에 실패했습니다.")
    elif args.resume_from:
        # 재개 모드: analysis.json과 structure.json 확인
        resume_path = os.path.join(base_output_dir, args.resume_from) if not args.resume_from.startswith(base_output_dir) else args.resume_from
        analysis_path = os.path.join(resume_path, 'analysis.json')
        structure_path = os.path.join(resume_path, 'structure.json')
        
        if not os.path.exists(analysis_path) or not os.path.exists(structure_path):
            print(f"필수 파일을 찾을 수 없습니다:")
            if not os.path.exists(analysis_path):
                print(f"- {analysis_path}")
            if not os.path.exists(structure_path):
                print(f"- {structure_path}")
            return
            
        print(f"\n기존 작업 재개: {resume_path}")
        story = generator.generate_story(trending_topics=[], output_dir=resume_path)
        
        if story:
            print("\n소설 생성 완료:")
            print("-" * 50)
            print(f"제목: {story['title']}")
            print(f"저장 위치: {generator.current_dir}")
        else:
            print("\n소설 생성에 실패했습니다.")
            if generator.current_dir:
                print(f"다음 명령으로 재시작할 수 있습니다:")
                print(f"python story_generator.py --resume-from {os.path.relpath(generator.current_dir, base_output_dir)}")
    else:
        # 새로 시작: 트렌드 데이터 수집
        trending_topics = generator.collect_trending_topics()
        logger.info(f"Collected {len(trending_topics)} trending topics")
        
        if trending_topics:
            print("\n수집된 트렌드:")
            print("-" * 50)
            for i, topic in enumerate(trending_topics, 1):
                print(f"\n{i}. {topic['keyword']}")
                for news in topic['news_items']:
                    print(f"  * {news['title']}")
            
            print("\n소설 생성 시작...")
            story = generator.generate_story(trending_topics)
            
            if story:
                print("\n생성된 소설:")
                print("-" * 50)
                print(f"제목: {story['title']}")
                print(f"저장 위치: {generator.current_dir}")
            else:
                print("\n소설 생성에 실패했습니다.")
                if generator.current_dir:
                    print(f"다음 명령으로 재시작할 수 있습니다:")
                    print(f"python story_generator.py --resume-from {os.path.relpath(generator.current_dir, base_output_dir)}")
        else:
            print("트렌드 수집에 실패했습니다.")

if __name__ == "__main__":
    main()