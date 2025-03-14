# AI 소설 생성기 (AI Story Generator)

트렌드 데이터를 기반으로 한국적 정서가 담긴 소설을 자동으로 생성하는 AI 기반 소설 생성기입니다.

## 주요 기능

- Google Trends에서 실시간 검색어와 관련 뉴스 수집
- 수집된 트렌드 데이터를 기반으로 소설 구조 자동 생성
- 8개의 챕터로 구성된 완성된 소설 생성
- 다양한 LLM 모델 지원 (GGUF, OpenAI, Transformer, LMStudio)
- 생성된 소설의 자동 저장 및 진행 상황 추적

## 지원하는 LLM 모델

1. GGUF 모델
   - 로컬에서 실행 가능한 GGUF 형식의 모델 지원
   - Hugging Face에서 모델 다운로드 지원

2. OpenAI GPT-4
   - OpenAI API를 통한 고품질 텍스트 생성

3. Transformer 모델
   - Hugging Face의 Transformer 모델 지원

4. LMStudio
   - 로컬 LLM 서버를 통한 텍스트 생성 지원

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 환경 설정:
   - OpenAI API 키 설정 (선택사항)
   - LMStudio 서버 URL 설정 (선택사항)

## 사용 방법

### 기본 실행
```bash
python story_generator.py
```

### 특정 LLM 모델 사용
```bash
# GGUF 모델 사용
python story_generator.py --llm-type gguf --model-path /path/to/model.gguf

# OpenAI 사용
python story_generator.py --llm-type openai --api-key your-api-key

# Transformer 모델 사용
python story_generator.py --llm-type transformer --model-name "model-name"

# LMStudio 사용
python story_generator.py --llm-type lmstudio --lmstudio-url "http://localhost:1234/v1"
```

### 기존 작업 이어서 하기
```bash
python story_generator.py --resume-from "2024-03-15_소설제목"
```

## 출력 형식

생성된 소설은 다음과 같은 구조로 저장됩니다:

```
YYYY-MM-DD_소설제목/
├── analysis.json      # 트렌드 분석 결과
├── structure.json     # 소설 구조
├── chapter_01.txt     # 1장
├── chapter_02.txt     # 2장
...
├── chapter_08.txt     # 8장
├── full_story.txt     # 전체 소설
└── generation.log     # 생성 로그
```

## 데이터베이스

- `stories.db`: 생성된 소설 저장
- `trends.db`: 수집된 트렌드 데이터 저장

## 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, 성능 향상)
- 충분한 디스크 공간 (모델 및 생성된 소설 저장용)

## 라이선스

MIT License
