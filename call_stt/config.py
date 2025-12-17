# Twilio 쌍방향 통화 스트리밍 설정

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 환경변수에서 로드 (민감한 정보)
# .env 파일에 설정하거나, 기본값 사용
AGENT_PHONE_NUMBER = os.getenv('AGENT_PHONE_NUMBER', '+821012345678')
WEBSOCKET_URL = os.getenv('WEBSOCKET_URL', 'wss://your-ngrok-url.ngrok-free.dev/stream')

# 서버 설정
HTTP_SERVER_PORT = int(os.getenv('HTTP_SERVER_PORT', '5000'))

# 오디오 처리 설정
SAMPLE_RATE_INPUT = 8000    # Twilio 입력 샘플레이트
SAMPLE_RATE_TARGET = 16000  # ASR 모델 요구 샘플레이트
CHUNK_DURATION = 2.0        # 처리 단위 (초)

# 모델 경로 (프로젝트 루트 기준 상대 경로)
DENOISER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'denoiser.th')
ASR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Conformer-CTC-BPE.nemo')
KEYWORD_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'qwen3-1.7b')

# KenLM 언어 모델 경로 (선택사항)
# 파일이 있으면 Beam Search + LM, 없으면 Beam Search만 사용
KENLM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'korean_lm.bin')
# 또는 42MARU 모델 사용: os.path.join(BASE_DIR, 'models', 'korean_lm_42maru.bin')

# 녹음 파일 저장 경로
RECORDINGS_DIR = os.path.join(BASE_DIR, 'call_recordings')
