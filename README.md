# 실시간 상담 통화 STT 시스템 (배포용)

실시간으로 상담 전화를 받아 음성을 텍스트로 변환하고 키워드를 추출하는 시스템입니다.

## 주요 기능

- ✅ **Twilio 쌍방향 통화 스트리밍**: 고객과 상담사 음성을 동시에 처리
- ✅ **실시간 음성 노이즈 제거**: Facebook Denoiser 모델 사용
- ✅ **한국어 음성 인식**: NeMo Conformer-CTC-BPE 모델
- ✅ **언어 모델 기반 디코딩**: KenLM + pyctcdecode로 정확도 향상
- ✅ **자동 키워드 추출**: Qwen3-1.7B 모델로 핵심 키워드 추출
- ✅ **화자별 분리 저장**: 고객과 상담사의 음성/텍스트를 각각 저장

## 시스템 요구사항

- **Python**: 3.8 - 3.10
- **RAM**: 16GB 이상 권장
- **GPU**: CUDA 지원 GPU (선택사항, CPU로도 동작)
- **저장 공간**: 최소 10GB (모델 파일 포함)

## 설치 가이드

### 1. 저장소 클론

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Python 환경 설정

```bash
# Conda 환경 생성 (권장)
conda create -n call_stt python=3.10
conda activate call_stt

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements2.txt
```

### 4. GPU 사용 시 PyTorch 재설치

자신의 CUDA 버전을 확인하고 적절한 PyTorch를 설치하세요:

```bash
# CUDA 버전 확인
nvcc --version
# 또는
nvidia-smi

# 기존 PyTorch 제거
pip uninstall -y torch torchvision torchaudio

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU만 사용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. 시스템 라이브러리 설치 (Linux/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg portaudio19-dev
```

### 6. 모델 파일 다운로드

다음 모델 파일들을 `models/` 폴더에 배치하세요:

| 파일명 | 설명 | 크기 | 다운로드 링크 |
|--------|------|------|---------------|
| `denoiser.th` | 음성 노이즈 제거 모델 | ~80MB | [Google Drive](https://drive.google.com/drive/folders/1CmP71JljIw9lsz1Ai6dnGpWkCEahB8db?usp=sharing) |
| `Conformer-CTC-BPE.nemo` | 한국어 ASR 모델 | ~120MB | [Google Drive](https://drive.google.com/drive/folders/1CmP71JljIw9lsz1Ai6dnGpWkCEahB8db?usp=sharing) |
| `korean_4gram.binary` | KenLM 언어 모델 | ~500MB | [Google Drive](https://drive.google.com/drive/folders/1CmP71JljIw9lsz1Ai6dnGpWkCEahB8db?usp=sharing) |
| `qwen3-1.7b/` | 키워드 추출 모델 | ~3.5GB | [HuggingFace](https://huggingface.co/Qwen/Qwen3-1.7B) |

**폴더 구조:**
```
models/
├── denoiser.th
├── Conformer-CTC-BPE.nemo
├── korean_4gram.binary
└── qwen3-1.7b/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

**Qwen3 모델 자동 다운로드:**
```bash
# HuggingFace에서 자동 다운로드 (인터넷 필요)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', cache_dir='./models/qwen3-1.7b'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', cache_dir='./models/qwen3-1.7b')"
```

## 설정

### 1. 설정 파일 수정

`call_ex/config.py` 파일을 열어 다음 항목을 수정하세요:

```python
# 상담사 전화번호 (E.164 형식)
AGENT_PHONE_NUMBER = "+821012345678"

# 서버 포트
HTTP_SERVER_PORT = 5000

# ngrok URL (ngrok 실행 후 생성된 URL로 변경)
WEBSOCKET_URL = "wss://your-ngrok-url.ngrok-free.dev/stream"

# 모델 경로는 기본값 사용 (models/ 폴더에 파일 배치 시)
```

### 2. ngrok 설정 (외부 접근용)

Twilio가 로컬 서버에 접근하려면 ngrok이 필요합니다:

```bash
# ngrok 설치
# https://ngrok.com/download

# ngrok 실행
ngrok http 5000
```

생성된 URL(예: `https://abc123.ngrok-free.dev`)을 `config.py`의 `WEBSOCKET_URL`에 설정:
```python
WEBSOCKET_URL = "wss://abc123.ngrok-free.dev/stream"
```

### 3. Twilio 설정

1. [Twilio Console](https://console.twilio.com/)에 로그인
2. 전화번호 구매 (Phone Numbers > Buy a number)
3. 구매한 번호 설정:
   - Voice Configuration > A Call Comes In
   - Webhook URL: `https://your-ngrok-url.ngrok-free.dev/twiml`
   - HTTP Method: POST

## 실행

```bash
python call_ex/server5.py
```

**실행 시 출력 예시:**
```
Media WS:  Loading models...
Media WS:  Using device: cuda
Media WS:  ✓ Denoiser model loaded successfully
Media WS:  ✓ ASR model loaded successfully
Media WS:  ✓ Loaded vocabulary: 2048 tokens
Media WS:  ✓ KenLM model loaded successfully
Media WS:  ✅ pyctcdecode + KenLM decoder configured successfully
Media WS:  ✓ Keyword extraction model loaded successfully
Media WS:  All models loaded and ready!
Media WS:  Starting server...
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

## 사용 방법

1. 서버가 실행되면 Twilio 번호로 전화를 겁니다
2. 자동으로 상담사 번호(`AGENT_PHONE_NUMBER`)로 연결됩니다
3. 통화 중 실시간으로 STT가 수행되며 콘솔에 출력됩니다:
   ```
   [고객] Transcription: 안녕하세요 상담 문의드립니다
   [고객] 🔑 Keywords: ['상담', '문의']
   [상담사] Transcription: 네 무엇을 도와드릴까요
   ```
4. 통화 종료 시 `call_recordings/` 폴더에 결과가 저장됩니다:
   - `call_YYYYMMDD_HHMMSS_customer.wav` - 고객 음성
   - `call_YYYYMMDD_HHMMSS_customer.txt` - 고객 전사 결과 + 키워드
   - `call_YYYYMMDD_HHMMSS_agent.wav` - 상담사 음성
   - `call_YYYYMMDD_HHMMSS_agent.txt` - 상담사 전사 결과 + 키워드

## 프로젝트 구조

```
.
├── call_ex/
│   ├── server5.py              # 메인 서버
│   ├── config.py               # 설정 파일
│   └── templates/
│       └── streams.xml         # Twilio 스트리밍 템플릿
├── src/
│   ├── denoiser/               # 음성 노이즈 제거 모듈
│   └── nemo_asr/
│       └── tokenizer_spe_bpe_v2048/
│           └── vocab.txt       # ASR vocabulary
├── models/                     # 모델 파일 (Git 제외)
│   ├── denoiser.th
│   ├── Conformer-CTC-BPE.nemo
│   ├── korean_4gram.binary
│   └── qwen3-1.7b/
├── call_recordings/            # 통화 녹음 저장 폴더
├── requirements2.txt           # 의존성 패키지
└── README2.md                  # 이 문서
```

## 문제 해결

### kenlm 설치 실패 (Windows)

```bash
# 소스에서 직접 설치
pip install https://github.com/kpu/kenlm/archive/master.zip

# 또는 Microsoft Visual C++ 14.0+ 설치 후 재시도
# https://visualstudio.microsoft.com/downloads/
```

### CUDA Out of Memory 오류

```python
# config.py에서 chunk 크기 줄이기
CHUNK_DURATION = 1.0  # 기본값 2.0에서 감소

# 또는 CPU 모드 사용
# server5.py 77번째 줄 수정:
device = torch.device('cpu')
```

### Twilio 연결 안됨

1. ngrok이 실행 중인지 확인
2. ngrok URL이 `config.py`와 Twilio Console에 정확히 입력되었는지 확인
3. 방화벽에서 5000 포트 허용 확인

### 모델 로딩 실패

```bash
# 모델 파일 경로 확인
ls -lh models/

# 파일 권한 확인 (Linux)
chmod 644 models/*.th models/*.nemo models/*.binary
```

## 성능 최적화

### GPU 사용 시
- RTX 3060 (12GB): 처리 시간 ~50ms/chunk
- RTX 4090 (24GB): 처리 시간 ~30ms/chunk

### CPU 사용 시
- Intel i7-12700: 처리 시간 ~200ms/chunk
- 실시간 처리 가능하나 GPU 대비 느림

## 기술 스택

- **Denoising**: Facebook Denoiser (DNS48 기반)
- **ASR**: NVIDIA NeMo Conformer-CTC-BPE
- **Language Model**: KenLM (4-gram)
- **Keyword Extraction**: Qwen3-1.7B
- **Web Framework**: Flask + Flask-Sock
- **Telephony**: Twilio Media Streams

## 모델 정보

### ASR 모델 성능
- 학습 데이터: 13,946시간 (10,916,423 샘플)
- WER (Word Error Rate): ~8.5%
- 실시간 처리 속도: RTF < 0.1 (GPU)

### 학습 데이터셋
- 고객응대음성, 한국어 음성, 한국인 대화 음성
- 자유대화음성, 복지 분야 콜센터 상담데이터
- 차량내 대화 데이터, 명령어 음성

## 라이선스

이 프로젝트는 다음 오픈소스 프로젝트를 기반으로 합니다:
- [Facebook Denoiser](https://github.com/facebookresearch/denoiser) (MIT License)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Apache 2.0)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0)

## 참고 자료

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams)
- [KenLM Language Model](https://github.com/kpu/kenlm)

## 지원

문제가 발생하거나 질문이 있으시면 Issue를 등록해주세요.
