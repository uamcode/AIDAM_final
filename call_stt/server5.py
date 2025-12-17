from flask import Flask, render_template
from flask_sock import Sock
import json
import base64
import audioop
import wave
from datetime import datetime
import os
import sys
import numpy as np
import torch
import librosa
import soundfile as sf

# 프로젝트 루트 경로 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Denoiser 경로 추가
denoiser_directory = os.path.join(BASE_DIR, 'src', 'denoiser')
sys.path.append(denoiser_directory)

from denoiser import pretrained
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import unicodedata

# Beam Search + LM 디코더 (pyctcdecode)
try:
    from pyctcdecode import build_ctcdecoder
    import kenlm
    HAS_PYCTCDECODE = True
except ImportError:
    HAS_PYCTCDECODE = False
    print("⚠️ pyctcdecode 또는 kenlm이 설치되지 않았습니다. Greedy 디코딩만 사용합니다.")

# 설정 파일 import
try:
    from config import (
        HTTP_SERVER_PORT,
        SAMPLE_RATE_INPUT,
        SAMPLE_RATE_TARGET,
        CHUNK_DURATION,
        DENOISER_MODEL_PATH,
        ASR_MODEL_PATH,
        KEYWORD_MODEL_PATH,
        RECORDINGS_DIR
    )
except ImportError:
    # config.py가 없는 경우 기본값 사용
    HTTP_SERVER_PORT = 5000
    SAMPLE_RATE_INPUT = 8000
    SAMPLE_RATE_TARGET = 16000
    CHUNK_DURATION = 2.0
    DENOISER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'denoiser.th')
    ASR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Conformer-CTC-BPE.nemo')
    KEYWORD_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'qwen3-1.7b')
    RECORDINGS_DIR = os.path.join(BASE_DIR, 'call_recordings')

app = Flask(__name__)
sock = Sock(app)

# 전역 변수로 모델 저장
denoiser_model = None
asr_model = None
keyword_model = None
keyword_tokenizer = None
device = None
ctc_decoder = None  # Beam Search + LM 디코더
USE_BEAM_SEARCH = False  # Beam Search 사용 여부
BEAM_WIDTH = 10  # Beam 크기

def log(msg, *args):
    print(f"Media WS: ", msg, *args)

def load_models():
    """서버 시작 시 모델 로드"""
    global denoiser_model, asr_model, keyword_model, keyword_tokenizer, device, ctc_decoder, USE_BEAM_SEARCH
    
    log("Loading models...")
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")
    
    # Denoiser 모델 로드
    try:
        import argparse
        denoiser_args = argparse.Namespace(
            dns64=False,
            dns48=False,
            master64=False,
            device=str(device),
            dry=0.04,
            model_path=DENOISER_MODEL_PATH
        )
        denoiser_model = pretrained.get_model(denoiser_args).to(device)
        denoiser_model.eval()
        log("✓ Denoiser model loaded successfully")
    except Exception as e:
        log(f"Warning: Could not load denoiser model: {e}")
        denoiser_model = None
    
    # ASR 모델 로드
    try:
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(ASR_MODEL_PATH, map_location=device)
        asr_model.eval()
        
        # Preprocessor 설정
        from omegaconf import OmegaConf
        import copy
        asr_cfg = copy.deepcopy(asr_model._cfg)
        OmegaConf.set_struct(asr_cfg.preprocessor, False)
        asr_cfg.preprocessor.dither = 0.0
        asr_cfg.preprocessor.pad_to = 0
        OmegaConf.set_struct(asr_cfg.preprocessor, True)
        asr_model.preprocessor = asr_model.from_config_dict(asr_cfg.preprocessor)
        
        if device.type == 'cuda':
            asr_model.cuda()
        
        log("✓ ASR model loaded successfully")
        
        # Vocabulary 로드 (pyctcdecode에 필요)
        vocab_list = None
        try:
            vocab_path = os.path.join(BASE_DIR, 'src', 'nemo_asr', 'tokenizer_spe_bpe_v2048', 'vocab.txt')
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_list = [line.strip() for line in f]
            log(f"✓ Loaded vocabulary: {len(vocab_list)} tokens")
        except Exception as e:
            log(f"Warning: Could not load vocabulary: {e}")
            vocab_list = None
        
        # Beam Search + LM 디코더 설정 (pyctcdecode 방식)
        if HAS_PYCTCDECODE and vocab_list:
            try:
                # KenLM 모델 경로 탐색 (우선순위 순)
                kenlm_paths = [
                    os.path.join(BASE_DIR, 'models', 'korean_4gram.binary'),  # 새로 학습한 모델
                    os.path.join(BASE_DIR, 'models', 'korean_4gram.arpa'),    # ARPA 버전
                ]
                
                kenlm_model_path = None
                for path in kenlm_paths:
                    if os.path.exists(path):
                        kenlm_model_path = path
                        log(f"✓ Found KenLM model: {kenlm_model_path}")
                        break
                
                if kenlm_model_path:
                    # KenLM 모델 로드
                    kenlm_model = kenlm.Model(kenlm_model_path)
                    log("✓ KenLM model loaded successfully")
                    
                    # pyctcdecode decoder 생성
                    ctc_decoder = build_ctcdecoder(
                        labels=vocab_list,
                        kenlm_model=kenlm_model,
                        alpha=0.5,  # LM weight (튜닝 가능)
                        beta=1.0,   # Word insertion bonus (튜닝 가능)
                    )
                    
                    USE_BEAM_SEARCH = True
                    log("✅ pyctcdecode + KenLM decoder configured successfully")
                    log(f"   - Method: pyctcdecode (Python-based, Windows compatible)")
                    log(f"   - Model: {os.path.basename(kenlm_model_path)}")
                    log(f"   - Vocabulary size: {len(vocab_list)}")
                    log(f"   - Alpha (LM weight): 0.5")
                    log(f"   - Beta (word bonus): 1.0")
                    log(f"   - Language model trained on 93,723 Korean sentences")
                else:
                    log("⚠ KenLM model not found in:")
                    for path in kenlm_paths:
                        log(f"   - {path}")
                    log("   Using Greedy decoding (without LM)")
                    USE_BEAM_SEARCH = False
                    
            except Exception as e:
                log(f"Warning: Could not configure KenLM: {e}")
                import traceback
                traceback.print_exc()
                USE_BEAM_SEARCH = False
        else:
            if not HAS_PYCTCDECODE:
                log("⚠ pyctcdecode not available. Install with: pip install pyctcdecode")
            if not vocab_list:
                log("⚠ Vocabulary not loaded")
            log("   Using Greedy decoding (without LM)")
            USE_BEAM_SEARCH = False
            
    except Exception as e:
        log(f"Warning: Could not load ASR model: {e}")
        asr_model = None
        ctc_decoder = None
        USE_BEAM_SEARCH = False
    
    # 키워드 추출 모델 로드 (Qwen3-1.7B)
    try:
        if os.path.exists(KEYWORD_MODEL_PATH):
            # 로컬 모델 사용
            keyword_model_path = KEYWORD_MODEL_PATH
            log(f"Loading keyword extraction model from local: {KEYWORD_MODEL_PATH}")
        else:
            # HuggingFace에서 다운로드
            keyword_model_path = "Qwen/Qwen3-1.7B"
            log(f"Local model not found. Downloading from HuggingFace: {keyword_model_path}")
        
        keyword_tokenizer = AutoTokenizer.from_pretrained(keyword_model_path)
        keyword_model = AutoModelForCausalLM.from_pretrained(
            keyword_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        log("✓ Keyword extraction model loaded successfully")
        log(f"  - Model source: {'Local' if os.path.exists(KEYWORD_MODEL_PATH) else 'HuggingFace'}")
        log(f"  - Model path: {keyword_model_path}")
    except Exception as e:
        log(f"Warning: Could not load keyword model: {e}")
        keyword_model = None
        keyword_tokenizer = None
    
    log("All models loaded and ready!")

@app.route("/", methods=["GET"])
def index():
    return "OK", 200

@app.route('/twiml', methods=['GET', 'POST'])
def return_twiml():
    print("POST TwiML")
    return render_template('streams.xml')

@sock.route("/stream")
def echo(ws):
    log("Connection accepted")
    count = 0
    has_seen_media = False
    
    # 화자별 이중 버퍼 구조
    buffers = {
        'inbound': {  # 고객
            'audio': [],           # 저장용 버퍼
            'processing': [],      # 실시간 처리용 버퍼
            'transcriptions': [],  # 전사 결과
            'keywords': []         # 추출된 키워드
        },
        'outbound': {  # 상담사
            'audio': [],
            'processing': [],
            'transcriptions': [],
            'keywords': []
        }
    }
    
    # 처리 파라미터
    CHUNK_SIZE = int(SAMPLE_RATE_INPUT * CHUNK_DURATION)  # 샘플 수
    
    # 화자 라벨 매핑
    speaker_labels = {
        'inbound': '고객',
        'outbound': '상담사'
    }
    
    while True:
        try:
            message = ws.receive()
            if message is None:
                log("No message received...")
                break
            
            data = json.loads(message)
            
            if data['event'] == "connected":
                log("Connected Message received")
                
            if data['event'] == "start":
                log("Start Message received")
                log("Starting real-time dual-track Denoise + STT processing...")
                log("Track: inbound (고객) / outbound (상담사)")
                
            if data['event'] == "media":
                if not has_seen_media:
                    log("Media messages received - processing started")
                    has_seen_media = True
                
                # track 필드로 화자 구분
                track = data['media'].get('track', 'inbound_track')
                
                # 디버깅: track 값 확인 (처음 몇 개만 출력)
                if count < 5:
                    log(f"DEBUG: Received track value: '{track}'")
                
                # track 값에 따라 화자 구분
                if 'inbound' in track.lower():
                    speaker = 'inbound'
                elif 'outbound' in track.lower():
                    speaker = 'outbound'
                else:
                    # 기본값은 inbound로 설정
                    speaker = 'inbound'
                    if count < 5:
                        log(f"WARNING: Unknown track value '{track}', defaulting to inbound")
                
                # base64 디코딩
                payload = data['media']['payload']
                audio_data = base64.b64decode(payload)
                
                # mu-law를 PCM으로 변환 (8bit mu-law -> 16bit PCM)
                pcm_data = audioop.ulaw2lin(audio_data, 2)
                
                # 해당 화자의 버퍼에 추가
                buffers[speaker]['audio'].append(pcm_data)
                buffers[speaker]['processing'].append(pcm_data)
                
                # 버퍼가 충분히 쌓이면 처리
                current_size = sum(len(chunk) for chunk in buffers[speaker]['processing'])
                if current_size >= CHUNK_SIZE * 2:  # 16-bit = 2 bytes per sample
                    # 실시간 처리
                    try:
                        transcription = process_audio_chunk(
                            buffers[speaker]['processing'], 
                            SAMPLE_RATE_INPUT, 
                            SAMPLE_RATE_TARGET
                        )
                        if transcription:
                            buffers[speaker]['transcriptions'].append(transcription)
                            log(f"[{speaker_labels[speaker]}] Transcription: {transcription}")
                            
                            # 키워드 추출
                            keywords = extract_keywords(transcription)
                            if keywords:
                                buffers[speaker]['keywords'].extend(keywords)
                                log(f"[{speaker_labels[speaker]}] 🔑 Keywords: {keywords}")
                    except Exception as e:
                        log(f"[{speaker_labels[speaker]}] Error processing chunk: {e}")
                    
                    # 처리용 버퍼 초기화
                    buffers[speaker]['processing'] = []
                
            if data['event'] == "closed":
                log("Closed Message received")
                break
                
            count += 1
            
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break

    log(f"Connection closed. Received a total of {count} messages")
    
    # 남은 버퍼 처리 (양쪽 화자 모두)
    for speaker in ['inbound', 'outbound']:
        if buffers[speaker]['processing']:
            try:
                transcription = process_audio_chunk(
                    buffers[speaker]['processing'], 
                    SAMPLE_RATE_INPUT, 
                    SAMPLE_RATE_TARGET
                )
                if transcription:
                    buffers[speaker]['transcriptions'].append(transcription)
                    log(f"[{speaker_labels[speaker]}] Final transcription: {transcription}")
                    
                    # 마지막 키워드 추출
                    keywords = extract_keywords(transcription)
                    if keywords:
                        buffers[speaker]['keywords'].extend(keywords)
                        log(f"[{speaker_labels[speaker]}] 🔑 Keywords: {keywords}")
            except Exception as e:
                log(f"[{speaker_labels[speaker]}] Error processing final chunk: {e}")
    
    # 화자별 파일 저장
    save_dual_track_results(buffers, speaker_labels)

def extract_keywords(text):
    """
    Qwen3-1.7B를 사용하여 한국어 문장에서 키워드 추출
    
    Args:
        text: 키워드를 추출할 한국어 문장
        
    Returns:
        list: 추출된 키워드 리스트
    """
    if not text or not text.strip() or keyword_model is None or keyword_tokenizer is None:
        return []
    
    try:
        system_prompt = (
            "당신은 한국어 한 문장에서 검색/분류에 유의미한 핵심 키워드만 추출합니다.\n"
            "규칙:\n"
            "- 키워드는 고유명사, 기술명, 개념, 객체 중심\n"
            "- 감정, 추임새, 일반적인 말은 제외\n"
            "- 키워드가 필요 없으면 반드시 빈 배열을 반환\n"
            "- 출력은 반드시 JSON 한 줄로만: {\"keywords\": [..]}\n"
            "- 추론 과정, 설명, 추가 문장 금지\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"문장: {text}"}
        ]

        text_input = keyword_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = keyword_tokenizer([text_input], return_tensors="pt").to(keyword_model.device)

        generated_ids = keyword_model.generate(
            **model_inputs,
            max_new_tokens=128,
            min_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=keyword_tokenizer.eos_token_id
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        decoded_text = keyword_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # think 태그 제거
        decoded_text = re.sub(r'<think>.*?</think>', '', decoded_text, flags=re.DOTALL).strip()

        # JSON 추출
        m = re.search(r'\{.*\}', decoded_text, flags=re.DOTALL)
        if not m:
            return []

        result = json.loads(m.group(0))
        return result.get('keywords', [])
        
    except Exception as e:
        log(f"Error in extract_keywords: {e}")
        return []

def process_audio_chunk(buffer, input_sr, target_sr):
    """오디오 청크를 Denoise + STT 처리"""
    try:
        # 버퍼를 numpy 배열로 변환
        audio_data = b''.join(buffer)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 리샘플링 (8kHz -> 16kHz)
        if input_sr != target_sr:
            audio_resampled = librosa.resample(audio_np, orig_sr=input_sr, target_sr=target_sr)
        else:
            audio_resampled = audio_np
        
        # Denoise
        if denoiser_model is not None:
            audio_tensor = torch.tensor(audio_resampled).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                audio_denoised = denoiser_model(audio_tensor)
            audio_denoised = audio_denoised.squeeze().cpu().numpy()
        else:
            audio_denoised = audio_resampled
        
        # STT
        if asr_model is not None:
            with torch.no_grad():
                if USE_BEAM_SEARCH and ctc_decoder is not None:
                    # pyctcdecode 사용: logits 추출 후 디코딩
                    try:
                        # audio를 tensor로 변환
                        audio_tensor = torch.tensor(audio_denoised).unsqueeze(0).to(device)
                        audio_length = torch.tensor([audio_tensor.shape[1]]).to(device)
                        
                        # NeMo 모델에서 logits 추출
                        processed_signal, processed_signal_length = asr_model.preprocessor(
                            input_signal=audio_tensor, length=audio_length
                        )
                        if asr_model.spec_augmentation is not None and asr_model.training:
                            processed_signal = asr_model.spec_augmentation(
                                input_spec=processed_signal, length=processed_signal_length
                            )
                        encoded, encoded_len = asr_model.encoder(
                            audio_signal=processed_signal, length=processed_signal_length
                        )
                        log_probs = asr_model.decoder(encoder_output=encoded)
                        
                        # pyctcdecode로 디코딩
                        # log_probs shape: [batch=1, time, vocab]
                        logits_np = log_probs[0].cpu().numpy()  # [time, vocab]
                        text = ctc_decoder.decode(logits_np)
                        
                        if text:
                            text = unicodedata.normalize('NFC', text)
                            return text.strip()
                    except Exception as e:
                        log(f"pyctcdecode failed, falling back to Greedy: {e}")
                        # Beam Search 실패 시 Greedy 디코딩으로 폴백
                
                # Greedy 디코딩 (기존 방식 또는 폴백)
                transcription = asr_model.transcribe([audio_denoised], batch_size=1)
                if transcription and len(transcription) > 0:
                    # Hypothesis 객체에서 text 속성 추출
                    result = transcription[0]
                    if hasattr(result, 'text'):
                        text = result.text
                    else:
                        text = str(result)
                    
                    if text:
                        text = unicodedata.normalize('NFC', text)
                        return text.strip()
        
        return None
        
    except Exception as e:
        log(f"Error in process_audio_chunk: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_dual_track_results(buffers, speaker_labels):
    """화자별 오디오, 전사 결과, 키워드 저장"""
    try:
        # 타임스탬프 기반 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        
        # 파일명 매핑
        file_suffixes = {
            'inbound': 'customer',   # 고객
            'outbound': 'agent'      # 상담사
        }
        
        total_duration = 0
        stats = {}
        
        # 각 화자별로 파일 저장
        for speaker in ['inbound', 'outbound']:
            suffix = file_suffixes[speaker]
            label = speaker_labels[speaker]
            
            # 오디오 데이터가 있는 경우에만 저장
            if buffers[speaker]['audio']:
                # WAV 파일 저장
                audio_filename = os.path.join(RECORDINGS_DIR, f"call_{timestamp}_{suffix}.wav")
                audio_data = b''.join(buffers[speaker]['audio'])
                
                with wave.open(audio_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(SAMPLE_RATE_INPUT)
                    wav_file.writeframes(audio_data)
                
                duration = len(audio_data) / (SAMPLE_RATE_INPUT * 2)
                total_duration = max(total_duration, duration)
                log(f"[{label}] Audio saved: {audio_filename}")
                log(f"[{label}] Duration: {duration:.2f} seconds")
                
                # 통계 저장
                stats[speaker] = {
                    'audio_file': audio_filename,
                    'duration': duration,
                    'chunks': len(buffers[speaker]['audio'])
                }
            
            # 전사 결과 및 키워드 저장
            if buffers[speaker]['transcriptions']:
                txt_filename = os.path.join(RECORDINGS_DIR, f"call_{timestamp}_{suffix}.txt")
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(f"=== 화자: {label} ({speaker.capitalize()} Track) ===\n\n")
                    
                    f.write("=== Real-time Transcription Results ===\n\n")
                    for i, trans in enumerate(buffers[speaker]['transcriptions'], 1):
                        f.write(f"[Chunk {i}] {trans}\n")
                    
                    f.write("\n=== Full Transcription ===\n")
                    full_text = " ".join(buffers[speaker]['transcriptions'])
                    f.write(full_text)
                    
                    # 키워드 추가
                    if buffers[speaker]['keywords']:
                        f.write("\n\n=== Extracted Keywords ===\n")
                        unique_keywords = list(set(buffers[speaker]['keywords']))
                        f.write(f"Total unique keywords: {len(unique_keywords)}\n")
                        f.write(f"Keywords: {', '.join(unique_keywords)}\n")
                
                log(f"[{label}] Transcription saved: {txt_filename}")
                log(f"[{label}] Total chunks transcribed: {len(buffers[speaker]['transcriptions'])}")
                
                if buffers[speaker]['keywords']:
                    unique_keywords = list(set(buffers[speaker]['keywords']))
                    log(f"[{label}] Extracted {len(unique_keywords)} unique keywords: {unique_keywords}")
                
                # 통계 업데이트
                if speaker in stats:
                    stats[speaker]['txt_file'] = txt_filename
                    stats[speaker]['transcriptions'] = len(buffers[speaker]['transcriptions'])
                    stats[speaker]['keywords'] = len(unique_keywords) if buffers[speaker]['keywords'] else 0
        
        # 전체 통화 요약
        log("\n=== Call Summary ===")
        log(f"Total call duration: {total_duration:.2f} seconds")
        for speaker in ['inbound', 'outbound']:
            if speaker in stats:
                label = speaker_labels[speaker]
                log(f"[{label}] Chunks: {stats[speaker].get('chunks', 0)}, "
                    f"Transcriptions: {stats[speaker].get('transcriptions', 0)}, "
                    f"Keywords: {stats[speaker].get('keywords', 0)}")
        
    except Exception as e:
        log(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 모델 로드
    load_models()
    
    # 서버 시작
    log("Starting server...")
    log(f"Server will listen on port {HTTP_SERVER_PORT}")
    app.run(host='0.0.0.0', port=HTTP_SERVER_PORT, debug=True, use_reloader=False)
