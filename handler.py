import runpod
import torch
import soundfile as sf
import base64, os, tempfile, json
import numpy as np
import importlib
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
import TTS.tts.layers.xtts.tokenizer as _tok_module

LOCAL_DIR_CV = '/workspace/xtts_luganda'
LOCAL_DIR_UG = '/workspace/xtts_uganda'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Startup diagnostics ────────────────────────────────────────────────────
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def verify_and_patch(local_dir, add_lg=False, add_all=False):
    vocab_path = f'{local_dir}/vocab.json'
    config_path = f'{local_dir}/config.json'
    
    for f in ['config.json', 'best_model.pth', 'vocab.json']:
        path = os.path.join(local_dir, f)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"{'OK' if exists else 'MISSING'}: {path} ({size} bytes)")
        if not exists:
            raise RuntimeError(f"MISSING required file: {path}")

    # Patch vocab.json
    print(f"Patching vocab.json in {local_dir}...")
    with open(vocab_path, encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    is_hf = 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']
    target_vocab = tokenizer_data['model']['vocab'] if is_hf else tokenizer_data

    tokens_to_add = []
    if add_lg:
        tokens_to_add.append('[lg]')
    if add_all:
        tokens_to_add.extend(['[lg]', '[ach]', '[mas]', '[nyn]', '[sog]', '[en-ug]'])
        
    added_any = False
    for lang_token in tokens_to_add:
        if lang_token not in target_vocab:
            numeric_ids = set()
            for v in target_vocab.values():
                try:
                    numeric_ids.add(int(v))
                except (ValueError, TypeError):
                    pass
            new_id = (max(numeric_ids) + 1) if numeric_ids else 0
            target_vocab[lang_token] = new_id
            tokenizer_data.setdefault('added_tokens', [])
            if not any(t.get('content') == lang_token for t in tokenizer_data['added_tokens']):
                tokenizer_data['added_tokens'].append({
                    'id': new_id, 'content': lang_token,
                    'single_word': False, 'lstrip': False, 'rstrip': False,
                    'normalized': False, 'special': True,
                })
            print(f"Added {lang_token} to {local_dir}/vocab.json (id={new_id})")
            added_any = True

    if added_any:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=4, ensure_ascii=False)

    # Patch config.json
    print(f"Patching config.json in {local_dir}...")
    with open(config_path) as f:
        config = json.load(f)

    langs_to_add = []
    if add_lg:
        langs_to_add.append('lg')
    if add_all:
        langs_to_add.extend(['lg', 'ach', 'mas', 'nyn', 'sog', 'en-ug'])

    config_langs = config.setdefault('languages', [])
    added_lang = False
    for l in langs_to_add:
        if l not in config_langs:
            config_langs.append(l)
            added_lang = True
            
    config.setdefault('model_args', {})['tokenizer_file'] = vocab_path

    if added_lang:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

verify_and_patch(LOCAL_DIR_CV, add_lg=True)
verify_and_patch(LOCAL_DIR_UG, add_all=True)

# ── Patch tokenizer preprocess — map custom langs → en text cleaner ──────────────
print("Patching tokenizer...")
importlib.reload(_tok_module)
_true_preprocess = _tok_module.VoiceBpeTokenizer.preprocess_text

def _patched_preprocess(self, txt, lang):
    if lang in ["en-ug", "lg", "ach", "mas", "nyn", "sog"]:
        lang = "en"
    return _true_preprocess(self, txt, lang)

VoiceBpeTokenizer.preprocess_text = _patched_preprocess

# ── Load models ─────────────────────────────────────────────────────────────
print("Loading CV model (Luganda)...")
xtts_config_cv = XttsConfig()
xtts_config_cv.load_json(f'{LOCAL_DIR_CV}/config.json')
model_cv = Xtts.init_from_config(xtts_config_cv)
model_cv.load_checkpoint(
    xtts_config_cv,
    checkpoint_path=f'{LOCAL_DIR_CV}/best_model.pth',
    vocab_path=f'{LOCAL_DIR_CV}/vocab.json',
    eval=True,
    use_deepspeed=False,
)
model_cv = model_cv.to(DEVICE)
model_cv.tokenizer.char_limits['lg'] = model_cv.tokenizer.char_limits['en']

print("Loading UG-Lang model (Other languages)...")
xtts_config_ug = XttsConfig()
xtts_config_ug.load_json(f'{LOCAL_DIR_UG}/config.json')
model_ug = Xtts.init_from_config(xtts_config_ug)
model_ug.load_checkpoint(
    xtts_config_ug,
    checkpoint_path=f'{LOCAL_DIR_UG}/best_model.pth',
    vocab_path=f'{LOCAL_DIR_UG}/vocab.json',
    eval=True,
    use_deepspeed=False,
)
model_ug = model_ug.to(DEVICE)
for l in ["en-ug", "lg", "ach", "mas", "nyn", "sog"]:
    model_ug.tokenizer.char_limits[l] = model_ug.tokenizer.char_limits['en']

print("Models ready!")

# ── Audio helpers ──────────────────────────────────────────────────────────
def crossfade_join(wavs, fade_samples=2000):
    if len(wavs) == 0: return np.array([])
    if len(wavs) == 1: return wavs[0]
    result = wavs[0]
    for nxt in wavs[1:]:
        fade_len = min(fade_samples, len(result), len(nxt))
        fade_out = np.linspace(1, 0, fade_len)
        fade_in  = np.linspace(0, 1, fade_len)
        overlap  = result[-fade_len:] * fade_out + nxt[:fade_len] * fade_in
        result   = np.concatenate([result[:-fade_len], overlap, nxt[fade_len:]])
    return result

def trim_leading_breath(wav, sample_rate, threshold=0.15, frame_length=1024, hop_length=256, pad_samples=500):
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    rms = np.array([
        np.sqrt(np.mean(wav[i * hop_length : i * hop_length + frame_length] ** 2))
        for i in range(num_frames)
    ])
    active = np.where(rms > threshold)[0]
    if len(active) == 0: return wav
    start_sample = max(0, active[0] * hop_length - pad_samples)
    return wav[start_sample:]

# ── Handler ────────────────────────────────────────────────────────────────
def handler(job):
    inputs = job["input"]
    text = inputs.get("text", "")
    lang = inputs.get("language", "lug")
    
    if not text:
        return {"error": "text is required"}

    # Map input language to XTTS internal language code
    lang_to_xtts = {
        'lug': 'lg',
        'lg': 'lg',
        'ach': 'ach',
        'nyn': 'nyn',
        'sog': 'sog',
        'mas': 'mas',
        'eng-ug': 'en-ug',
        'en-ug': 'en-ug'
    }
    
    xtts_lang = lang_to_xtts.get(lang, lang)

    # Select model based on language
    if xtts_lang == 'lg':
        model = model_cv
        xtts_config = xtts_config_cv
    else:
        model = model_ug
        xtts_config = xtts_config_ug

    # Expected reference audio filenames based on notebook
    audio_filename_map = {
        'lg': 'speaker_lug.wav',
        'ach': 'speaker_ach.wav',
        'nyn': 'speaker_nyn.wav',
        'sog': 'speaker_sog.wav',
        'mas': 'speaker_mas.wav',
        'en-ug': 'speaker_eng-ug.wav'
    }
    
    filename = audio_filename_map.get(xtts_lang, f"speaker_{lang}.wav")
    speaker_path = os.path.join("ref", filename)
    
    if not os.path.exists(speaker_path):
        return {"error": f"Reference audio not found for language {lang}: {speaker_path}"}

    try:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[speaker_path],
            gpt_cond_len=xtts_config.gpt_cond_len,
            max_ref_length=xtts_config.max_ref_len,
            sound_norm_refs=xtts_config.sound_norm_refs,
        )

        # Split text into chunks of 20 words
        CHUNK_SIZE = 20
        words = text.split()
        sentences = []
        for j in range(0, len(words), CHUNK_SIZE):
            sentences.append(" ".join(words[j:j + CHUNK_SIZE]))

        all_wavs = []
        for sentence in sentences:
            inference_text = sentence.rstrip('.,!?')
            if not inference_text:
                continue
            out = model.inference(
                text=inference_text,
                language=xtts_lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.75,
                top_p=0.9,
                top_k=50,
                repetition_penalty=5.0,
                enable_text_splitting=False,
            )
            all_wavs.append(trim_leading_breath(np.array(out["wav"]), xtts_config.audio.output_sample_rate))

        final_wav = crossfade_join(all_wavs, fade_samples=2000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, final_wav, xtts_config.audio.output_sample_rate)
            with open(f.name, 'rb') as audio_file:
                audio_b64 = base64.b64encode(audio_file.read()).decode('utf-8')

        os.unlink(f.name)
        return {"audio_base64": audio_b64, "language": lang, "text": text}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})