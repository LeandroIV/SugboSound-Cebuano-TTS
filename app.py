"""
SugboSound — Cebuano Text-to-Speech Demo
=========================================
Gradio app for VITS (LoRA & Full Fine-tune) and Glow-TTS models.
Deploy on Hugging Face Spaces or run locally.
"""

import os
import re
import json
import traceback
import torch
import numpy as np
import librosa
import gradio as gr
import torch.nn as nn
from huggingface_hub import hf_hub_download
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.vocoder.models.hifigan_generator import HifiganGenerator

HF_MODEL_REPO = "LeandroIV/sugbosound-cebuano-tts-vits"

# ---------------------------------------------------------------------------
# LoRA module (must match training)
# ---------------------------------------------------------------------------

class LoRALinearConv1D(nn.Module):
    def __init__(self, original_conv, r=8, alpha=1.0):
        super().__init__()
        self.original = original_conv
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Conv1d(original_conv.in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(r, original_conv.out_channels, kernel_size=1, bias=False)
        self.scaling = alpha / r
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.original(x)
        lora = self.scaling * self.lora_B(self.lora_A(x))
        if lora.size(-1) != base.size(-1):
            tgt = base.size(-1)
            cur = lora.size(-1)
            if cur > tgt:
                start = (cur - tgt) // 2
                lora = lora[:, :, start:start + tgt]
            else:
                pad = tgt - cur
                lora = nn.functional.pad(lora, (0, pad))
        return base + lora


def inject_lora_into_vits(model, r=8):
    device = next(model.parameters()).device
    for name, module in list(model.named_modules()):
        t = str(type(module))
        if any(k in t for k in ["RelativePositionMultiHeadAttention", "MultiHeadAttention", "Attention"]):
            for attr in ("conv_q", "conv_k", "conv_v"):
                if hasattr(module, attr):
                    orig = getattr(module, attr)
                    wrapper = LoRALinearConv1D(orig, r=r).to(device)
                    setattr(module, attr, wrapper)
        if "text_encoder" in name and "ffn" in name.lower():
            for attr in ("conv_1", "conv_2"):
                if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Conv1d):
                    orig = getattr(module, attr)
                    wrapper = LoRALinearConv1D(orig, r=r).to(device)
                    setattr(module, attr, wrapper)


def inject_lora_into_glow_tts(model, r=8):
    """Inject LoRA into Glow-TTS: attention + FFN + decoder flow layers."""
    device = next(model.parameters()).device
    injected = 0
    for name, module in list(model.named_modules()):
        t = str(type(module))
        if any(k in t for k in ["RelativePositionMultiHeadAttention", "MultiHeadAttention", "Attention"]):
            for attr in ("conv_q", "conv_k", "conv_v"):
                if hasattr(module, attr):
                    orig = getattr(module, attr)
                    setattr(module, attr, LoRALinearConv1D(orig, r=r).to(device))
                    injected += 1
        if "encoder" in name and "ffn" in name.lower():
            for attr in ("conv_1", "conv_2"):
                if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Conv1d):
                    orig = getattr(module, attr)
                    setattr(module, attr, LoRALinearConv1D(orig, r=r).to(device))
                    injected += 1
        # Extended: decoder flow layers
        if "decoder" in name and isinstance(module, nn.Conv1d):
            if any(skip in name for skip in ["skip_connection", "conv", "coupling"]):
                parent_name = ".".join(name.split(".")[:-1])
                module_attr = name.split(".")[-1]
                parent_module = model
                for a in parent_name.split("."):
                    if a:
                        parent_module = getattr(parent_module, a, None)
                        if parent_module is None:
                            break
                if parent_module is not None:
                    try:
                        setattr(parent_module, module_attr, LoRALinearConv1D(module, r=r).to(device))
                        injected += 1
                    except Exception:
                        pass
    # Fallback if no injection points found
    if injected == 0:
        for name, module in list(model.named_modules()):
            for attr in ("conv_q", "conv_k", "conv_v"):
                if hasattr(module, attr):
                    orig = getattr(module, attr)
                    setattr(module, attr, LoRALinearConv1D(orig, r=r).to(device))
                    injected += 1
    print(f"[LoRA] Injected {injected} LoRA adapters into Glow-TTS")
    return injected


def load_lora_adapters(model, lora_state):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearConv1D):
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key in lora_state:
                module.lora_A.weight.data.copy_(lora_state[a_key])
            if b_key in lora_state:
                module.lora_B.weight.data.copy_(lora_state[b_key])


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _find_pth(directory, prefix="best_model"):
    """Find a .pth file in directory, accepting best_model.pth or best_model_#.pth."""
    if not os.path.isdir(directory):
        return None
    candidates = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pth")]
    if not candidates:
        return None
    exact = f"{prefix}.pth"
    if exact in candidates:
        return os.path.join(directory, exact)
    return os.path.join(directory, max(candidates, key=lambda f: os.path.getmtime(os.path.join(directory, f))))


# VITS paths
VITS_DIR = os.path.join(MODEL_DIR, "vits")
VITS_STAGE_A_LORA_DIR = os.path.join(VITS_DIR, "stage_a_lora")
VITS_STAGE_A_FFT_DIR = os.path.join(VITS_DIR, "stage_a_fft")
VITS_LORA_DIR = os.path.join(VITS_DIR, "stage_b_lora")
VITS_FFT_DIR = os.path.join(VITS_DIR, "stage_b_fft")


def ensure_stage_a_lora_files():
    """Download VITS Stage A LoRA checkpoint from Hugging Face Hub on first run."""
    os.makedirs(VITS_STAGE_A_LORA_DIR, exist_ok=True)
    targets = {
        "best_model.pth": os.path.join(VITS_STAGE_A_LORA_DIR, "best_model.pth"),
        "config.json":    os.path.join(VITS_STAGE_A_LORA_DIR, "config.json"),
    }
    for filename, dest in targets.items():
        if os.path.exists(dest):
            continue
        print(f"[HF] Downloading {filename} from {HF_MODEL_REPO} ...")
        cached = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
        try:
            os.replace(cached, dest)
        except OSError:
            import shutil
            shutil.copy2(cached, dest)
        print(f"[HF] Saved to {dest}")

# Glow-TTS paths
GLOW_DIR = os.path.join(MODEL_DIR, "glow_tts")
GLOW_STAGE_A_LORA_DIR = os.path.join(GLOW_DIR, "stage_a_lora")
GLOW_STAGE_A_FFT_DIR = os.path.join(GLOW_DIR, "stage_a_fft")
GLOW_STAGE_B_LORA_DIR = os.path.join(GLOW_DIR, "stage_b_lora")
GLOW_STAGE_B_FFT_DIR = os.path.join(GLOW_DIR, "stage_b_fft")

# HiFi-GAN vocoder for Glow-TTS
VOCODER_DIR = os.path.join(MODEL_DIR, "vocoder")

# Volume boost for specific models
VOLUME_BOOST = {
    "VITS Stage B + LoRA": 1.65,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 22050
LORA_RANK = 8

# ---------------------------------------------------------------------------
# Model cache (lazy loading)
# ---------------------------------------------------------------------------
_model_cache = {}


def _build_vits_model(config_path):
    """Build VITS model from config."""
    if not os.path.exists(config_path):
        return None, None, None, None
    config = VitsConfig()
    config.load_json(config_path)
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer)
    return model, ap, tokenizer, config


def _build_glow_tts_model(config_path):
    """Build Glow-TTS model from config."""
    if not os.path.exists(config_path):
        return None, None, None, None
    config = GlowTTSConfig()
    config.load_json(config_path)
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = GlowTTS(config, ap, tokenizer)
    return model, ap, tokenizer, config


# --- VITS loaders ---

def get_vits_stage_a_lora():
    if "vits_stage_a_lora" in _model_cache:
        return _model_cache["vits_stage_a_lora"]
    model, ap, tokenizer, config = _build_vits_model(os.path.join(VITS_STAGE_A_LORA_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(VITS_STAGE_A_LORA_DIR)
    if pth is None:
        return None, None, None
    ckpt = torch.load(pth, map_location="cpu")
    rank = _detect_lora_rank(ckpt["model"])
    print(f"[VITS Stage A LoRA] Detected rank: {rank}")
    model = model.to(DEVICE)
    inject_lora_into_vits(model, r=rank)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["vits_stage_a_lora"] = (model, tokenizer, ap)
    return model, tokenizer, ap


def get_vits_stage_a_fft():
    if "vits_stage_a_fft" in _model_cache:
        return _model_cache["vits_stage_a_fft"]
    model, ap, tokenizer, config = _build_vits_model(os.path.join(VITS_STAGE_A_FFT_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(VITS_STAGE_A_FFT_DIR)
    if pth is None:
        return None, None, None
    model = model.to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["vits_stage_a_fft"] = (model, tokenizer, ap)
    return model, tokenizer, ap


def _detect_lora_rank(state_dict):
    """Auto-detect LoRA rank from checkpoint weight shapes."""
    for key, val in state_dict.items():
        if "lora_A.weight" in key:
            return val.shape[0]
    return LORA_RANK


def get_vits_lora():
    if "vits_lora" in _model_cache:
        return _model_cache["vits_lora"]
    model, ap, tokenizer, config = _build_vits_model(os.path.join(VITS_LORA_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(VITS_LORA_DIR)
    if pth is None:
        return None, None, None
    ckpt = torch.load(pth, map_location="cpu")
    rank = _detect_lora_rank(ckpt["model"])
    print(f"[VITS Stage B LoRA] Detected rank: {rank}")
    model = model.to(DEVICE)
    inject_lora_into_vits(model, r=rank)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["vits_lora"] = (model, tokenizer, ap)
    return model, tokenizer, ap


def get_vits_fft():
    if "vits_fft" in _model_cache:
        return _model_cache["vits_fft"]
    model, ap, tokenizer, config = _build_vits_model(os.path.join(VITS_FFT_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(VITS_FFT_DIR)
    if pth is None:
        return None, None, None
    model = model.to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["vits_fft"] = (model, tokenizer, ap)
    return model, tokenizer, ap


# --- Glow-TTS loaders ---

def get_glow_stage_a_lora():
    if "glow_stage_a_lora" in _model_cache:
        return _model_cache["glow_stage_a_lora"]
    model, ap, tokenizer, config = _build_glow_tts_model(os.path.join(GLOW_STAGE_A_LORA_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(GLOW_STAGE_A_LORA_DIR)
    if pth is None:
        return None, None, None
    ckpt = torch.load(pth, map_location="cpu")
    rank = _detect_lora_rank(ckpt["model"])
    print(f"[Glow-TTS Stage A LoRA] Detected rank: {rank}")
    model = model.to(DEVICE)
    inject_lora_into_glow_tts(model, r=rank)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["glow_stage_a_lora"] = (model, tokenizer, ap)
    return model, tokenizer, ap


def get_glow_stage_a_fft():
    if "glow_stage_a_fft" in _model_cache:
        return _model_cache["glow_stage_a_fft"]
    model, ap, tokenizer, config = _build_glow_tts_model(os.path.join(GLOW_STAGE_A_FFT_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(GLOW_STAGE_A_FFT_DIR)
    if pth is None:
        return None, None, None
    model = model.to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["glow_stage_a_fft"] = (model, tokenizer, ap)
    return model, tokenizer, ap


def get_glow_stage_b_lora():
    if "glow_stage_b_lora" in _model_cache:
        return _model_cache["glow_stage_b_lora"]
    model, ap, tokenizer, config = _build_glow_tts_model(os.path.join(GLOW_STAGE_B_LORA_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(GLOW_STAGE_B_LORA_DIR)
    if pth is None:
        return None, None, None
    ckpt = torch.load(pth, map_location="cpu")
    rank = _detect_lora_rank(ckpt["model"])
    print(f"[Glow-TTS Stage B LoRA] Detected rank: {rank}")
    model = model.to(DEVICE)
    inject_lora_into_glow_tts(model, r=rank)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["glow_stage_b_lora"] = (model, tokenizer, ap)
    return model, tokenizer, ap


def get_glow_stage_b_fft():
    if "glow_stage_b_fft" in _model_cache:
        return _model_cache["glow_stage_b_fft"]
    model, ap, tokenizer, config = _build_glow_tts_model(os.path.join(GLOW_STAGE_B_FFT_DIR, "config.json"))
    if model is None:
        return None, None, None
    pth = _find_pth(GLOW_STAGE_B_FFT_DIR)
    if pth is None:
        return None, None, None
    model = model.to(DEVICE)
    ckpt = torch.load(pth, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    _model_cache["glow_stage_b_fft"] = (model, tokenizer, ap)
    return model, tokenizer, ap


# ---------------------------------------------------------------------------
# Synthesis functions
# ---------------------------------------------------------------------------

def synthesize_vits(text, model, tokenizer):
    """Run VITS inference — returns numpy waveform directly."""
    with torch.no_grad():
        seq = tokenizer.text_to_ids(text)
        seq_t = torch.LongTensor([seq]).to(DEVICE)
        out = model.inference(seq_t) if hasattr(model, "inference") else model(seq_t)
        if isinstance(out, dict):
            for key in ("wav", "model_outputs"):
                if key in out and out[key] is not None:
                    wav = out[key]
                    break
            else:
                wav = next(v for v in out.values() if torch.is_tensor(v))
        else:
            wav = out
        wav = wav.detach().cpu().numpy().squeeze()
    return wav


def _get_vocoder():
    """Load HiFi-GAN vocoder (cached)."""
    if "vocoder" in _model_cache:
        return _model_cache["vocoder"]

    config_path = os.path.join(VOCODER_DIR, "config.json")
    model_path = os.path.join(VOCODER_DIR, "model_file.pth")
    if not os.path.exists(model_path):
        print("[Vocoder] No HiFi-GAN found, will use Griffin-Lim fallback")
        return None

    # Parse config (has // comments)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = re.sub(r'//.*', '', raw)
    cfg = json.loads(cleaned)

    gen_params = cfg.get("generator_model_params", {})
    vocoder = HifiganGenerator(
        in_channels=cfg.get("audio", {}).get("num_mels", 80),
        out_channels=1,
        **gen_params,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        vocoder.load_state_dict(ckpt["model"])
    else:
        vocoder.load_state_dict(ckpt)
    vocoder = vocoder.to(DEVICE)
    vocoder.eval()
    vocoder.remove_weight_norm()
    print("[Vocoder] HiFi-GAN loaded successfully")
    _model_cache["vocoder"] = vocoder
    return vocoder


def synthesize_glow_tts(text, model, tokenizer, ap):
    """Run Glow-TTS inference — returns numpy waveform via HiFi-GAN or Griffin-Lim."""
    with torch.no_grad():
        seq = tokenizer.text_to_ids(text)
        seq_t = torch.LongTensor([seq]).to(DEVICE)
        seq_len = torch.LongTensor([len(seq)]).to(DEVICE)

        outputs = model.inference(seq_t, aux_input={"x_lengths": seq_len})

        # Extract mel spectrogram
        if isinstance(outputs, dict):
            for key in ("model_outputs", "mel_outputs", "z"):
                if key in outputs and outputs[key] is not None:
                    mel = outputs[key]
                    break
            else:
                mel = next(v for v in outputs.values() if torch.is_tensor(v))
        elif isinstance(outputs, (tuple, list)):
            mel = outputs[0]
        else:
            mel = outputs

        # Try HiFi-GAN vocoder first
        vocoder = _get_vocoder()
        if vocoder is not None:
            if isinstance(mel, np.ndarray):
                mel = torch.FloatTensor(mel)
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)
            # Ensure shape is (batch, n_mels, time)
            if mel.shape[1] != 80 and mel.shape[2] == 80:
                mel = mel.transpose(1, 2)
            mel = mel.to(DEVICE)
            wav = vocoder(mel)
            wav = wav.squeeze().cpu().numpy()
            return wav

        # Fallback to Griffin-Lim
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        if mel.ndim == 3:
            mel = mel.squeeze(0)
        if mel.shape[0] != 80 and mel.shape[1] == 80:
            mel = mel.T

    mel = np.clip(mel, -100.0, 2.0)
    wav = ap.inv_melspectrogram(mel)

    if not np.isfinite(wav).all() or np.abs(wav).max() < 1e-10:
        mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=1024, n_mels=80, fmin=50.0, fmax=7600.0
        )
        mel_linear = np.power(10.0, mel)
        S = np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_linear))
        wav = librosa.griffinlim(S, n_iter=60, hop_length=256, win_length=1024)
    return wav


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

EXAMPLE_SENTENCES = [
    "Maayong buntag",
    "Kumusta ka?",
    "Salamat kaayo",
    "Maayong gabii kanimo",
    "Unsa imong pangalan?",
]

# Public deployment ships only the VITS Stage A LoRA checkpoint.
# (loader_fn, model_path, model_type)
MODEL_LOADERS = {
    "VITS Stage A + LoRA": (get_vits_stage_a_lora, "models/vits/stage_a_lora/", "vits"),
}


def tts_handler(text, model_choice):
    """Main TTS handler called by Gradio."""
    if not text or not text.strip():
        raise gr.Error("Please enter some Cebuano text.")

    text = text.strip()

    if model_choice not in MODEL_LOADERS:
        raise gr.Error(f"Unknown model: {model_choice}")

    loader_fn, model_path, model_type = MODEL_LOADERS[model_choice]

    try:
        result = loader_fn()
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Failed to load {model_choice}: {e}")

    if result[0] is None:
        raise gr.Error(f"Model not found. Please add model files to {model_path}")
    model, tokenizer, ap = result

    try:
        if model_type == "glow":
            wav = synthesize_glow_tts(text, model, tokenizer, ap)
        else:
            wav = synthesize_vits(text, model, tokenizer)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Synthesis failed: {e}")

    # Apply volume boost if configured
    gain = VOLUME_BOOST.get(model_choice, 1.0)
    if gain != 1.0:
        wav = wav * gain
        wav = np.clip(wav, -1.0, 1.0)

    return (SAMPLE_RATE, wav)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_app():
    with gr.Blocks(
        title="SugboSound — Cebuano TTS",
    ) as demo:
        gr.Markdown(
            """
            # SugboSound
            ### Cebuano (Bisaya) Text-to-Speech
            *Data-Adaptive Transfer Learning using LoRA for Low-Resource Cebuano TTS*

            Type or paste Cebuano text below and select a model to generate speech.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Cebuano Text",
                    placeholder="e.g. Maayong buntag, kumusta ka?",
                    lines=3,
                )
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_LOADERS.keys()),
                    value="VITS Stage A + LoRA",
                    label="Model",
                )
                synthesize_btn = gr.Button("Synthesize", variant="primary")

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Generated Speech", type="numpy")

        gr.Examples(
            examples=[[s, "VITS Stage A + LoRA"] for s in EXAMPLE_SENTENCES],
            inputs=[text_input, model_dropdown],
            outputs=audio_output,
            fn=tts_handler,
            cache_examples=False,
        )

        gr.Markdown(
            """
            ---
            **Deployed model:**
            | Model | Architecture | Stage | Method |
            |-------|-------------|-------|--------|
            | VITS Stage A + LoRA | VITS | Phoneme Adaptation | LoRA (rank 8, ~2.87M trainable params) |

            *Thesis project — University of Science and Technology of Southern Philippines (USTP)*
            """
        )

        synthesize_btn.click(
            fn=tts_handler,
            inputs=[text_input, model_dropdown],
            outputs=audio_output,
        )

    return demo


if __name__ == "__main__":
    ensure_stage_a_lora_files()
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
