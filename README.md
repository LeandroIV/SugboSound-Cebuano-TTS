# SugboSound — Cebuano (Bisaya) Text-to-Speech

**Data-Adaptive Transfer Learning using LoRA for Low-Resource Cebuano TTS**

An undergraduate thesis project from the **University of Science and Technology of Southern Philippines (USTP)** that adapts a pre-trained English VITS model to synthesize natural Cebuano speech using **LoRA (Low-Rank Adaptation)**.

## Why this matters

Cebuano (Bisaya) is spoken by over 20 million Filipinos, yet remains drastically under-served by modern speech technology. Training a TTS model from scratch typically requires tens of hours of recorded speech and millions of trainable parameters — resources that are rarely available for low-resource languages.

This project shows that a high-quality Cebuano TTS system can be built by adapting an existing English model with **only ~2.87M trainable parameters (3.5% of the original 83M)** using LoRA, demonstrating a 29× efficiency gain over full fine-tuning.

## Live Demo

- **App:** Run locally with `python app.py` (Gradio UI on port 7860)
- **Model weights:** [`LeandroIV/sugbosound-cebuano-tts-vits`](https://huggingface.co/LeandroIV/sugbosound-cebuano-tts-vits) on Hugging Face Hub (auto-downloaded on first run)

## Methodology

A two-stage transfer-learning pipeline:

| Stage | Goal | Dataset | Method | Result |
|-------|------|---------|--------|--------|
| **A — Phoneme Adaptation** | Teach the model Cebuano phonemes | 2,204 isolated CV/CVC syllables | LoRA (rank 8) on text-encoder attention + FFN | 46% loss reduction (34.2 → 18.55 over 24 epochs) |
| **B — Prosody Fine-tuning** | Teach natural sentence-level rhythm | 657 full sentences (Bloom Cebuano corpus) | Full fine-tune from Stage A checkpoint | Completed; better MCD than full fine-tune, but lower perceptual quality due to limited dataset coverage |

LoRA adapters follow `ΔW = (α/r) · B · A` and are injected into:
- Text-encoder attention: `conv_q`, `conv_k`, `conv_v`
- Text-encoder FFN: `conv_1`, `conv_2`

The base architecture is **VITS** (Variational Inference Text-to-Speech) — a single-stage end-to-end TTS model that combines a flow-based decoder with an adversarial training objective.

### Why this demo deploys Stage A (not Stage B)

Both stages were trained and evaluated. Stage B achieved **better MCD (Mel-Cepstral Distortion) scores than the full fine-tune baseline**, but during listening tests we found that the **Stage A checkpoint is more audible and intelligible** to native Cebuano speakers. The 657-sentence Bloom corpus used for Stage B turned out to be too narrow in domain and prosodic variety to outperform Stage A perceptually, despite the favorable objective metric.

This is consistent with a known limitation in TTS evaluation: objective metrics like MCD do not always track human-perceived quality, particularly in low-resource settings where distribution mismatch between training and inference text dominates the error budget. We document this finding in detail in the thesis manuscript.

This public demo therefore ships the **Stage A + LoRA** checkpoint.

## Setup

```bash
git clone https://github.com/LeandroIV/SugboSound-Cebuano-TTS.git
cd SugboSound-Cebuano-TTS
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first launch will automatically download the VITS Stage A LoRA checkpoint (~499 MB) from Hugging Face Hub into `models/vits/stage_a_lora/`. Subsequent launches use the cached copy.

Then open **http://localhost:7860** in your browser.

## Example phrases

- *Maayong buntag* — Good morning
- *Kumusta ka?* — How are you?
- *Salamat kaayo* — Thank you very much
- *Maayong gabii kanimo* — Good evening to you
- *Unsa imong pangalan?* — What is your name?

## Repository structure

```
sugbosound_app/
├── app.py              # Gradio inference app (auto-downloads model from HF)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── models/             # Auto-populated on first run (gitignored)
```

The full training pipeline, datasets, and experimental scripts live in the parent thesis repository and are not part of this deployment.

## Authors

**CS4A — Computer Science, USTP (2025–2026 academic year):**

- Leandro O. Gica IV
- Zioney Jayce A. Bajalan
- Harvey Francis P. Magarin
- Erwin Dane P. Yarra

**Thesis Adviser:** Dr. Paul Joseph M. Estrera
**Thesis Panel:** PAUL JOSEPH M. ESTRERA, DIT, JUNAR A. LANDICHO, PhD, ENGR. MARICEL A. ESCLAMADO, PhD



## Acknowledgements

This work was conducted as the undergraduate thesis requirement for the Bachelor of Science in Computer Science program at the **University of Science and Technology of Southern Philippines (USTP)**.

We thank:
- The **Bloom Library** initiative for the Cebuano corpus used in Stage B.
- The **Coqui TTS** team for the open-source TTS framework.
- The original **VITS** authors (Kim, Kong, Son, 2021) for the base architecture.
- The **LoRA** authors (Hu et al., 2021) for the adaptation method.

## License

MIT License — see source for full terms. Model weights distributed under the same license at the Hugging Face repository.

## Citation

If you use this work in academic research, please cite:

```bibtex
@thesis{sugbosound2026,
  title  = {Data-Adaptive Transfer Learning using LoRA for Low-Resource Cebuano Text-to-Speech},
  author = {Gica, Leandro IV and others},
  school = {University of Science and Technology of Southern Philippines (USTP)},
  year   = {2026},
  type   = {Undergraduate Thesis}
}
```
