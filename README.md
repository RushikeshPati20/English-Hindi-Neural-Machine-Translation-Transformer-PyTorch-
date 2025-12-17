# 🇬🇧➡️🇮🇳 English → Hindi Neural Machine Translation (Transformer, PyTorch)

A **single-file, production-ready Encoder–Decoder Transformer** for **English → Hindi** translation, implemented in **pure PyTorch** and trained on the **CFILT IITB English–Hindi dataset**.

This project focuses on **robustness, reproducibility, and practical training concerns**, including Windows-safe multiprocessing, lazy SentencePiece loading, AMP training, and optional `torch.compile()` acceleration.

---

## ✨ Features

- 🔁 **Encoder–Decoder Transformer** (from scratch, no `nn.Transformer`)
- 📚 **CFILT IITB English–Hindi** dataset support (`datasets` library)
- 🧩 **Robust dataset field extraction** (handles nested & irregular schemas)
- 🔤 **SentencePiece BPE tokenization**
  - Joint English–Hindi vocabulary
  - Lazy, worker-safe loading
- ⚡ **Mixed Precision Training (AMP)** with `GradScaler`
- 🧵 **Windows-safe DataLoader**
  - Automatic fallback if multi-worker loading fails
- 🚀 **Optional `torch.compile()`** (PyTorch 2.x, safe fallback)
- 📊 **BLEU evaluation** (via `sacrebleu`)
- ⏱️ **ETA & performance tracking**
- 🔍 **Greedy + Beam Search decoding**
- 📦 **Single-file implementation** for easy inspection & modification

---

## 🧠 Model Architecture

- Token + positional embeddings  
- 6-layer Transformer Encoder  
- 6-layer Transformer Decoder  
- Multi-head self-attention & cross-attention  
- LayerNorm + residual connections  
- Vocabulary size: **32,000 (SentencePiece BPE)**  

---

## 📦 Requirements

```bash
pip install torch datasets sentencepiece sacrebleu tqdm
