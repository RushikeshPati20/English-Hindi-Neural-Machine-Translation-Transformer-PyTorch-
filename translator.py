"""
en_hi_translator_full_patched.py

Patched single-file Encoder-Decoder Transformer (PyTorch) for English -> Hindi.
Features:
- Robust dataset field handling for cfilt/iitb-english-hindi
- Lazy SentencePiece loading (worker-safe)
- DataLoader worker-fallback for Windows (uses num_workers but falls back to 0 if workers crash)
- Mixed precision (AMP) training with GradScaler
- Optional torch.compile() if available
- ETA/timing prints and faster default validation
- All runtime inside `main()` guarded by if __name__ == "__main__"

Install required packages:
    pip install torch datasets sentencepiece sacrebleu tqdm
Run:
    python en_hi_translator_full_patched.py
"""

import os
import random
import time
import traceback
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import sentencepiece as spm
import sacrebleu
from pprint import pprint
from tqdm import tqdm

# ------------------------
# Config / hyperparameters
# ------------------------
SEED = 1337
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
D_MODEL = 384
N_HEAD = 6
N_LAYER_ENC = 6
N_LAYER_DEC = 6
DROPOUT = 0.2
FFWD_MULT = 4

SP_VOCAB_SIZE = 32000
# SentencePiece special ids (ensure unique)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_STEPS = 50000            # total training steps (change for quick demo)
EVAL_INTERVAL = 500
WARMUP_STEPS = 400
GRAD_CLIP = 1.0

# DataLoader performance prefs
DATA_LOADER_NUM_WORKERS = 4
DATA_LOADER_PIN_MEMORY = True

# Quick-run option (set to integer to use a subset of train for fast debug)
SUBSET_FOR_QUICK_RUN = None  # e.g. 20000

# SentencePiece model path
SP_MODEL_PREFIX = "spm_en_hi"
SP_MODEL_FILE = f"{SP_MODEL_PREFIX}.model"

# reproducibility
torch.manual_seed(SEED)
random.seed(SEED)

# ------------------------
# Lazy SentencePiece (worker-safe)
# ------------------------
_sp = None


def get_sp():
    global _sp
    if _sp is None:
        if not os.path.exists(SP_MODEL_FILE):
            raise RuntimeError(f"SentencePiece model not found at {SP_MODEL_FILE}. Run training block first.")
        _sp = spm.SentencePieceProcessor()
        _sp.load(SP_MODEL_FILE)
    return _sp


def encode_text(text: str) -> List[int]:
    sp = get_sp()
    return sp.encode(text, out_type=int)


def decode_ids(ids: List[int]) -> str:
    sp = get_sp()
    return sp.decode(ids)


# ------------------------
# Dataset extraction helper
# ------------------------
def extract_pair(ex) -> Tuple[str, str]:
    """
    Try extracting (en, hi) from dataset example `ex`.
    Handles nested 'translation' dict and common top-level keys; falls back to script heuristics.
    """
    # nested translation dict
    if isinstance(ex, dict) and 'translation' in ex and isinstance(ex['translation'], dict):
        tr = ex['translation']
        en = tr.get('en') or tr.get('english') or tr.get('source') or tr.get('src')
        hi = tr.get('hi') or tr.get('hindi') or tr.get('target') or tr.get('tgt')
        return en, hi

    en_candidates = ['en', 'english', 'source', 'src', 'sent1', 'text']
    hi_candidates = ['hi', 'hindi', 'target', 'tgt', 'sent2']

    en = None
    hi = None

    if isinstance(ex, dict):
        for k in en_candidates:
            if k in ex:
                en = ex.get(k)
                break
        for k in hi_candidates:
            if k in ex:
                hi = ex.get(k)
                break

    # fallback heuristics: Latin vs Devanagari ranges
    if (not en or not hi) and isinstance(ex, dict):
        for k, v in ex.items():
            if isinstance(v, str):
                if en is None and any(('a' <= ch.lower() <= 'z') for ch in v):
                    en = v
                if hi is None and any('\u0900' <= ch <= '\u097F' for ch in v):
                    hi = v

    return en, hi


# ------------------------
# PyTorch Dataset and collate (worker-safe)
# ------------------------
class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN):
        self.ds = hf_dataset
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            ex = self.ds[idx]
            en, hi = extract_pair(ex)
            if en is None or hi is None:
                return {'src_ids': [], 'tgt_ids': []}
            # lazy tokenize using get_sp() inside worker
            src_ids = encode_text(en)[:self.max_src]
            tgt_ids = encode_text(hi)[:(self.max_tgt - 2)]
            return {'src_ids': src_ids, 'tgt_ids': tgt_ids}
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Error in Dataset.__getitem__ idx={idx}: {e}\n{tb}")


def collate_fn(batch):
    # filter empty
    batch = [b for b in batch if len(b['src_ids']) > 0 and len(b['tgt_ids']) > 0]
    if len(batch) == 0:
        return {
            'src_input': torch.zeros((1, 1), dtype=torch.long),
            'src_mask': torch.zeros((1, 1), dtype=torch.long),
            'tgt_input': torch.zeros((1, 1), dtype=torch.long),
            'tgt_out': torch.full((1, 1), -100, dtype=torch.long)
        }

    srcs = [b['src_ids'] for b in batch]
    tgts = [b['tgt_ids'] for b in batch]

    S_max = min(MAX_SRC_LEN, max(len(s) for s in srcs))
    T_max = min(MAX_TGT_LEN, max(len(t) + 2 for t in tgts))

    def pad(seqs, L, pad_value=PAD_ID):
        out = torch.full((len(seqs), L), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            Ls = min(len(s), L)
            out[i, :Ls] = torch.tensor(s[:Ls], dtype=torch.long)
        return out

    src_padded = pad(srcs, S_max)
    tgt_input = []
    tgt_out = []
    for t in tgts:
        inp = [BOS_ID] + t
        out = t + [EOS_ID]
        inp = inp[:T_max]
        out = out[:T_max]
        tgt_input.append(inp)
        tgt_out.append(out)

    tgt_input_padded = pad(tgt_input, T_max)
    tgt_out_padded = pad(tgt_out, T_max, pad_value=-100)

    src_mask = (src_padded != PAD_ID).long()
    tgt_mask = (tgt_input_padded != PAD_ID).long()

    return {
        'src_input': src_padded,
        'src_mask': src_mask,
        'tgt_input': tgt_input_padded,
        'tgt_out': tgt_out_padded
    }


# ------------------------
# Transformer building blocks (in style of your original file)
# ------------------------
def subsequent_mask(sz: int):
    return torch.tril(torch.ones((sz, sz), dtype=torch.bool))


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.scale = head_size ** -0.5

    def forward(self, q, k, v, mask=None):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        scores = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            if mask.dim() == 2:
                mask_qkv = mask.unsqueeze(1)
                scores = scores.masked_fill(~mask_qkv, float('-inf'))
            elif mask.dim() == 3:
                scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        assert n_embd % num_heads == 0
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, q, k, v, mask=None):
        out = torch.cat([h(q, k, v, mask=mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, FFWD_MULT * n_embd),
            nn.ReLU(),
            nn.Linear(FFWD_MULT * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)

    def forward(self, x, src_mask):
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x), mask=src_mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.self_attn = MultiHeadAttention(n_embd, n_head)
        self.cross_attn = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        B, T, _ = x.size()
        tgt_pad = tgt_mask.to(torch.bool)
        final_mask = torch.zeros((B, T, T), dtype=torch.bool, device=x.device)
        arangeT = torch.arange(T, device=x.device)
        for b in range(B):
            allowed = tgt_pad[b]
            for i in range(T):
                final_mask[b, i, :] = allowed & (arangeT <= i)
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=final_mask)
        cross_mask = src_mask.to(torch.bool)[:, None, :].expand(-1, T, -1)
        x = x + self.cross_attn(self.ln2(x), enc_out, enc_out, mask=cross_mask)
        x = x + self.ff(self.ln3(x))
        return x


# ------------------------
# Seq2Seq model
# ------------------------
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer_enc, n_layer_dec, max_src_len, max_tgt_len):
        super().__init__()
        self.tok_enc = nn.Embedding(vocab_size, n_embd, padding_idx=PAD_ID)
        self.tok_dec = nn.Embedding(vocab_size, n_embd, padding_idx=PAD_ID)
        self.pos_enc = nn.Embedding(max_src_len, n_embd)
        self.pos_dec = nn.Embedding(max_tgt_len, n_embd)
        self.enc_layers = nn.ModuleList([EncoderLayer(n_embd, n_head) for _ in range(n_layer_enc)])
        self.dec_layers = nn.ModuleList([DecoderLayer(n_embd, n_head) for _ in range(n_layer_dec)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_input, src_mask):
        B, S = src_input.size()
        pos = torch.arange(S, device=src_input.device)[None, :].expand(B, -1)
        x = self.tok_enc(src_input) + self.pos_enc(pos)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_input, enc_out, tgt_mask, src_mask):
        B, T = tgt_input.size()
        pos = torch.arange(T, device=tgt_input.device)[None, :].expand(B, -1)
        x = self.tok_dec(tgt_input) + self.pos_dec(pos)
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, src_input, src_mask, tgt_input, tgt_out=None):
        enc_out = self.encode(src_input, src_mask)
        logits = self.decode(tgt_input, enc_out, (tgt_input != PAD_ID).long(), src_mask)
        loss = None
        if tgt_out is not None:
            B, T, V = logits.size()
            logits_flat = logits.view(B * T, V)
            targets_flat = tgt_out.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        return logits, loss

    def generate_greedy(self, src_input, src_mask, max_len=80):
        self.eval()
        with torch.no_grad():
            enc_out = self.encode(src_input, src_mask)
            B = src_input.size(0)
            ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src_input.device)
            finished = torch.zeros(B, dtype=torch.bool, device=src_input.device)
            for _ in range(max_len):
                logits = self.decode(ys, enc_out, (ys != PAD_ID).long(), src_mask)
                next_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tokens], dim=1)
                finished = finished | (next_tokens.squeeze(1) == EOS_ID)
                if finished.all():
                    break
            ys = ys[:, 1:]
            outputs = []
            for i in range(B):
                seq = ys[i].tolist()
                if EOS_ID in seq:
                    seq = seq[:seq.index(EOS_ID)]
                outputs.append(seq)
            self.train()
            return outputs


# ------------------------
# Evaluation helpers
# ------------------------
def detok_from_seq(seq: List[int]) -> str:
    return decode_ids(seq)


def evaluate_bleu_sample(model, loader, device, num_batches=5, max_decode_len=48):
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            src = batch['src_input'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_out = batch['tgt_out'].to(device)
            outs = model.generate_greedy(src, src_mask, max_len=max_decode_len)
            for b in range(len(outs)):
                hyp = detok_from_seq(outs[b])
                ref_ids = [x for x in tgt_out[b].cpu().tolist() if x != -100 and x != PAD_ID]
                ref = detok_from_seq(ref_ids)
                hyps.append(hyp)
                refs.append(ref)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]) if len(hyps) > 0 else type('obj', (), {'score': 0.0})()
    model.train()
    return bleu.score if hasattr(bleu, 'score') else 0.0


# ------------------------
# Beam search (single-sentence) - kept simple
# ------------------------
def beam_search_single(model, src_input, src_mask, beam_size=5, max_len=80, length_penalty=1.0):
    assert src_input.size(0) == 1
    model.eval()
    with torch.no_grad():
        enc_out = model.encode(src_input, src_mask)
        beams = [([BOS_ID], 0.0)]
        finished = []
        for _ in range(max_len):
            all_candidates = []
            for seq, score in beams:
                if seq[-1] == EOS_ID:
                    all_candidates.append((seq, score))
                    continue
                tgt = torch.tensor([seq], dtype=torch.long, device=src_input.device)
                logits = model.decode(tgt, enc_out, (tgt != PAD_ID).long(), src_mask)
                logp = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                topk_logp, topk_idx = torch.topk(logp, beam_size)
                for k in range(beam_size):
                    cand_seq = seq + [int(topk_idx[k].item())]
                    cand_score = score + float(topk_logp[k].item())
                    all_candidates.append((cand_seq, cand_score))
            beams = sorted(all_candidates, key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)[:beam_size]
            beams_alive = []
            for seq, score in beams:
                if seq[-1] == EOS_ID:
                    finished.append((seq, score))
                else:
                    beams_alive.append((seq, score))
            beams = beams_alive
            if not beams:
                break
        candidates = finished if finished else beams
        best = sorted(candidates, key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)[0][0]
        if best[0] == BOS_ID:
            best = best[1:]
        if EOS_ID in best:
            best = best[:best.index(EOS_ID)]
        model.train()
        return best


# ------------------------
# Main: all runtime inside guard (Windows-safe)
# ------------------------
def main():
    # 1) Load dataset (Hugging Face)
    print("Loading dataset 'cfilt/iitb-english-hindi' via Hugging Face datasets...")
    ds = load_dataset("cfilt/iitb-english-hindi")
    print("Available splits:", ds.keys())

    raw_train = ds['train']
    print("Train split column names:", raw_train.column_names)
    print("Train features:", raw_train.features)
    print("Example item (first):")
    pprint(raw_train[0])

    if SUBSET_FOR_QUICK_RUN is not None:
        n_total = min(len(raw_train), SUBSET_FOR_QUICK_RUN)
        raw_train = raw_train.select(range(n_total))
        print(f"Reduced train split to first {n_total} samples for quick run.")

    raw = raw_train.shuffle(seed=SEED)
    n_total = len(raw)
    n_train = int(0.9 * n_total)
    train_raw = raw.select(range(0, n_train))
    val_raw = raw.select(range(n_train, n_total))
    print(f"Dataset ready. total={n_total}, train={len(train_raw)}, val={len(val_raw)}")

    # 2) SentencePiece training (if no model exists)
    if not os.path.exists(SP_MODEL_FILE):
        print("Preparing combined corpus for SentencePiece training...")
        tmp_all = "spm_all_text.txt"
        count_pairs = 0
        with open(tmp_all, "w", encoding="utf-8") as f:
            for ex in tqdm(train_raw, desc="Collecting sentences"):
                en, hi = extract_pair(ex)
                if not en or not hi:
                    continue
                en = en.strip()
                hi = hi.strip()
                if en:
                    f.write(en + "\n")
                if hi:
                    f.write(hi + "\n")
                count_pairs += 1
        if count_pairs == 0:
            raise RuntimeError("No parallel pairs found in dataset. Inspect dataset structure and adjust extract_pair().")
        print(f"Wrote {count_pairs} pairs (2 lines per pair) to {tmp_all}. Training SentencePiece...")
        spm.SentencePieceTrainer.Train(
            f"--input={tmp_all} --model_prefix={SP_MODEL_PREFIX} --vocab_size={SP_VOCAB_SIZE} "
            f"--character_coverage=1.0 --model_type=bpe "
            f"--pad_id={PAD_ID} --unk_id={UNK_ID} --bos_id={BOS_ID} --eos_id={EOS_ID}"
        )
        os.remove(tmp_all)
    else:
        print("Found existing SentencePiece model:", SP_MODEL_FILE)

    # load sp lazily in workers via get_sp()
    _ = get_sp()
    vocab_size = get_sp().get_piece_size()
    print("Loaded SentencePiece. vocab_size =", vocab_size)

    # 3) Create datasets and DataLoaders, with worker fallback
    train_ds = TranslationDataset(train_raw)
    val_ds = TranslationDataset(val_raw)

    num_workers = DATA_LOADER_NUM_WORKERS
    pin_memory = DATA_LOADER_PIN_MEMORY

    try:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        # sanity check: iterate one batch to surface errors early
        _ = next(iter(train_loader))
    except Exception as e:
        print("Warning: DataLoader with num_workers=", num_workers, "failed with error:\n", e)
        print("Falling back to num_workers=0 (single-process loading).")
        num_workers = 0
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)

    # 4) Instantiate model, optimizer, scheduler
    model = Seq2Seq(vocab_size=vocab_size, n_embd=D_MODEL, n_head=N_HEAD,
                    n_layer_enc=N_LAYER_ENC, n_layer_dec=N_LAYER_DEC,
                    max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN)

    # optional torch.compile for PyTorch 2.x
    # optional torch.compile for PyTorch 2.x — safe fallback if Triton missing
    try:
        # Try to compile if available. This may raise if Triton or other deps are missing.
        model = torch.compile(model, backend="eager")
        print("torch.compile applied (eager backend, no Triton)")

    except Exception as e:
        # If compile fails (e.g. TritonMissing), continue without compiling.
        print("torch.compile not applied — continuing without compile. Reason:")
        print(repr(e))
        # Optionally show user-friendly hint
        print("If you want to enable torch.compile, install a compatible Triton build;")
        print("otherwise training will run fine without compile (AMP + other speedups remain).")


    model = model.to(DEVICE)
    print("Model parameters (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        return max(0.0, 1.0 - (step - WARMUP_STEPS) / float(MAX_STEPS - WARMUP_STEPS))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5) Training loop with AMP and timing
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    print("Starting training loop (AMP enabled)...")
    global_step = 0
    best_val_bleu = 0.0
    measured_steps = 0
    measured_time = 0.0
    t0_global = time.time()

    for epoch in range(1000):
        for batch in train_loader:
            model.train()
            # move to device
            src = batch['src_input'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            tgt_in = batch['tgt_input'].to(DEVICE)
            tgt_out = batch['tgt_out'].to(DEVICE)

            t0 = time.time()
            optimizer.zero_grad()

            with autocast():
                logits, loss = model(src, src_mask, tgt_in, tgt_out)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step_time = time.time() - t0
            measured_steps += 1
            measured_time += step_time

            global_step += 1

            if global_step % 50 == 0:
                avg = measured_time / max(1, measured_steps)
                remaining_steps = max(0, MAX_STEPS - global_step)
                eta_seconds = remaining_steps * avg
                print(f"step {global_step} loss {loss.item():.4f}  avg_step_time={avg:.3f}s  ETA={eta_seconds/60:.2f}min")
                measured_steps = 0
                measured_time = 0.0

            if global_step % EVAL_INTERVAL == 0:
                print("Evaluating on validation (quick sample)...")
                val_bleu = evaluate_bleu_sample(model, val_loader, DEVICE, num_batches=5, max_decode_len=48)
                print(f"Validation BLEU (quick, greedy) at step {global_step}: {val_bleu:.2f}")
                if val_bleu > best_val_bleu:
                    best_val_bleu = val_bleu
                    torch.save(model.state_dict(), "best_en_hi_seq2seq.pt")
                    print(f"Saved best model (BLEU {best_val_bleu:.2f})")

            if global_step >= MAX_STEPS:
                break
        if global_step >= MAX_STEPS:
            break

    total_minutes = (time.time() - t0_global) / 60.0
    print(f"Training finished in {total_minutes:.2f} minutes")

    # 6) final evaluation & demo
    if os.path.exists("best_en_hi_seq2seq.pt"):
        model.load_state_dict(torch.load("best_en_hi_seq2seq.pt", map_location=DEVICE))
        print("Loaded best checkpoint.")

    print("Final validation BLEU (sampled, quick):", evaluate_bleu_sample(model, val_loader, DEVICE, num_batches=32, max_decode_len=64))

    def translate(text: str, beam=False, beam_size=5):
        ids = encode_text(text)[:MAX_SRC_LEN]
        src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        src_mask = (src != PAD_ID).long()
        if beam:
            seq = beam_search_single(model, src, src_mask, beam_size=beam_size, max_len=MAX_TGT_LEN)
        else:
            seqs = model.generate_greedy(src, src_mask, max_len=MAX_TGT_LEN)
            seq = seqs[0]
        return decode_ids(seq)

    examples = [
        "Hello, how are you?",
        "India is a beautiful country with diverse languages and cultures.",
        "Can you help me translate this sentence into Hindi?"
    ]

    for s in examples:
        print("EN:", s)
        print("HI (greedy):", translate(s, beam=False))
        print("HI (beam):", translate(s, beam=True, beam_size=5))
        print("-" * 40)

    print("Done.")


if __name__ == "__main__":
    main()
