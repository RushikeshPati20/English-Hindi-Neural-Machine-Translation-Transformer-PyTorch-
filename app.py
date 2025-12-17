import streamlit as st
import torch
import sentencepiece as spm

# Import model class from your training file
from translator import (
    Seq2Seq,
    PAD_ID,
    MAX_SRC_LEN,
    MAX_TGT_LEN
)

# ---------------------------
# Config
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_en_hi_seq2seq.pt"
SP_MODEL_PATH = "spm_en_hi.model"

st.set_page_config(
    page_title="English → Hindi Translator",
    page_icon="🌐",
    layout="centered"
)

# ---------------------------
# Load SentencePiece
# ---------------------------
@st.cache_resource
def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL_PATH)
    return sp

sp = load_tokenizer()

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    model = Seq2Seq(
        vocab_size=sp.get_piece_size(),
        n_embd=384,
        n_head=6,
        n_layer_enc=6,
        n_layer_dec=6,
        max_src_len=128,
        max_tgt_len=128,
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)

    # 🔑 FIX: remove "_orig_mod." prefix if present
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("_orig_mod.", "")] = v
        state = new_state

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# ---------------------------
# Translation function
# ---------------------------
@torch.no_grad()
def translate(text: str):
    ids = sp.encode(text, out_type=int)[:MAX_SRC_LEN]
    src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    src_mask = (src != PAD_ID).long()

    out_ids = model.generate_greedy(
        src_input=src,
        src_mask=src_mask,
        max_len=MAX_TGT_LEN
    )[0]

    return sp.decode(out_ids)

# ---------------------------
# UI
# ---------------------------
st.title("🌐 English → Hindi Translator")
st.markdown(
    """
    **Custom Transformer model trained from scratch**  
    Dataset: *IITB English–Hindi Parallel Corpus*  
    """
)

st.divider()

english_text = st.text_area(
    "Enter English text:",
    height=120,
    placeholder="Type an English sentence here..."
)

if st.button("Translate 🇮🇳", use_container_width=True):
    if english_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            hindi_text = translate(english_text)
        st.success("Translation completed!")
        st.text_area(
            "Hindi Translation:",
            hindi_text,
            height=120
        )

st.divider()

st.markdown(
    """
    👨‍💻 **About this project**
    - Encoder–Decoder Transformer implemented from scratch
    - SentencePiece BPE tokenizer
    - Mixed precision (AMP) training
    - BLEU score evaluation
    """
)
