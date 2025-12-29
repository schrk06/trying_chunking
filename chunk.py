from transformers import AutoTokenizer
import re

# ----------------------------
# Initialisation du tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", use_fast=True)

# ----------------------------
# Utils
# ----------------------------
def normalize_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def split_paragraphs(text: str):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def token_len(text: str):
    """Compte le nombre exact de tokens pour un texte, sans limite."""
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    return encoded["input_ids"].shape[1]

def get_token_overlap(text: str, overlap_tokens: int):
    """Récupère les derniers tokens convertis en texte pour l'overlap."""
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    token_ids = encoded["input_ids"][0]
    overlap_ids = token_ids[-overlap_tokens:]
    return tokenizer.decode(overlap_ids, skip_special_tokens=True)

def split_long_paragraph(text: str, max_tokens: int):
    """Découpe un paragraphe trop long en sous-chunks par phrase."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    current_tokens = 0
    for s in sentences:
        s_tokens = token_len(s)
        if current_tokens + s_tokens <= max_tokens:
            current += (" " + s if current else s)
            current_tokens += s_tokens
        else:
            chunks.append(current.strip())
            current = s
            current_tokens = s_tokens
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ----------------------------
# Chunking sémantique
# ----------------------------
def semantic_chunking(text: str, min_tokens=80, max_tokens=350, overlap_tokens=50):
    text = normalize_text(text)
    paragraphs = split_paragraphs(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_len(para)

        if para_tokens > max_tokens:
            sub_chunks = split_long_paragraph(para, max_tokens)
            chunks.extend(sub_chunks)
            continue

        if current_tokens + para_tokens <= max_tokens:
            current_chunk += ("\n\n" + para if current_chunk else para)
            current_tokens += para_tokens
            continue

        if current_tokens < min_tokens:
            current_chunk += ("\n\n" + para)
            current_tokens += para_tokens
            continue

        # chunk valide → sauvegarde
        chunks.append(current_chunk.strip())

        # overlap
        if overlap_tokens > 0:
            overlap_text = get_token_overlap(current_chunk, overlap_tokens)
            current_chunk = overlap_text + "\n\n" + para
            current_tokens = token_len(current_chunk)
        else:
            current_chunk = para
            current_tokens = para_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# ----------------------------
# Fonction pour ajouter un document
# ----------------------------
def add_document(doc_id: str, text: str, metadata=None):
    metadata = metadata or {}
    chunks = semantic_chunking(text)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "token_count": token_len(chunk),
            "metadata": metadata
        })
    return documents


raw_text = "Ton texte long ici ..."
docs = add_document("doc_01", raw_text, metadata={"source": "user_upload"})

# docs contient tous les chunks en texte
for d in docs:
    print(d["chunk_id"])
    print(d["text"])
    print("tokens:", d["token_count"])
