import os, re, json, base64
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# Optional deps (only needed in certain methods)
# - fitz (PyMuPDF) for PDF parsing
# - PIL for image IO
# - faiss for indexing
# - openai for OpenAI embeddings/vision/answers
# - transformers/torch for BLIP + CLIP
# - cohere for Cohere embeddings

# -----------------------
# Data schemas
# -----------------------
@dataclass
class TextChunk:
    chunk_id: str   # e.g., "p3_c2"
    page: int
    text: str

@dataclass
class ImageRecord:
    image_id: str   # e.g., "p3_img1"
    page: int
    image_path: str

# -----------------------
# Chunking + PDF extraction
# -----------------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Character-based chunker with overlap. Replace with token-based chunking if you want."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def load_pdf_extract_text_and_images(
    pdf_path: str,
    artifacts_dir: str = "artifacts",
    chunk_size: int = 900,
    overlap: int = 150,
) -> Tuple[List[TextChunk], List[ImageRecord]]:
    import fitz  # PyMuPDF

    os.makedirs(artifacts_dir, exist_ok=True)
    images_dir = os.path.join(artifacts_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    text_chunks: List[TextChunk] = []
    image_records: List[ImageRecord] = []

    for p in range(len(doc)):
        page = doc[p]
        page_text = page.get_text("text") or ""

        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            chunk_id = f"p{p+1}_c{i+1}"
            text_chunks.append(TextChunk(chunk_id=chunk_id, page=p+1, text=ch))

        img_list = page.get_images(full=True)
        for j, img in enumerate(img_list):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            img_ext = base.get("ext", "png")
            image_id = f"p{p+1}_img{j+1}"
            out_path = os.path.join(images_dir, f"{image_id}.{img_ext}")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            image_records.append(ImageRecord(image_id=image_id, page=p+1, image_path=out_path))

    return text_chunks, image_records

def save_metadata(text_chunks: List[TextChunk], image_records: List[ImageRecord], artifacts_dir: str = "artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "text_chunks.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in text_chunks], f, ensure_ascii=False, indent=2)
    with open(os.path.join(artifacts_dir, "image_records.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in image_records], f, ensure_ascii=False, indent=2)

# -----------------------
# FAISS helpers
# -----------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)

def build_faiss_ip_index(embeddings: np.ndarray):
    import faiss
    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def faiss_search(index, query_emb: np.ndarray, top_k: int = 5):
    D, I = index.search(query_emb.astype(np.float32), top_k)
    return D[0], I[0]

# -----------------------
# OpenAI embeddings + vision + answering
# -----------------------
def openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def openai_embed_texts(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 128) -> np.ndarray:
    client = openai_client()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_vecs.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 1), dtype=np.float32)

def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def gpt4o_describe_image(image_path: str, model: str = "gpt-4o-mini") -> str:
    client = openai_client()
    b64 = encode_image_b64(image_path)
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Write a concise but descriptive caption of this image for semantic search/retrieval."},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
    )
    return resp.output_text.strip()

def answer_with_citations(query: str, retrieved_chunks: List[Tuple[TextChunk, float]], model: str = "gpt-4o-mini") -> str:
    client = openai_client()
    context = "\n\n".join([f"[{ch.chunk_id} | p{ch.page}] {ch.text}" for ch, _ in retrieved_chunks])
    prompt = f"""You are answering a question using ONLY the provided context excerpts.
- Cite sources inline using the bracketed chunk id like [p3_c2].
- If you don't know, say so.

Question: {query}

Context:
{context}
"""
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text.strip()

# -----------------------
# BLIP captions (offline)
# -----------------------
def blip_caption_images(image_records: List[ImageRecord], batch_size: int = 8, max_new_tokens: int = 40) -> Dict[str, str]:
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import BlipProcessor, BlipForConditionalGeneration

    blip_model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(blip_model_name)
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    captions: Dict[str, str] = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(image_records), batch_size), desc="Captioning images (BLIP)"):
            batch = image_records[i:i+batch_size]
            imgs = [Image.open(r.image_path).convert("RGB") for r in batch]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            texts = processor.batch_decode(out, skip_special_tokens=True)
            for rec, cap in zip(batch, texts):
                captions[rec.image_id] = cap.strip()
    return captions

# -----------------------
# CLIP embeddings (shared space)
# -----------------------
def clip_load(model_name: str = "openai/clip-vit-base-patch32"):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device

def clip_embed_images(image_records: List[ImageRecord], model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 16) -> np.ndarray:
    import torch
    from PIL import Image
    from tqdm import tqdm
    model, processor, device = clip_load(model_name)
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_records), batch_size), desc="Embedding images (CLIP)"):
            batch = image_records[i:i+batch_size]
            imgs = [Image.open(r.image_path).convert("RGB") for r in batch]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(embs) if embs else np.zeros((0, model.config.projection_dim), dtype=np.float32)

def clip_embed_texts(texts: List[str], model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 64) -> np.ndarray:
    import torch
    from tqdm import tqdm
    model, processor, device = clip_load(model_name)
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts (CLIP)"):
            batch = texts[i:i+batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(embs) if embs else np.zeros((0, model.config.projection_dim), dtype=np.float32)

# -----------------------
# Cohere embeddings (text + multimodal)
# -----------------------
def cohere_client():
    import cohere
    return cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))

def image_to_base64_data_url(image_path: str) -> str:
    _, ext = os.path.splitext(image_path)
    file_type = (ext[1:] or "jpeg").lower()
    with open(image_path, "rb") as f:
        enc = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{file_type};base64,{enc}"

def cohere_embed_texts(texts: List[str], model: str, input_type: str, batch_size: int = 64) -> np.ndarray:
    co = cohere_client()
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = [{"content": [{"type": "text", "text": t}]} for t in batch]

        resp = co.embed(
            model=model,
            inputs=inputs,
            input_type=input_type,
            embedding_types=["float"],
        )
        vecs.append(np.array(resp.embeddings.float_, dtype=np.float32))

    return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)


def cohere_embed_images(image_paths: List[str], model: str, input_type: str, batch_size: int = 16) -> np.ndarray:
    co = cohere_client()
    vecs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        inputs = []
        for p in batch_paths:
            base64_url = image_to_base64_data_url(p)
            inputs.append({
                "content": [{"type": "image_url", "image_url": {"url": base64_url}}]
            })

        resp = co.embed(
            model=model,
            inputs=inputs,
            input_type=input_type,
            embedding_types=["float"],
        )
        vecs.append(np.array(resp.embeddings.float_, dtype=np.float32))

    return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)

# -----------------------
# Display helper (not required)
# -----------------------
def show_images(image_hits, max_show: int = 4):
    from PIL import Image
    from IPython.display import display
    for rec, score in image_hits[:max_show]:
        print(f"{rec.image_id} (page {rec.page}) score={score:.3f} -> {rec.image_path}")
        display(Image.open(rec.image_path))
