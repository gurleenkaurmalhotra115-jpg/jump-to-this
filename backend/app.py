import os
import sys
import json
import time
import uuid
import threading
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Inject backend directory into python path to guarantee local imports resolve
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Qdrant Cloud Setup
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
qdrant_client = None
if QDRANT_URL:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qd_models
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("[*] Successfully initialized Qdrant Cloud client connection.")
    except Exception as e:
        print(f"[!] Error: Could not connect to Qdrant Cloud: {e}")

app = FastAPI(title="JumpToThis Backend", description="Upgraded Dual-Embedding Moment Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLOUD_DEPLOY = os.environ.get("CLOUD_DEPLOY") == "1"

# Global model references for local execution
clip_model = None
tokenizer = None
text_encoder = None
device = "cpu"

if not CLOUD_DEPLOY:
    import torch
    torch.set_num_threads(4)
    import open_clip
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[*] Loading CLIP Model locally...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.to(device).eval()
    
    print("[*] Loading SentenceTransformer locally...")
    text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_embedding(data):
    while isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        data = data[0]
    return data

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("media", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

loaded_indices = {}
search_cache = {}

def get_video_indices(video: str):
    global loaded_indices
    if video not in loaded_indices:
        print(f"[*] Dynamically loading indexes for video: '{video}'...")
        video_data_dir = os.path.join(DATA_DIR, video)
        
        vec_path = os.path.join(video_data_dir, "visual_vectors.npy")
        ts_path = os.path.join(video_data_dir, "visual_timestamps.json")
        trans_path = os.path.join(video_data_dir, "transcript.json")
        emb_path = os.path.join(video_data_dir, "transcript_embeddings.npy")
        trans_en_path = os.path.join(video_data_dir, "transcript_en.json")
        emb_en_path = os.path.join(video_data_dir, "transcript_en_embeddings.npy")
        
        # We only strictly require the timestamp metadata to play/seek the video
        if not os.path.exists(ts_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Index data files for '{video}' not found. Run indexer for this video first."
            )
            
        try:
            v_vecs = np.load(vec_path) if os.path.exists(vec_path) else None
            with open(ts_path, "r", encoding="utf-8") as f:
                v_timestamps = json.load(f)
                
            trans = []
            if os.path.exists(trans_path):
                with open(trans_path, "r", encoding="utf-8") as f:
                    trans = json.load(f)
            t_embeddings = np.load(emb_path) if os.path.exists(emb_path) else None
            
            trans_en = []
            if os.path.exists(trans_en_path):
                with open(trans_en_path, "r", encoding="utf-8") as f:
                    trans_en = json.load(f)
            t_en_embeddings = np.load(emb_en_path) if os.path.exists(emb_en_path) else None
            
            loaded_indices[video] = {
                "visual_vectors": v_vecs,
                "visual_timestamps": v_timestamps,
                "transcripts": trans,
                "transcript_embeddings": t_embeddings,
                "transcripts_en": trans_en,
                "transcript_en_embeddings": t_en_embeddings
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading index data for '{video}': {e}"
            )
            
    return loaded_indices[video]

def temporal_nms(scores, timestamps_metadata, visual_sims, text_sims_native, text_sims_en, query, transcripts, transcripts_en, window_sec=4.0, top_k=3):
    sorted_indices = np.argsort(scores)[::-1]
    selected = []
    q_words = set(query.lower().split())
    
    for idx in sorted_indices:
        item = timestamps_metadata[idx]
        t = item["timestamp"]
        
        if any(abs(t - prev["timestamp"]) < window_sec for prev in selected):
            continue
            
        native_seg_idx = item.get("segment_idx")
        en_seg_idx = item.get("en_segment_idx")
        
        v_sim = float(visual_sims[idx]) if visual_sims is not None and len(visual_sims) > idx else 0.0
        native_sim = float(text_sims_native[native_seg_idx]) if (native_seg_idx is not None and text_sims_native is not None and len(text_sims_native) > 0 and native_seg_idx < len(text_sims_native)) else 0.0
        en_sim = float(text_sims_en[en_seg_idx]) if (en_seg_idx is not None and text_sims_en is not None and len(text_sims_en) > 0 and en_seg_idx < len(text_sims_en)) else 0.0
        t_sim = max(native_sim, en_sim)
        
        matched_context = "Visual match"
        match_type = "visual"
        exact_timestamp = t
        
        matched_seg = None
        if native_seg_idx is not None and native_seg_idx < len(transcripts):
            matched_seg = transcripts[native_seg_idx]
            matched_context = f"Speech: \"{matched_seg['text']}\""
            match_type = "transcript" if t_sim > v_sim else "fused"
        elif en_seg_idx is not None and en_seg_idx < len(transcripts_en):
            matched_seg = transcripts_en[en_seg_idx]
            matched_context = f"Translated Speech: \"{matched_seg['text']}\""
            match_type = "transcript" if t_sim > v_sim else "fused"
            
        word_found = False
        if matched_seg and "words" in matched_seg:
            for w_item in matched_seg["words"]:
                clean_w = w_item["word"].strip(".,!?\"'").lower()
                if clean_w in q_words:
                    exact_timestamp = w_item["start"]
                    word_found = True
                    break
                    
        if not word_found and en_seg_idx is not None and en_seg_idx < len(transcripts_en):
            en_seg = transcripts_en[en_seg_idx]
            if "words" in en_seg:
                for w_item in en_seg["words"]:
                    clean_w = w_item["word"].strip(".,!?\"'").lower()
                    if clean_w in q_words:
                        exact_timestamp = w_item["start"]
                        break
                        
        match_pct = round(scores[idx] * 100, 1)
        selected.append({
            "timestamp": exact_timestamp,
            "score": match_pct,
            "formatted_time": f"{int(exact_timestamp // 60):02d}:{int(exact_timestamp % 60):02d}",
            "type": match_type,
            "matched_context": matched_context,
            "frame_path": item["frame_path"],
            "visual_similarity": round(v_sim, 3),
            "text_similarity": round(t_sim, 3)
        })
        
        if len(selected) >= top_k:
            break
            
    return selected

whisper_model = None

@app.post("/voice-search")
async def voice_search(file: UploadFile = File(...)):
    if CLOUD_DEPLOY:
        hf_token = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        content = await file.read()
        
        whisper_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
        res_trans = None
        for i in range(12): # Warmup retry up to 24s
            try:
                res_trans = requests.post(whisper_url, headers=headers, data=content, timeout=30)
                if res_trans.status_code == 200:
                    break
                elif res_trans.status_code == 503:
                    time.sleep(2)
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
                
        if res_trans and res_trans.status_code == 200:
            trans_res = res_trans.json()
            text = trans_res.get("text", "").strip()
            print(f"[Voice Search] Transcribed audio via HF: '{text}'")
            return {"text": text}
        else:
            status_code = res_trans.status_code if res_trans else 500
            detail = res_trans.text if res_trans else "Failed to contact Hugging Face API"
            raise HTTPException(status_code=status_code, detail=f"Serverless voice search failed: {detail}")
            
    global whisper_model
    try:
        temp_path = os.path.join(DATA_DIR, "voice.wav")
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        if whisper_model is None:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("small", device=device, compute_type="int8" if device == "cpu" else "float16")
            
        segments, _ = whisper_model.transcribe(temp_path, vad_filter=True)
        text = " ".join([s.text.strip() for s in segments]).strip()
        print(f"[Voice Search] Transcribed local audio: '{text}'")
        return {"text": text}
    except Exception as e:
        print(f"[!] Local voice search error: {e}")
        raise HTTPException(status_code=500, detail=f"Local voice transcription failed: {e}")

@app.get("/search")
def search_moment(q: str, video: str = "lecture", top_k: int = 3, w_v: float = None, w_t: float = None):
    try:
        return _search_moment_impl(q, video, top_k, w_v, w_t)
    except Exception as err:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search endpoint exception: {str(err)}")

def _search_moment_impl(q: str, video: str = "lecture", top_k: int = 3, w_v: float = None, w_t: float = None):
    global search_cache
    cache_key = f"{video}:{q.lower().strip()}:{top_k}:{w_v}:{w_t}"
    if cache_key in search_cache:
        print(f"[*] Cache hit for query: '{q}'")
        return search_cache[cache_key]

    try:
        indices = get_video_indices(video)
    except Exception:
        indices = None

    t_start = time.time()
    
    # 1. Encode text query using CLIP and SentenceTransformer
    if CLOUD_DEPLOY:
        hf_token = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        
        clip_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/clip-ViT-B-32-multilingual-v1/pipeline/feature-extraction"
        q_clip_arr = None
        for i in range(12):
            try:
                res = requests.post(clip_url, headers=headers, json={"inputs": [q]}, timeout=15)
                if res.status_code == 200:
                    emb = extract_embedding(res.json())
                    q_clip_arr = np.array(emb, dtype=np.float32)
                    q_clip_arr /= np.linalg.norm(q_clip_arr)
                    break
                elif res.status_code == 503:
                    time.sleep(2)
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
                
        if q_clip_arr is None:
            raise HTTPException(status_code=503, detail="Hugging Face CLIP text encoder is currently cold-starting. Please try again in 5 seconds.")
            
        st_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
        q_text_arr = None
        for i in range(12):
            try:
                res = requests.post(st_url, headers=headers, json={"inputs": [q]}, timeout=15)
                if res.status_code == 200:
                    emb = extract_embedding(res.json())
                    q_text_arr = np.array(emb, dtype=np.float32)
                    q_text_arr /= np.linalg.norm(q_text_arr)
                    break
                elif res.status_code == 503:
                    time.sleep(2)
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
                
        if q_text_arr is None:
            raise HTTPException(status_code=503, detail="Hugging Face MiniLM text encoder is currently cold-starting. Please try again in 5 seconds.")
    else:
        tokens = tokenizer([q]).to(device)
        with torch.no_grad():
            q_clip_vec = clip_model.encode_text(tokens)
            q_clip_vec /= q_clip_vec.norm(dim=-1, keepdim=True)
            q_clip_arr = q_clip_vec.cpu().numpy()[0]
            
        q_text_arr = text_encoder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    
    t_enc = (time.time() - t_start) * 1000
    
    # 2. Check if we should use Qdrant Cloud
    if qdrant_client is not None:
        try:
            visual_results = qdrant_client.search(
                collection_name="visual_moments",
                query_vector=q_clip_arr.tolist(),
                query_filter=qd_models.Filter(
                    must=[qd_models.FieldCondition(key="video_id", match=qd_models.MatchValue(value=video))]
                ),
                limit=50
            )
            
            text_results = qdrant_client.search(
                collection_name="transcript_moments",
                query_vector=q_text_arr.tolist(),
                query_filter=qd_models.Filter(
                    must=[qd_models.FieldCondition(key="video_id", match=qd_models.MatchValue(value=video))]
                ),
                limit=50
            )
            
            if w_v is not None and w_t is not None:
                weight_v = float(w_v)
                weight_t = float(w_t)
            else:
                max_text_score = max([hit.score for hit in text_results]) if text_results else 0.0
                VISUAL_CUES = {"shown", "wearing", "color", "car", "crash", "goal", "board", "diagram", "gesturing", "graphics", "stairs", "sofa", "carrying", "couch", "person"}
                SPEECH_CUES = {"says", "explains", "mentions", "asks", "discusses", "teacher", "speaker", "career", "guidance"}
                q_words = set(q.lower().split())
                
                if max_text_score > 0.35:
                    weight_v, weight_t = 0.10, 0.90
                elif q_words & SPEECH_CUES:
                    weight_v, weight_t = 0.15, 0.85
                elif q_words & VISUAL_CUES:
                    weight_v, weight_t = 0.85, 0.15
                else:
                    weight_v, weight_t = 0.50, 0.50
                    
            scores_dict = {}
            for hit in visual_results:
                ts = hit.payload["timestamp"]
                scores_dict[ts] = {
                    "timestamp": ts,
                    "visual_similarity": hit.score,
                    "text_similarity": 0.0,
                    "frame_path": hit.payload.get("frame_path", ""),
                    "matched_context": "Visual match",
                    "type": "visual"
                }
                
            for hit in text_results:
                ts = hit.payload["timestamp"]
                txt = hit.payload.get("text", "")
                if ts in scores_dict:
                    scores_dict[ts]["text_similarity"] = hit.score
                    scores_dict[ts]["matched_context"] = f"Speech: \"{txt}\""
                    scores_dict[ts]["type"] = "transcript" if hit.score > scores_dict[ts]["visual_similarity"] else "fused"
                else:
                    scores_dict[ts] = {
                        "timestamp": ts,
                        "visual_similarity": 0.0,
                        "text_similarity": hit.score,
                        "frame_path": "",
                        "matched_context": f"Speech: \"{txt}\"",
                        "type": "transcript"
                    }
                    
            fused_list = []
            for ts, item in scores_dict.items():
                item["score"] = (weight_v * item["visual_similarity"]) + (weight_t * item["text_similarity"])
                fused_list.append(item)
                
            fused_list.sort(key=lambda x: x["score"], reverse=True)
            results = []
            window_sec = 4.0
            
            try:
                local_trans = indices.get("transcripts", []) if indices else []
                local_trans_en = indices.get("transcripts_en", []) if indices else []
            except Exception:
                local_trans, local_trans_en = [], []
                
            for item in fused_list:
                t = item["timestamp"]
                if any(abs(t - prev["timestamp"]) < window_sec for prev in results):
                    continue
                    
                exact_timestamp = t
                matched_seg = None
                for seg in local_trans:
                    if seg["start"] <= t <= seg["end"]:
                        matched_seg = seg
                        break
                if not matched_seg:
                    for seg in local_trans_en:
                        if seg["start"] <= t <= seg["end"]:
                            matched_seg = seg
                            break
                            
                word_found = False
                if matched_seg and "words" in matched_seg:
                    q_words = set(q.lower().split())
                    for w_item in matched_seg["words"]:
                        clean_w = w_item["word"].strip(".,!?\"'").lower()
                        if clean_w in q_words:
                            exact_timestamp = w_item["start"]
                            word_found = True
                            break
                            
                match_pct = round(item["score"] * 100, 1)
                results.append({
                    "timestamp": exact_timestamp,
                    "score": match_pct,
                    "formatted_time": f"{int(exact_timestamp // 60):02d}:{int(exact_timestamp % 60):02d}",
                    "type": item["type"],
                    "matched_context": item["matched_context"],
                    "frame_path": item["frame_path"] if item["frame_path"] else "frame_00000.jpg",
                    "visual_similarity": round(item["visual_similarity"], 3),
                    "text_similarity": round(item["text_similarity"], 3)
                })
                if len(results) >= top_k:
                    break
                    
            t_search = (time.time() - t_start) * 1000 - t_enc
            t_rank = 0.0
            total_ms = t_enc + t_search
            
            res = {
                "query": q,
                "weights": {"visual": weight_v, "text": weight_t},
                "latency_ms": round(total_ms, 1),
                "latency": {
                    "encoding_ms": round(t_enc, 1),
                    "search_ms": round(t_search, 1),
                    "ranking_ms": round(t_rank, 1),
                    "total_ms": round(total_ms, 1)
                },
                "results": results
            }
            search_cache[cache_key] = res
            return res
        except Exception as qd_err:
            print(f"[!] Qdrant Cloud Search failed: {qd_err}. Falling back to numpy...")

    # 3. Visual Similarity calculation (Matrix Multiplication Fallback)
    if indices is None:
        raise HTTPException(status_code=404, detail="Index data not found. Run indexer first.")
        
    visual_vectors = indices["visual_vectors"]
    visual_timestamps = indices["visual_timestamps"]
    transcripts = indices["transcripts"]
    transcript_embeddings = indices["transcript_embeddings"]
    transcripts_en = indices["transcripts_en"]
    transcript_en_embeddings = indices["transcript_en_embeddings"]
    
    visual_sims = np.dot(visual_vectors, q_clip_arr) if visual_vectors is not None else np.zeros(len(visual_timestamps))
    text_sims_native = np.dot(transcript_embeddings, q_text_arr) if transcript_embeddings is not None and len(transcript_embeddings) > 0 else np.array([])
    text_sims_en = np.dot(transcript_en_embeddings, q_text_arr) if transcript_en_embeddings is not None and len(transcript_en_embeddings) > 0 else np.array([])
    
    t_search = (time.time() - t_start) * 1000 - t_enc
    
    # Calculate weights
    if w_v is not None and w_t is not None:
        weight_v, weight_t = float(w_v), float(w_t)
    else:
        max_text_score = 0.0
        if len(text_sims_native) > 0:
            max_text_score = max(max_text_score, float(text_sims_native.max()))
        if len(text_sims_en) > 0:
            max_text_score = max(max_text_score, float(text_sims_en.max()))
            
        VISUAL_CUES = {"shown", "wearing", "color", "car", "crash", "goal", "board", "diagram", "gesturing", "graphics", "stairs", "sofa", "carrying", "couch", "person"}
        SPEECH_CUES = {"says", "explains", "mentions", "asks", "discusses", "teacher", "speaker", "career", "guidance"}
        q_words = set(q.lower().split())
        
        if max_text_score > 0.35:
            weight_v, weight_t = 0.10, 0.90
        elif q_words & SPEECH_CUES:
            weight_v, weight_t = 0.15, 0.85
        elif q_words & VISUAL_CUES:
            weight_v, weight_t = 0.85, 0.15
        else:
            weight_v, weight_t = 0.50, 0.50
            
    scores = np.zeros(len(visual_timestamps), dtype=np.float32)
    for i, item in enumerate(visual_timestamps):
        v_score = visual_sims[i] if visual_sims is not None and len(visual_sims) > i else 0.0
        native_seg_idx = item.get("segment_idx")
        native_score = text_sims_native[native_seg_idx] if (native_seg_idx is not None and len(text_sims_native) > 0 and native_seg_idx < len(text_sims_native)) else 0.0
        en_seg_idx = item.get("en_segment_idx")
        en_score = text_sims_en[en_seg_idx] if (en_seg_idx is not None and len(text_sims_en) > 0 and en_seg_idx < len(text_sims_en)) else 0.0
        t_score = max(native_score, en_score)
        scores[i] = (weight_v * v_score) + (weight_t * t_score)
        
    results = temporal_nms(scores, visual_timestamps, visual_sims, text_sims_native, text_sims_en, q, transcripts, transcripts_en, window_sec=4.0, top_k=top_k)
    
    t_rank = (time.time() - t_start) * 1000 - t_enc - t_search
    total_ms = t_enc + t_search + t_rank
    
    res = {
        "query": q,
        "weights": {"visual": weight_v, "text": weight_t},
        "latency_ms": round(total_ms, 1),
        "latency": {
            "encoding_ms": round(t_enc, 1),
            "search_ms": round(t_search, 1),
            "ranking_ms": round(t_rank, 1),
            "total_ms": round(total_ms, 1)
        },
        "results": results
    }
    search_cache[cache_key] = res
    return res

@app.get("/videos")
def list_videos():
    presets = [
        {"id": "lecture", "title": "EdTech CS Coding Lecture", "type": "preset", "src": "/media/sample.mp4"},
        {"id": "keynote", "title": "Apple WWDC Keynote Recap", "type": "preset", "src": "/media/keynote.mp4"},
        {"id": "friends", "title": "🛋️ Friends - Ross's Couch 'Pivot' Scene", "type": "preset", "src": "/media/friends.mp4"},
        {"id": "sports", "title": "🎬 De Dana Dan - Hindi Comedy (Dialogue Seeking)", "type": "preset", "src": "/media/sports.mp4"},
        {"id": "song", "title": "🎵 Imagine Dragons - Believer (Lyrics Seeking)", "type": "preset", "src": "/media/song.mp4"}
    ]
    return {"presets": presets, "ingested": []}

@app.get("/status")
def get_status(video: str = "lecture"):
    try:
        indices = get_video_indices(video)
        status = {
            "indexed": True,
            "frames_count": len(indices["visual_timestamps"]),
            "transcript_count": len(indices["transcripts"]),
            "device": device
        }
    except Exception:
        status = {
            "indexed": False,
            "frames_count": 0,
            "transcript_count": 0,
            "device": device
        }
    return status

def warmup_models_background():
    print("[*] Starting background warmup for Hugging Face Inference models...")
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    clip_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/clip-ViT-B-32-multilingual-v1/pipeline/feature-extraction"
    st_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    whisper_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
    
    # Warmup pings
    try:
        requests.post(clip_url, headers=headers, json={"inputs": ["warmup"]}, timeout=10)
        print("[+] Hugging Face CLIP visual model warmed up successfully.")
    except Exception as e:
        print(f"[!] CLIP warmup ping failed: {e}")
        
    try:
        requests.post(st_url, headers=headers, json={"inputs": ["warmup"]}, timeout=10)
        print("[+] Hugging Face MiniLM text model warmed up successfully.")
    except Exception as e:
        print(f"[!] MiniLM warmup ping failed: {e}")

@app.on_event("startup")
def startup_event():
    # Trigger non-blocking warmup pings in a background daemon thread
    threading.Thread(target=warmup_models_background, daemon=True).start()

    if qdrant_client:
        try:
            collections = qdrant_client.get_collections().collections
            existing = [c.name for c in collections]
            if "visual_moments" not in existing:
                qdrant_client.create_collection(
                    collection_name="visual_moments",
                    vectors_config=qd_models.VectorParams(size=512, distance=qd_models.Distance.COSINE)
                )
                print("[*] Created Qdrant collection: 'visual_moments'")
            if "transcript_moments" not in existing:
                qdrant_client.create_collection(
                    collection_name="transcript_moments",
                    vectors_config=qd_models.VectorParams(size=384, distance=qd_models.Distance.COSINE)
                )
                print("[*] Created Qdrant collection: 'transcript_moments'")
        except Exception as e:
            print(f"[!] Warning: Could not initialize Qdrant collections: {e}")

# Serve static folders
app.mount("/media", StaticFiles(directory="media"), name="media")
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/thumbnails", StaticFiles(directory="data/lecture/thumbnails"), name="thumbnails")

@app.get("/")
def index():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "JumpToThis API is running. Place index.html in frontend/ to serve client UI."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
