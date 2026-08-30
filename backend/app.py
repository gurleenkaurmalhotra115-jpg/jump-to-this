import os
import sys
import json
import time
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
import tasks

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


# Global model references
clip_model = None
tokenizer = None
text_encoder = None
device = "cpu"

if not CLOUD_DEPLOY:
    # Local Mode: Load heavy PyTorch libraries
    import torch
    torch.set_num_threads(4)  # Optimize PyTorch CPU thread pools to prevent contention
    import open_clip
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load CLIP Model
    print("[*] Loading CLIP Model...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.to(device).eval()
    
    # 2. Load Multilingual SentenceTransformer Model
    print("[*] Loading SentenceTransformer 'paraphrase-multilingual-MiniLM-L12-v2'...")
    text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("[*] Models loaded successfully.")
else:
    print("[*] Running in CLOUD_DEPLOY (lightweight API mode) under 50MB RAM!")
    if not os.environ.get("HF_TOKEN"):
        print("[WARNING] ⚠️ CLOUD_DEPLOY is active, but HF_TOKEN is not set in environment variables!")
        print("[WARNING] ⚠️ The Hugging Face Inference API requires a token and will return 401 Unauthorized.")
        print("[WARNING] ⚠️ Please set the HF_TOKEN environment variable in your deployment settings.")

def extract_embedding(data):
    """Recursively extract the 1D list of floats from a nested list response."""
    while isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        data = data[0]
    return data

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "thumbnails"), exist_ok=True)
os.makedirs("media", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# Global cache for loaded indexes and query searches
loaded_indices = {}
search_cache = {}

def get_video_indices(video: str):
    """Dynamically gets or loads the index data for a specific video."""
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
        
        if not os.path.exists(vec_path) or not os.path.exists(ts_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Index data files for '{video}' not found. Run indexer for this video first."
            )
            
        try:
            v_vecs = np.load(vec_path)
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
            
            # Assertions to ensure data consistency
            assert v_vecs.shape[0] == len(v_timestamps), f"Visual vector/timestamp mismatch for {video}!"
            if trans and t_embeddings is not None:
                assert t_embeddings.shape[0] == len(trans), f"Transcript embedding/segment mismatch for {video}!"
            if trans_en and t_en_embeddings is not None:
                assert t_en_embeddings.shape[0] == len(trans_en), f"Transcript EN embedding/segment mismatch for {video}!"
                
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
    """Temporal Non-Maximum Suppression to group contiguous seconds into 1 scene, seeking to the exact word."""
    sorted_indices = np.argsort(scores)[::-1]
    selected = []
    
    q_words = set(query.lower().split())
    
    for idx in sorted_indices:
        item = timestamps_metadata[idx]
        t = item["timestamp"]
        
        # Skip if within NMS window of already selected times
        if any(abs(t - prev["timestamp"]) < window_sec for prev in selected):
            continue
            
        native_seg_idx = item.get("segment_idx")
        en_seg_idx = item.get("en_segment_idx")
        
        v_sim = float(visual_sims[idx])
        
        native_sim = float(text_sims_native[native_seg_idx]) if (native_seg_idx is not None and text_sims_native is not None and len(text_sims_native) > 0 and native_seg_idx < len(text_sims_native)) else 0.0
        en_sim = float(text_sims_en[en_seg_idx]) if (en_seg_idx is not None and text_sims_en is not None and len(text_sims_en) > 0 and en_seg_idx < len(text_sims_en)) else 0.0
        
        t_sim = max(native_sim, en_sim)
        
        matched_context = "Visual match"
        match_type = "visual"
        
        # Start targeted timestamp as the timeline frame second
        exact_timestamp = t
        
        # Resolve transcript segment context (prefer native, fallback to translation)
        matched_seg = None
        if native_seg_idx is not None and native_seg_idx < len(transcripts):
            matched_seg = transcripts[native_seg_idx]
            matched_context = f"Speech: \"{matched_seg['text']}\""
            match_type = "transcript" if t_sim > v_sim else "fused"
        elif en_seg_idx is not None and en_seg_idx < len(transcripts_en):
            matched_seg = transcripts_en[en_seg_idx]
            matched_context = f"Translated Speech: \"{matched_seg['text']}\""
            match_type = "transcript" if t_sim > v_sim else "fused"
            
        # Pinpoint exact timestamp using word-level alignment if exact word matches
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

# Global variable for voice search transcription model
whisper_model = None

@app.post("/voice-search")
async def voice_search(file: UploadFile = File(...)):
    """Transcribes client microphone audio block locally using the small Whisper model."""
    if CLOUD_DEPLOY:
        raise HTTPException(status_code=501, detail="Voice search is disabled in cloud hosting to save RAM. Please type your query.")
    global whisper_model
    try:
        temp_path = os.path.join(DATA_DIR, "voice.wav")
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        if whisper_model is None:
            print("[*] Lazily loading WhisperModel for local voice search...")
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("small", device=device, compute_type="int8" if device == "cpu" else "float16")
            
        # Run local transcription
        segments, _ = whisper_model.transcribe(temp_path, vad_filter=True)
        text = " ".join([s.text.strip() for s in segments]).strip()
        print(f"[Voice Search] Transcribed local audio: '{text}'")
        return {"text": text}
    except Exception as e:
        print(f"[!] Local voice search error: {e}")
        raise HTTPException(status_code=500, detail=f"Local voice transcription failed: {e}")

@app.get("/search")
def search_moment(q: str, video: str = "lecture", top_k: int = 3):
    global search_cache
    cache_key = f"{video}:{q.lower().strip()}:{top_k}"
    if cache_key in search_cache:
        print(f"[*] Cache hit for query: '{q}'")
        return search_cache[cache_key]

    indices = get_video_indices(video)
    
    visual_vectors = indices["visual_vectors"]
    visual_timestamps = indices["visual_timestamps"]
    transcripts = indices["transcripts"]
    transcript_embeddings = indices["transcript_embeddings"]
    transcripts_en = indices["transcripts_en"]
    transcript_en_embeddings = indices["transcript_en_embeddings"]
    
    t_start = time.time()
    
    # 1. Encode text query using CLIP and SentenceTransformer
    if CLOUD_DEPLOY:
        hf_token = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        
        # Encode CLIP (512-dim) via HF API
        clip_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/clip-ViT-B-32-multilingual-v1/pipeline/feature-extraction"
        q_clip_arr = None
        last_clip_error = None
        for _ in range(5):
            try:
                res = requests.post(clip_url, headers=headers, json={"inputs": [q]}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    emb = extract_embedding(data)
                    q_clip_arr = np.array(emb, dtype=np.float32)
                    q_clip_arr /= np.linalg.norm(q_clip_arr)
                    break
                elif res.status_code == 503:
                    time.sleep(3)
                else:
                    last_clip_error = f"HF API Status {res.status_code}: {res.text}"
                    break
            except Exception as e:
                last_clip_error = str(e)
                time.sleep(1)
                
        if q_clip_arr is None:
            if last_clip_error:
                error_msg = f"Failed to encode CLIP query via Hugging Face Inference API ({last_clip_error})."
                if "401" in last_clip_error or "Unauthorized" in last_clip_error:
                    error_msg += " Please set a valid HF_TOKEN environment variable in your deployment configuration."
                raise HTTPException(status_code=500, detail=error_msg)
            q_clip_arr = np.zeros(512, dtype=np.float32)
            
        # Encode SentenceTransformer (384-dim) via HF API
        st_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
        q_text_arr = None
        last_st_error = None
        for _ in range(5):
            try:
                res = requests.post(st_url, headers=headers, json={"inputs": [q]}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    emb = extract_embedding(data)
                    q_text_arr = np.array(emb, dtype=np.float32)
                    q_text_arr /= np.linalg.norm(q_text_arr)
                    break
                elif res.status_code == 503:
                    time.sleep(3)
                else:
                    last_st_error = f"HF API Status {res.status_code}: {res.text}"
                    break
            except Exception as e:
                last_st_error = str(e)
                time.sleep(1)
                
        if q_text_arr is None:
            if last_st_error:
                error_msg = f"Failed to encode text query via Hugging Face Inference API ({last_st_error})."
                if "401" in last_st_error or "Unauthorized" in last_st_error:
                    error_msg += " Please set a valid HF_TOKEN environment variable in your deployment configuration."
                raise HTTPException(status_code=500, detail=error_msg)
            q_text_arr = np.zeros(384, dtype=np.float32)
    else:
        # Local Mode: Encode text query using CLIP (512-dim)
        tokens = tokenizer([q]).to(device)
        with torch.no_grad():
            q_clip_vec = clip_model.encode_text(tokens)
            q_clip_vec /= q_clip_vec.norm(dim=-1, keepdim=True)
            q_clip_arr = q_clip_vec.cpu().numpy()[0]
            
        # Local Mode: Encode text query using SentenceTransformer (384-dim)
        q_text_arr = text_encoder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    
    t_enc = (time.time() - t_start) * 1000
    
    # 2. Check if we should use Qdrant Cloud
    if qdrant_client is not None:
        try:
            # Query Qdrant Visual Collection
            visual_results = qdrant_client.search(
                collection_name="visual_moments",
                query_vector=q_clip_arr.tolist(),
                query_filter=qd_models.Filter(
                    must=[qd_models.FieldCondition(key="video_id", match=qd_models.MatchValue(value=video))]
                ),
                limit=50
            )
            
            # Query Qdrant Transcript Collection
            text_results = qdrant_client.search(
                collection_name="transcript_moments",
                query_vector=q_text_arr.tolist(),
                query_filter=qd_models.Filter(
                    must=[qd_models.FieldCondition(key="video_id", match=qd_models.MatchValue(value=video))]
                ),
                limit=50
            )
            
            # Fuse scores
            scores_dict = {}
            max_text_score = max([hit.score for hit in text_results]) if text_results else 0.0
            
            VISUAL_CUES = {"shown", "wearing", "color", "car", "crash", "goal", "board", "diagram", "gesturing", "graphics", "stairs", "sofa", "carrying", "couch", "person"}
            SPEECH_CUES = {"says", "explains", "mentions", "asks", "discusses", "teacher", "speaker", "career", "guidance"}
            q_words = set(q.lower().split())
            
            if max_text_score > 0.35:
                w_v, w_t = 0.10, 0.90
            elif q_words & SPEECH_CUES:
                w_v, w_t = 0.15, 0.85
            elif q_words & VISUAL_CUES:
                w_v, w_t = 0.85, 0.15
            else:
                w_v, w_t = 0.50, 0.50
                
            # Populate scores
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
                    
            # Combine scores
            fused_list = []
            for ts, item in scores_dict.items():
                item["score"] = (w_v * item["visual_similarity"]) + (w_t * item["text_similarity"])
                fused_list.append(item)
                
            fused_list.sort(key=lambda x: x["score"], reverse=True)
            results = []
            window_sec = 4.0
            
            # Load local transcripts if available for word alignment
            try:
                local_trans = indices.get("transcripts", [])
                local_trans_en = indices.get("transcripts_en", [])
            except Exception:
                local_trans, local_trans_en = [], []
                
            for item in fused_list:
                t = item["timestamp"]
                if any(abs(t - prev["timestamp"]) < window_sec for prev in results):
                    continue
                    
                exact_timestamp = t
                matched_seg = None
                
                # Retrieve local segment for word alignment if available
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
                "weights": {"visual": w_v, "text": w_t},
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
            print(f"[!] Qdrant Cloud Search error: {qd_err}. Falling back to local numpy match...")

    # 3. Visual Similarity calculation (Matrix Multiplication)
    visual_sims = np.dot(visual_vectors, q_clip_arr)
    
    # 4. Transcript Semantic Similarity (Native + Translated English)
    text_sims_native = np.array([])
    if transcript_embeddings is not None and len(transcript_embeddings) > 0:
        text_sims_native = np.dot(transcript_embeddings, q_text_arr)
        
    text_sims_en = np.array([])
    if transcript_en_embeddings is not None and len(transcript_en_embeddings) > 0:
        text_sims_en = np.dot(transcript_en_embeddings, q_text_arr)
        
    t_search = (time.time() - t_start) * 1000 - t_enc
    
    # 5. Dynamic Weight Intent Routing based on actual match strength
    max_text_score = 0.0
    if len(text_sims_native) > 0:
        max_text_score = max(max_text_score, float(text_sims_native.max()))
    if len(text_sims_en) > 0:
        max_text_score = max(max_text_score, float(text_sims_en.max()))
        
    VISUAL_CUES = {"shown", "wearing", "color", "car", "crash", "goal", "board", "diagram", "gesturing", "graphics", "stairs", "sofa", "carrying", "couch", "person"}
    SPEECH_CUES = {"says", "explains", "mentions", "asks", "discusses", "teacher", "speaker", "career", "guidance"}
    q_words = set(q.lower().split())
    
    if max_text_score > 0.35:
        # Strong dialogue semantic similarity -> dynamically pivot to speech-heavy weights!
        w_v, w_t = 0.10, 0.90
    elif q_words & SPEECH_CUES:
        w_v, w_t = 0.15, 0.85
    elif q_words & VISUAL_CUES:
        w_v, w_t = 0.85, 0.15
    else:
        # Default balanced weights for general searches
        w_v, w_t = 0.50, 0.50
        
    # 6. Weight Fusion & Score Calculation
    scores = np.zeros(len(visual_timestamps), dtype=np.float32)
    for i, item in enumerate(visual_timestamps):
        v_score = visual_sims[i]
        
        native_seg_idx = item.get("segment_idx")
        native_score = text_sims_native[native_seg_idx] if (native_seg_idx is not None and len(text_sims_native) > 0 and native_seg_idx < len(text_sims_native)) else 0.0
        
        en_seg_idx = item.get("en_segment_idx")
        en_score = text_sims_en[en_seg_idx] if (en_seg_idx is not None and len(text_sims_en) > 0 and en_seg_idx < len(text_sims_en)) else 0.0
        
        t_score = max(native_score, en_score)
        scores[i] = (w_v * v_score) + (w_t * t_score)
        
    # 7. Run Non-Maximum Suppression (NMS)
    results = temporal_nms(scores, visual_timestamps, visual_sims, text_sims_native, text_sims_en, q, transcripts, transcripts_en, window_sec=4.0, top_k=top_k)
    
    t_rank = (time.time() - t_start) * 1000 - t_enc - t_search
    total_ms = t_enc + t_search + t_rank
    
    res = {
        "query": q,
        "weights": {
            "visual": w_v,
            "text": w_t
        },
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

@app.get("/debug-hf")
def debug_hf(q: str = "hello"):
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    # 1. Test CLIP via HF API
    clip_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/clip-ViT-B-32-multilingual-v1/pipeline/feature-extraction"
    res_clip = requests.post(clip_url, headers=headers, json={"inputs": [q]})
    
    # 2. Test ST via HF API
    st_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    res_st = requests.post(st_url, headers=headers, json={"inputs": [q]})
    
    clip_data = res_clip.json() if res_clip.status_code == 200 else str(res_clip.text)
    st_data = res_st.json() if res_st.status_code == 200 else str(res_st.text)
    
    clip_shape = "None"
    if isinstance(clip_data, list):
        clip_shape = f"list length={len(clip_data)}"
        if len(clip_data) > 0 and isinstance(clip_data[0], list):
            clip_shape += f", sublist={len(clip_data[0])}"
            if len(clip_data[0]) > 0 and isinstance(clip_data[0][0], list):
                clip_shape += f", subsublist={len(clip_data[0][0])}"
                
    st_shape = "None"
    if isinstance(st_data, list):
        st_shape = f"list length={len(st_data)}"
        if len(st_data) > 0 and isinstance(st_data[0], list):
            st_shape += f", sublist={len(st_data[0])}"
            if len(st_data[0]) > 0 and isinstance(st_data[0][0], list):
                st_shape += f", subsublist={len(st_data[0][0])}"
                
    return {
        "clip_status": res_clip.status_code,
        "clip_shape": clip_shape,
        "clip_sample": str(clip_data)[:300],
        "st_status": res_st.status_code,
        "st_shape": st_shape,
        "st_sample": str(st_data)[:300]
    }

@app.post("/api/videos/ingest-url")
def ingest_url(payload: dict, background_tasks: BackgroundTasks):
    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing video URL in payload")
    
    video_id = f"video_{int(time.time())}"
    file_ext = ".mp4" # Default to mp4 extension
    file_path = os.path.join("media", f"{video_id}{file_ext}")
    
    # Define downloader callback
    def download_and_process():
        try:
            tasks.set_status(video_id, "downloading")
            print(f"[*] Downloading remote video: {url} -> {file_path}")
            res = requests.get(url, stream=True, timeout=30)
            if res.status_code != 200:
                raise Exception(f"Failed to fetch video: HTTP {res.status_code}")
                
            with open(file_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            print(f"[+] Download complete: {file_path}. Starting ingestion...")
            tasks.ingest_video(video_id, file_path)
        except Exception as e:
            print(f"[!] Dynamic indexing failed: {e}")
            tasks.set_status(video_id, "failed", error_msg=str(e))

    # Trigger background thread execution
    background_tasks.add_task(download_and_process)
    return {"video_id": video_id, "status": "processing"}

@app.get("/api/videos/{video_id}/status")
def get_video_ingest_status(video_id: str):
    import tasks
    return tasks.get_status(video_id)

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

@app.on_event("startup")
def startup_event():
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
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = backend_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
