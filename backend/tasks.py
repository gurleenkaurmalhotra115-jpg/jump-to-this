import os
import sys
import cv2
import json
import time
import uuid
import torch
import numpy as np
import open_clip
from PIL import Image
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qd_models

# Qdrant Cloud Setup
QDRANT_URL = os.environ.get("QDRANT_URL", None)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
DATA_DIR = "data"
MEDIA_DIR = "media"

# Lazy loaders for models
models_cache = {}

def get_models():
    if not models_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Loading AI models locally on device: {device}...")
        
        # 1. Load CLIP Model
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model.to(device).eval()
        
        # 2. Load Multilingual SentenceTransformer
        text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # 3. Load Whisper Model
        whisper = WhisperModel("small", device=device, compute_type="int8" if device == "cpu" else "float16")
        
        models_cache["device"] = device
        models_cache["clip"] = clip_model
        models_cache["clip_preprocess"] = clip_preprocess
        models_cache["text_encoder"] = text_encoder
        models_cache["whisper"] = whisper
        
    return models_cache

def set_status(video_id, status, error_msg=None):
    video_data_dir = os.path.join(DATA_DIR, video_id)
    os.makedirs(video_data_dir, exist_ok=True)
    status_path = os.path.join(video_data_dir, "status.json")
    with open(status_path, "w") as f:
        json.dump({"status": status, "error": error_msg, "updated_at": time.time()}, f)

def get_status(video_id):
    status_path = os.path.join(DATA_DIR, video_id, "status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "unknown"}

def get_qdrant_client():
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return None

def ingest_video(video_id: str, file_path: str):
    print(f"[*] Starting dynamic indexing for video: {video_id} (path: {file_path})")
    set_status(video_id, "processing")
    
    t_start = time.time()
    
    try:
        # 1. Initialize folders
        video_data_dir = os.path.join(DATA_DIR, video_id)
        thumb_dir = os.path.join(video_data_dir, "thumbnails")
        os.makedirs(video_data_dir, exist_ok=True)
        os.makedirs(thumb_dir, exist_ok=True)
        
        # Load models
        models = get_models()
        device = models["device"]
        clip_model = models["clip"]
        clip_preprocess = models["clip_preprocess"]
        text_encoder = models["text_encoder"]
        whisper = models["whisper"]
        
        # Connect to Qdrant Cloud
        qdrant_client = get_qdrant_client()
        
        # 2. Extract Keyframes (1 FPS)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at: {file_path}")
            
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = int(fps)
        
        frames = []
        raw_timestamps = []
        frame_paths = []
        frame_count = 0
        
        print("[*] Extracting frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                idx = len(frames)
                thumb_name = f"frame_{idx:05d}.jpg"
                thumb_path = os.path.join(thumb_dir, thumb_name)
                
                cv2.imwrite(thumb_path, frame)
                frame_paths.append(thumb_name)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                raw_timestamps.append(round(frame_count / fps, 2))
            frame_count += 1
        cap.release()
        
        print(f"[+] Extracted {len(frames)} frames.")
        set_status(video_id, "indexing_visual")
        
        # 3. Compute Visual Embeddings
        print("[*] Encoding visual vectors...")
        visual_embeddings = []
        qdrant_visual_points = []
        
        with torch.no_grad():
            for i, img in enumerate(frames):
                tensor = clip_preprocess(img).unsqueeze(0).to(device)
                vec = clip_model.encode_image(tensor)
                vec /= vec.norm(dim=-1, keepdim=True)
                vec_np = vec.cpu().numpy()[0]
                visual_embeddings.append(vec_np)
                
                ts = raw_timestamps[i]
                frame_path = frame_paths[i]
                
                if qdrant_client:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_visual_{ts}"))
                    qdrant_visual_points.append(
                        qd_models.PointStruct(
                            id=point_id,
                            vector=vec_np.tolist(),
                            payload={
                                "video_id": video_id,
                                "timestamp": ts,
                                "frame_path": frame_path
                            }
                        )
                    )
                
        # Save local fallback visual vectors
        np.save(os.path.join(video_data_dir, "visual_vectors.npy"), np.array(visual_embeddings, dtype=np.float32))
        
        # Bulk upsert to Qdrant Cloud if connected
        if qdrant_client and qdrant_visual_points:
            qdrant_client.upsert(collection_name="visual_moments", points=qdrant_visual_points)
            print(f"[+] Upserted {len(qdrant_visual_points)} visual vectors to Qdrant Cloud.")
        
        set_status(video_id, "transcribing")
        
        # 4. Whisper Transcribe (Native Stream)
        print("[*] Running native speech transcription...")
        native_generator, native_info = whisper.transcribe(
            file_path,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=(
                "This is an Indian technical lecture with code-mixed Hindi, English, and Hinglish. "
                "Terms: Python, C++, Java, Rust, compile, algorithm, syntax, concept, Gurutvakarshan, गुरुत्वाकर्षण, सूत्र।"
            )
        )
        native_segments = list(native_generator)
        
        transcript_data = []
        native_segment_texts = []
        for idx, s in enumerate(native_segments):
            words_list = []
            if s.words:
                for w in s.words:
                    words_list.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "prob": round(w.probability, 2)
                    })
                    
            transcript_data.append({
                "segment_idx": idx,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
                "lang": native_info.language,
                "words": words_list
            })
            native_segment_texts.append(s.text.strip())
            
        with open(os.path.join(video_data_dir, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
        # 5. Whisper Translate (English Stream)
        print("[*] Running English speech translation...")
        translation_generator, _ = whisper.transcribe(
            file_path,
            task="translate",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt="This is a translation of a technical lecture into English. Terms: Python, C++, Java, Rust."
        )
        translation_segments = list(translation_generator)
        
        transcript_en_data = []
        en_segment_texts = []
        for idx, s in enumerate(translation_segments):
            words_list = []
            if s.words:
                for w in s.words:
                    words_list.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "prob": round(w.probability, 2)
                    })
                    
            transcript_en_data.append({
                "segment_idx": idx,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
                "lang": "en",
                "words": words_list
            })
            en_segment_texts.append(s.text.strip())
            
        with open(os.path.join(video_data_dir, "transcript_en.json"), "w", encoding="utf-8") as f:
            json.dump(transcript_en_data, f, ensure_ascii=False, indent=2)
            
        set_status(video_id, "indexing_speech")
        
        # 6. Compute Speech Embeddings
        qdrant_text_points = []
        
        if transcript_data:
            print("[*] Encoding native transcript semantic embeddings...")
            transcript_embeddings = text_encoder.encode(native_segment_texts, normalize_embeddings=True)
            np.save(os.path.join(video_data_dir, "transcript_embeddings.npy"), np.array(transcript_embeddings, dtype=np.float32))
            
            if qdrant_client:
                for i, text in enumerate(native_segment_texts):
                    seg = transcript_data[i]
                    ts = seg["start"]
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_text_{i}"))
                    
                    qdrant_text_points.append(
                        qd_models.PointStruct(
                            id=point_id,
                            vector=transcript_embeddings[i].tolist(),
                            payload={
                                "video_id": video_id,
                                "timestamp": ts,
                                "text": text,
                                "segment_idx": i
                            }
                        )
                    )
        else:
            np.save(os.path.join(video_data_dir, "transcript_embeddings.npy"), np.empty((0, 384), dtype=np.float32))
            
        if transcript_en_data:
            print("[*] Encoding translated transcript semantic embeddings...")
            transcript_en_embeddings = text_encoder.encode(en_segment_texts, normalize_embeddings=True)
            np.save(os.path.join(video_data_dir, "transcript_en_embeddings.npy"), np.array(transcript_en_embeddings, dtype=np.float32))
        else:
            np.save(os.path.join(video_data_dir, "transcript_en_embeddings.npy"), np.empty((0, 384), dtype=np.float32))
            
        # Bulk upsert to Qdrant Cloud if connected
        if qdrant_client and qdrant_text_points:
            qdrant_client.upsert(collection_name="transcript_moments", points=qdrant_text_points)
            print(f"[+] Upserted {len(qdrant_text_points)} transcript vectors to Qdrant Cloud.")
            
        # 7. Pre-map keyframe timestamps to segment indices
        def find_segment_idx_at_time(t, segments):
            for idx, seg in enumerate(segments):
                if seg["start"] <= t <= seg["end"]:
                    return idx
            return None
            
        timestamps_metadata = []
        for i, t in enumerate(raw_timestamps):
            native_seg_idx = find_segment_idx_at_time(t, transcript_data)
            en_seg_idx = find_segment_idx_at_time(t, transcript_en_data)
            timestamps_metadata.append({
                "timestamp": t,
                "frame_path": frame_paths[i],
                "segment_idx": native_seg_idx,
                "en_segment_idx": en_seg_idx
            })
            
        with open(os.path.join(video_data_dir, "visual_timestamps.json"), "w") as f:
            json.dump(timestamps_metadata, f)
            
        duration = time.time() - t_start
        print(f"[+] Dynamic Ingestion completed successfully for {video_id} in {duration:.1f}s.")
        set_status(video_id, "completed")
        
    except Exception as e:
        print(f"[!] Dynamic Ingestion failed: {e}")
        set_status(video_id, "failed", error_msg=str(e))
