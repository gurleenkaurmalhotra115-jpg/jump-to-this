import os
import sys
import cv2
import json
import torch
import numpy as np
import open_clip
from PIL import Image
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

# Ensure ffmpeg in virtual environment is discoverable
ffmpeg_dir = os.path.abspath(os.path.dirname(sys.executable))
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video", default="lecture", choices=["lecture", "keynote", "friends", "sports", "song"])
args = parser.parse_args()

if args.video == "lecture":
    VIDEO_PATH = os.path.join("media", "sample.mp4")
    DATA_DIR = os.path.join("data", "lecture")
elif args.video == "keynote":
    VIDEO_PATH = os.path.join("media", "keynote.mp4")
    DATA_DIR = os.path.join("data", "keynote")
elif args.video == "friends":
    VIDEO_PATH = os.path.join("media", "friends.mp4")
    DATA_DIR = os.path.join("data", "friends")
elif args.video == "sports":
    VIDEO_PATH = os.path.join("media", "sports.mp4")
    DATA_DIR = os.path.join("data", "sports")
elif args.video == "song":
    VIDEO_PATH = os.path.join("media", "song.mp4")
    DATA_DIR = os.path.join("data", "song")

THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Indexing engine running on: {device}")

# 1. Load CLIP Model
print("[*] Loading CLIP ViT-B-32 weights (pretrained='openai')...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.to(device).eval()

# 2. Load Multilingual Sentence Transformer
print("[*] Loading Multilingual SentenceTransformer 'paraphrase-multilingual-MiniLM-L12-v2'...")
text_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 3. Extract Real Keyframes (1 FPS)
if not os.path.exists(VIDEO_PATH):
    print(f"[!] Error: Video file not found at: {VIDEO_PATH}")
    print("[!] Please download a test video and place it at media/sample.mp4")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
interval = int(fps)

frames = []
raw_timestamps = []
frame_paths = []
frame_count = 0

print(f"[*] Extracting frames & saving thumbnails to {THUMB_DIR}...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        idx = len(frames)
        thumb_name = f"frame_{idx:05d}.jpg"
        thumb_path = os.path.join(THUMB_DIR, thumb_name)
        
        cv2.imwrite(thumb_path, frame)
        frame_paths.append(thumb_name)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        raw_timestamps.append(round(frame_count / fps, 2))
    frame_count += 1
cap.release()

print(f"[+] Total frames captured: {len(frames)}")

# 4. Compute Real Visual Embeddings
print("[*] Encoding visual vectors...")
visual_embeddings = []
with torch.no_grad():
    for img in frames:
        tensor = clip_preprocess(img).unsqueeze(0).to(device)
        vec = clip_model.encode_image(tensor)
        vec /= vec.norm(dim=-1, keepdim=True)
        visual_embeddings.append(vec.cpu().numpy()[0])

np.save(os.path.join(DATA_DIR, "visual_vectors.npy"), np.array(visual_embeddings, dtype=np.float32))

# 5. Whisper Setup
print("[*] Loading Whisper model 'small'...")
whisper = WhisperModel("small", device=device, compute_type="int8" if device == "cpu" else "float16")

# Stream 1: Native Transcription (Hinglish/Hindi text)
print("[*] Running native speech transcription...")
native_generator, native_info = whisper.transcribe(
    VIDEO_PATH,
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

with open(os.path.join(DATA_DIR, "transcript.json"), "w", encoding="utf-8") as f:
    json.dump(transcript_data, f, ensure_ascii=False, indent=2)

# Stream 2: English Translation (English translated text)
print("[*] Running English speech translation...")
translation_generator, _ = whisper.transcribe(
    VIDEO_PATH,
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

with open(os.path.join(DATA_DIR, "transcript_en.json"), "w", encoding="utf-8") as f:
    json.dump(transcript_en_data, f, ensure_ascii=False, indent=2)

# 6. Compute Semantic Embeddings
if transcript_data:
    print("[*] Encoding native transcript semantic embeddings...")
    transcript_embeddings = text_encoder.encode(native_segment_texts, normalize_embeddings=True)
    np.save(os.path.join(DATA_DIR, "transcript_embeddings.npy"), np.array(transcript_embeddings, dtype=np.float32))
else:
    np.save(os.path.join(DATA_DIR, "transcript_embeddings.npy"), np.empty((0, 384), dtype=np.float32))

if transcript_en_data:
    print("[*] Encoding translated transcript semantic embeddings...")
    transcript_en_embeddings = text_encoder.encode(en_segment_texts, normalize_embeddings=True)
    np.save(os.path.join(DATA_DIR, "transcript_en_embeddings.npy"), np.array(transcript_en_embeddings, dtype=np.float32))
else:
    np.save(os.path.join(DATA_DIR, "transcript_en_embeddings.npy"), np.empty((0, 384), dtype=np.float32))

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

with open(os.path.join(DATA_DIR, "visual_timestamps.json"), "w") as f:
    json.dump(timestamps_metadata, f)

print(f"[+] Real Index Saved! Total speech chunks: {len(transcript_data)}")
