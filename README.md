# JumpToThis: Dual-Embedding Moment Search Engine 🎬

JumpToThis is an advanced, lightweight dual-embedding video moment search engine. It indexes video frames (using CLIP) and transcripts (using SentenceTransformers) to enable semantic search across video content, allowing users to search and jump directly to the exact timestamp of interest.

---

## 🚀 Key Features

* **Dual-Embedding Search**: Combines visual embeddings (CLIP `ViT-B-32`) and text embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) for multimodal video search.
* **Hybrid Search Interface**: Query videos visually (based on what is happening on screen) or textual (based on what is spoken in the transcript).
* **FastAPI Backend**: High-performance backend supporting PyTorch locally or lightweight Hugging Face Inference API cloud deployments.
* **Responsive Frontend Dashboard**: Simple, intuitive user interface styled with custom CSS grid layouts.
* **Dockerized Setup**: Containerized using Docker and Docker Compose for easy deployment.

---

## 🛠️ Tech Stack

* **Backend**: FastAPI (Python 3.11), PyTorch, NumPy, OpenCLIP, Sentence-Transformers, Uvicorn
* **Frontend**: HTML5, CSS3 (Flexbox/Grid), Vanilla JavaScript (ES6+)
* **DevOps**: Docker, Docker Compose

---

## 📁 Repository Structure

```
├── backend/                  # FastAPI Application
│   ├── app.py                # Main backend controller & API endpoints
│   ├── indexer.py            # Script to generate video indexes & embeddings
│   ├── requirements.txt      # Core python dependencies (Local PyTorch execution)
│   └── requirements-cloud.txt# Lightweight dependency config for cloud deployment
├── frontend/                 # Static web dashboard
│   ├── index.html            # Main search client
│   └── style.css             # Glassmorphic user interface styling
├── voice sheild/             # Voice shield module workspace
│   ├── README.md             # Profile page customization readme
│   └── profile.png           # Profile photo asset
├── docker-compose.yml        # Multi-container local deployment
├── Dockerfile                # Backend container configuration
├── git_push.bat              # Script to push changes to GitHub
└── run_live.bat              # Script to execute backend & host locally
```

---

## ⚙️ Quick Start

### Option 1: Run Locally using Virtual Environment

1. **Set up Python Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

2. **Index your video content**:
   ```bash
   python backend/indexer.py --video path/to/your/video.mp4
   ```

3. **Start the API server**:
   ```bash
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```

4. Open `frontend/index.html` in your web browser.

### Option 2: Run via Docker Compose

```bash
docker-compose up --build
```

The app will compile the environment and become available locally.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open a Pull Request or report an Issue.
