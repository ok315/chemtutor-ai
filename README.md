---
title: ChemTutor AI
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# ChemTutor AI

**ChemTutor AI: An Adaptive Intelligent Tutoring Agent for FSc Chemistry** — RAG-grounded explanations, audio + YouTube multimodal delivery, and an adaptive quiz powered by Gemini and fine-tuned BERT classifiers.

**Platform choice:** **Hugging Face Spaces** is used because it is built for **Gradio + ML demos**, supports **Secrets** for `GEMINI_API_KEY`, CPU/GPU runtimes, and large repos without provisioning a separate backend (Railway is better when you need long-lived APIs or databases — this app does not).

---

## Local setup

1. **Python 3.10+** recommended.
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Secrets:** create a `.env` file in the project root (optional locally):

   ```bash
   GEMINI_API_KEY=your_key_here
   ```

   The app reads **`GEMINI_API_KEY` only from the environment** (`os.environ`), via `python-dotenv` for local `.env` loading. Never commit keys.

4. **Knowledge base & models:** ensure these paths exist (see “Large assets” below):

   - `data/kb/chunks.json`, `data/kb/embeddings.npy` (build with `scripts/build_knowledge_base.py`, or let the app download embeddings on first run if you use an HF Dataset — see below)
   - `models/bert_difficulty/` (full checkpoint incl. `model.safetensors`)
   - `models/bert_bloom/` (full checkpoint incl. `model.safetensors`)

5. Run:

   ```bash
   python app.py
   ```

   Open the URL shown (locally, defaults to `127.0.0.1:7860`).

---

## Deploy on Hugging Face Spaces (step-by-step)

### Why a separate Dataset for embeddings

Hugging Face **Spaces Git** rejects **`embeddings.npy`** as a binary file (even with Git LFS). **`chunks.json`** can live in your Space repo. **`embeddings.npy`** must be uploaded to a **Hugging Face Dataset** once; at startup the app downloads it automatically (via `CHEMTUTOR_KB_DATASET`, default **`osamaaok/chemtutor-kb`** until you rename it).

### One-time: create Dataset and upload `embeddings.npy`

1. Go to **[Create new Dataset](https://huggingface.co/new-dataset)** (choose **Apache-2.0** or whatever matches your preference).
2. Name it **`chemtutor-kb`** under your username so it becomes **`osamaaok/chemtutor-kb`** (or another name — then set the repo id in **`CHEMTUTOR_KB_DATASET`** in Space **Secrets / Variables > Variables**, or rely on defaults in code if you still use **`osamaaok/chemtutor-kb`**).
3. In the Dataset **Files** tab, upload **`embeddings.npy`** from your local `data/kb/` folder (same file you generated with **`python -m scripts.build_knowledge_base`**).
4. Make the Dataset **Public** unless you configure an `HF_TOKEN` for private downloads (advanced).

### Deploy the Space

1. **Push your code** to Hugging Face (and GitHub), including **`data/kb/chunks.json`** and **`models/**`** via **Git LFS** (`*.safetensors`). Do **not** commit **`embeddings.npy`** — it stays on the Dataset.
2. In the Space **Settings → Secrets and variables**, set **`GEMINI_API_KEY`**.
3. Optionally set variable **`CHEMTUTOR_KB_DATASET`** if your Dataset id is **not** `osamaaok/chemtutor-kb`.
4. Choose **CPU Basic** hardware to start (upgrade if builds run out of memory).
5. **Restart** / wait for logs until the app listens; first boot downloads **embeddings** (~2 MB) and the **sentence-transformers** embedding model (~420 MB).

6. Open the Space URL and test **Explain a Topic**, then **Take a Quiz**.

---

## Large assets & `.gitignore`

| Asset | Approx. size | Role |
| :--- | :--- | :--- |
| `models/bert_difficulty/model.safetensors` | ~438 MB | Difficulty classifier weights |
| `models/bert_bloom/model.safetensors` | ~438 MB | Bloom classifier weights |
| `data/kb/embeddings.npy` | ~2.4 MB | Precomputed chunk embeddings |
| `data/kb/chunks.json` | ~0.8 MB | Textbook chunks + metadata |
| `data/chemistry_book.pdf` | ~10 MB | Source PDF (needed only to **rebuild** KB via script; optional at runtime if KB files exist) |

**Spaces:** **`chunks.json`** is tracked in Git. **`embeddings.npy`** is **gitignored** and loaded from **`CHEMTUTOR_KB_DATASET`** — see Deploy section above. **`models/**`** uses **Git LFS** (`*.safetensors`).

---

## Team

- Muhammad Osama Khan — BSDSF23A015  
- Muhammad Hamza Rauf — BSDSF23A040  
- Ali Hassan — BSDSF21A011  

**Course:** Deep Learning · **Instructor:** Prof. Dr. Muhammad Kamran Malik  
