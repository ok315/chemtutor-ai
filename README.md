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

   - `data/kb/chunks.json`, `data/kb/embeddings.npy`
   - `models/bert_difficulty/` (full checkpoint incl. `model.safetensors`)
   - `models/bert_bloom/` (full checkpoint incl. `model.safetensors`)

5. Run:

   ```bash
   python app.py
   ```

   Open the URL shown (locally, defaults to `127.0.0.1:7860`).

---

## Deploy on Hugging Face Spaces (step-by-step)

1. **Push your code** to a GitHub/GitLab repo **including** `data/kb/` artifacts and `models/` checkpoints (see `.gitignore` note below — you may need to **adjust `.gitignore`** or use **Git LFS** for files over ~100MB).

2. Log in to [Hugging Face](https://huggingface.co/), click **Create new Space**.

3. Choose **Gradio** SDK, link your repo (or upload files), set **App file** to **`app.py`**.

4. Under **Settings → Repository secrets**, add:

   - Name: `GEMINI_API_KEY`  
   - Value: your Google AI Studio / Gemini API key.

   Spaces inject secrets into **`os.environ`** before `python app.py` runs — no code changes required.

5. **Hardware:** start with **CPU Basic**. If startup time or inference is slow, upgrade to **CPU Upgrade** or **GPU** in Space settings.

6. **Build:** trigger a new build (push a commit or **Factory reboot**). Wait until logs show `Running on local URL`.

7. Open your public Space URL and test **Explain a Topic** then **Take a Quiz**.

**Notes:**

- First cold start downloads **`sentence-transformers/all-mpnet-base-v2`** (~420MB) into the HF cache — allow several minutes.
- **gTTS** needs outbound network access for audio (enabled by default on Spaces).

---

## Large assets & `.gitignore`

| Asset | Approx. size | Role |
| :--- | :--- | :--- |
| `models/bert_difficulty/model.safetensors` | ~438 MB | Difficulty classifier weights |
| `models/bert_bloom/model.safetensors` | ~438 MB | Bloom classifier weights |
| `data/kb/embeddings.npy` | ~2.4 MB | Precomputed chunk embeddings |
| `data/kb/chunks.json` | ~0.8 MB | Textbook chunks + metadata |
| `data/chemistry_book.pdf` | ~10 MB | Source PDF (needed only to **rebuild** KB via script; optional at runtime if KB files exist) |

The default `.gitignore` ignores **`data/`**, so **`chunks.json` / `embeddings.npy` will not deploy** unless you change ignore rules or ship those files via **Git LFS**, a **release artifact**, or **HF Dataset** + download script.

**Recommended:** track `data/kb/*.json`, `data/kb/*.npy`, and `models/**` with **Git LFS**, or narrow `.gitignore` to exclude only `data/chemistry_book.pdf` / `data/audio/` if you prefer not to publish the raw PDF.

---

## Team

- Muhammad Osama Khan — BSDSF23A015  
- Muhammad Hamza Rauf — BSDSF23A040  
- Ali Hassan — BSDSF21A011  

**Course:** Deep Learning · **Instructor:** Prof. Dr. Muhammad Kamran Malik  
