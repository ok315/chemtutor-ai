"""
ChemTutor AI — Gradio web UI.

Wraps the Phase 1-6 components into a browser-based tutoring app.

Run:
    python app.py

Then open the URL printed in the terminal (usually http://localhost:7860).
"""

import gradio as gr

from src.knowledge_base.retriever import Retriever
from src.generation.explainer import Explainer
from src.multimodal.audio import text_to_audio
from src.multimodal.video import search_videos
from src.quiz.engine import QuizEngine, QuizSession
from src.config import QUIZ_MAX_COUNT, QUIZ_MIN_COUNT, QUIZ_FINAL_COUNT


# ============================================================
# Component initialization (loads once at startup)
# ============================================================

print("=" * 60)
print("ChemTutor AI — starting up")
print("=" * 60)

print("\n[1/3] Loading shared retriever...")
retriever = Retriever()

print("\n[2/3] Loading explainer (uses shared retriever)...")
explainer = Explainer(retriever=retriever)

print("\n[3/3] Loading quiz engine (uses shared retriever)...")
quiz_engine = QuizEngine(retriever=retriever)

print("\nAll components loaded. Starting Gradio UI...\n")


# Render up to this many question slots in the UI.
NUM_QUIZ_QUESTIONS = QUIZ_MAX_COUNT


# ============================================================
# About tab — project overview (demo / deployment)
# ============================================================

ABOUT_MARKDOWN = """
## ChemTutor AI: An Adaptive Intelligent Tutoring Agent for FSc Chemistry

**Tagline:** *Grounded explanations, multi-modal learning, and quizzes that adapt to you.*

---

### What it does

ChemTutor AI supports **F.Sc Part 2 (2nd Year) Chemistry students in Pakistan** with a full learning loop:

- **RAG-grounded teaching** — Explanations are built from your chunked **F.Sc Chemistry textbook** using retrieval + the **Gemini** language model, so answers stay tied to syllabus content.
- **Multi-modal delivery** — Read the explanation, **listen** with **gTTS** audio, and explore **curated YouTube** links biased toward Class 12 / Pakistan context.
- **Adaptive quiz engine** — Gemini drafts practice questions; **fine-tuned BERT** models classify **difficulty** (SciQ + ARC–style supervision) and **Bloom’s taxonomy** (LOTS vs HOTS), so question selection matches your chosen level.
- **Adaptive feedback** — After each quiz you get scores, per-question feedback, and a recommendation for **easier / same / harder** practice next time.

---

### How to use it (student guide)

1. **Pick your focus** — In **Explain a Topic**, type a **chapter-related topic** from the F.Sc syllabus (your “chapter” choice is expressed as the topic you search for).
2. **Get the explanation** — Click **Explain** and read the structured, textbook-grounded answer (with **Sources** when available).
3. **Listen** — Play the **audio** track to revise on the go.
4. **Watch** — Open **recommended YouTube** videos for secondary explanations and visuals.
5. **Quiz yourself** — Under **Take a Quiz**, set level and format, then **Start Quiz**, answer every question, and **Submit All Answers**.
6. **Adaptive feedback** — Read per-question feedback and the **summary** with **next-step** guidance for your level.

---

### Tech stack

| Layer | Tools |
| :--- | :--- |
| **LLM & grading** | **Google Gemini API** — explanations, quiz generation, and free-text answer evaluation (`GEMINI_API_KEY` via environment only). |
| **Semantic retrieval** | **sentence-transformers** (MPNet) embeddings + **NumPy** cosine similarity over `chunks.json` / `embeddings.npy`. |
| **Classification** | Fine-tuned **BERT** checkpoints under `models/bert_difficulty` and `models/bert_bloom`. |
| **Audio / video** | **gTTS** · **`youtube-search`** (same ecosystem as *youtube-search-python*) |
| **PDF / KB build** | **pdfplumber** (offline ingestion scripts). |
| **UI** | **Gradio** |

---

### Team & course

**Team**

- **Muhammad Osama Khan** — BSDSF23A015  
- **Muhammad Hamza Rauf** — BSDSF23A040  
- **Ali Hassan** — BSDSF21A011  

**Course:** Deep Learning  

**Submitted to:** Prof. Dr. Muhammad Kamran Malik  

---

### Note for reviewers

This demo runs **fully in the browser** via Gradio. No separate database server is required; quiz **sessions** exist only in memory for the current visit.
"""


# ============================================================
# Tab 1: Topic Explainer
# ============================================================

_explanation_cache: dict[str, tuple[str, str | None, str]] = {}


def explain_topic(topic: str) -> tuple[str, str | None, str]:
    """Generate explanation + audio + video links for a chemistry topic."""
    if not topic or not topic.strip():
        return ("Please enter a chemistry topic.", None, "")

    topic = topic.strip()

    cache_key = topic.lower()
    if cache_key in _explanation_cache:
        print(f"[cache hit] '{topic}' — skipping Gemini call")
        return _explanation_cache[cache_key]

    try:
        result = explainer.explain(topic)
        explanation_text = result["explanation"]
        chapters = result.get("sources", [])

        if chapters:
            chapter_str = "\n".join(f"- {ch}" for ch in chapters)
            explanation_md = (
                f"{explanation_text}\n\n"
                f"---\n\n"
                f"**📚 Sources:**\n{chapter_str}"
            )
        else:
            explanation_md = explanation_text
    except Exception as e:
        return (f"⚠️ Explanation failed: {e}", None, "")

    audio_path = None
    try:
        audio_path_obj = text_to_audio(explanation_text)
        audio_path = str(audio_path_obj)
    except Exception as e:
        print(f"Audio generation failed: {e}")

    videos_md = ""
    try:
        videos = search_videos(topic)
        if videos:
            video_lines = ["### 🎥 Recommended videos\n"]
            for i, v in enumerate(videos, start=1):
                video_lines.append(
                    f"**{i}. [{v['title']}]({v['url']})**  \n"
                    f"_{v['channel']} · {v['duration']} · {v['view_count']}_\n"
                )
            videos_md = "\n".join(video_lines)
        else:
            videos_md = "_No videos found for this topic._"
    except Exception as e:
        videos_md = f"_Video search unavailable: {e}_"

    final_result = (explanation_md, audio_path, videos_md)
    _explanation_cache[cache_key] = final_result
    return final_result


# ============================================================
# Tab 2: Adaptive Quiz
# ============================================================

def start_quiz(
    topic: str, level: str, count: int, mode: str
) -> tuple:
    """
    Generate a new quiz session.

    Returns a tuple matching the bound output components, including
    visibility updates for question groups, textbox groups, and radio groups.
    """

    def _empty_return(status_msg: str) -> tuple:
        empty_q = ["_Quiz not started yet._"] * NUM_QUIZ_QUESTIONS
        empty_textbox_values = [""] * NUM_QUIZ_QUESTIONS
        empty_radio_values = [gr.update(choices=[], value=None)] * NUM_QUIZ_QUESTIONS
        empty_text_grp = [gr.update(visible=False)] * NUM_QUIZ_QUESTIONS
        empty_radio_grp = [gr.update(visible=False)] * NUM_QUIZ_QUESTIONS
        empty_r = [""] * NUM_QUIZ_QUESTIONS
        group_visibility = [gr.update(visible=False)] * NUM_QUIZ_QUESTIONS
        return (
            None,
            status_msg,
            *empty_q,
            *empty_textbox_values,
            *empty_radio_values,
            *empty_text_grp,
            *empty_radio_grp,
            *empty_r,
            *group_visibility,
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if not topic or not topic.strip():
        return _empty_return("⚠️ Please enter a chemistry topic.")

    count = max(QUIZ_MIN_COUNT, min(QUIZ_MAX_COUNT, int(count)))

    try:
        session = quiz_engine.start_quiz(
            topic.strip(),
            student_level=level,
            final_count=count,
            mode=mode,
        )
    except Exception as e:
        return _empty_return(f"⚠️ Failed to start quiz: {e}")

    actual_count = len(session.questions)
    question_displays: list[str] = []
    textbox_values: list[str] = []
    radio_value_updates: list = []
    text_grp_updates: list = []
    radio_grp_updates: list = []
    group_visibility: list = []

    for i in range(NUM_QUIZ_QUESTIONS):
        if i < actual_count:
            q = session.questions[i]
            question_displays.append(
                f"**Q{i+1}.** {q.text}\n\n"
                f"<sub>_Difficulty: {q.difficulty} · Cognitive level: {q.bloom}_</sub>"
            )
            group_visibility.append(gr.update(visible=True))

            if q.is_mcq and q.mcq_data is not None:
                # MCQ — show radio group, hide textbox group
                radio_choices = [
                    (f"{chr(ord('A')+j)}. {opt}", j)
                    for j, opt in enumerate(q.mcq_data.options)
                ]
                radio_value_updates.append(
                    gr.update(choices=radio_choices, value=None)
                )
                radio_grp_updates.append(gr.update(visible=True))
                textbox_values.append("")
                text_grp_updates.append(gr.update(visible=False))
            else:
                # Free-text — show textbox group, hide radio group
                textbox_values.append("")
                text_grp_updates.append(gr.update(visible=True))
                radio_value_updates.append(gr.update(choices=[], value=None))
                radio_grp_updates.append(gr.update(visible=False))
        else:
            # Slot beyond actual question count
            question_displays.append("_(no question)_")
            textbox_values.append("")
            radio_value_updates.append(gr.update(choices=[], value=None))
            text_grp_updates.append(gr.update(visible=False))
            radio_grp_updates.append(gr.update(visible=False))
            group_visibility.append(gr.update(visible=False))

    mode_label = "MCQ" if mode == "mcq" else "free-text"
    return (
        session,
        f"✅ Quiz ready! {actual_count} {mode_label} questions on **{topic}** at **{level}** level.",
        *question_displays,
        *textbox_values,
        *radio_value_updates,
        *text_grp_updates,
        *radio_grp_updates,
        *([""] * NUM_QUIZ_QUESTIONS),       # clear results
        *group_visibility,
        "",                                  # clear summary
        gr.update(visible=True),             # submit button
        gr.update(visible=True),             # reset button
    )


def submit_quiz(
    session: QuizSession | None,
    *all_inputs,
) -> tuple:
    """
    Grade all answers and return per-question feedback + summary.

    *all_inputs is structured as:
      - First NUM_QUIZ_QUESTIONS values: textbox strings
      - Next NUM_QUIZ_QUESTIONS values: radio integers (or None)
    """
    if session is None:
        empty_results = [""] * NUM_QUIZ_QUESTIONS
        return ("⚠️ Start a quiz first.", *empty_results, "")

    text_answers = list(all_inputs[:NUM_QUIZ_QUESTIONS])
    radio_answers = list(all_inputs[NUM_QUIZ_QUESTIONS : 2 * NUM_QUIZ_QUESTIONS])

    actual_count = len(session.questions)
    results = []

    for i in range(NUM_QUIZ_QUESTIONS):
        if i >= actual_count:
            results.append("")
            continue

        q = session.questions[i]

        # Pick the right input based on this question's mode
        if q.is_mcq:
            answer = radio_answers[i] if i < len(radio_answers) else None
        else:
            answer = text_answers[i] if i < len(text_answers) else ""

        try:
            result = session.submit_answer(i, answer)
            verdict_emoji = (
                "✅" if result.is_correct
                else "🟡" if result.is_partial
                else "❌"
            )
            results.append(
                f"{verdict_emoji} **{result.score}/100** ({result.verdict})  \n"
                f"_{result.feedback}_"
            )
        except Exception as e:
            results.append(f"⚠️ Grading failed: {e}")

    summary = session.get_summary()
    summary_md = (
        f"## 📊 Quiz Summary\n\n"
        f"**Topic:** {summary.topic}  \n"
        f"**Level:** {summary.student_level}  \n"
        f"**Score:** {summary.average_score:.1f} / 100  \n"
        f"**Correct:** {summary.correct_count}  ·  "
        f"**Partial:** {summary.partial_count}  ·  "
        f"**Incorrect:** {summary.incorrect_count}\n\n"
        f"---\n\n"
        f"### 💡 Recommendation\n"
        f"{summary.recommendation}"
    )
    if summary.next_level:
        summary_md += f"\n\n_Next time, try: **{summary.next_level}** level._"

    return (
        f"✅ Quiz graded! Average score: {summary.average_score:.1f} / 100",
        *results,
        summary_md,
    )


def reset_quiz() -> tuple:
    """Reset quiz UI to initial empty state."""
    empty_q = ["_Quiz not started yet._"] * NUM_QUIZ_QUESTIONS
    empty_textbox_values = [""] * NUM_QUIZ_QUESTIONS
    empty_radio_values = [gr.update(choices=[], value=None)] * NUM_QUIZ_QUESTIONS
    empty_text_grp = [gr.update(visible=False)] * NUM_QUIZ_QUESTIONS
    empty_radio_grp = [gr.update(visible=False)] * NUM_QUIZ_QUESTIONS
    empty_r = [""] * NUM_QUIZ_QUESTIONS
    group_visibility = [gr.update(visible=False)] * NUM_QUIZ_QUESTIONS
    return (
        None,
        "Click **Start Quiz** to begin.",
        *empty_q,
        *empty_textbox_values,
        *empty_radio_values,
        *empty_text_grp,
        *empty_radio_grp,
        *empty_r,
        *group_visibility,
        "",
        gr.update(visible=False),
        gr.update(visible=False),
    )


# ============================================================
# Custom CSS for polished dark theme
# ============================================================

CUSTOM_CSS = """
/* GLOBAL BASE */
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%) !important;
    min-height: 100vh;
    color: #f1f5f9 !important;
}
.gradio-container * {
    color: #f1f5f9 !important;
}

/* Header */
#app-header {
    text-align: center;
    padding: 24px 0 8px 0;
    background: transparent;
}
#app-header h1 {
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    color: transparent !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
#app-header h3 {
    color: #cbd5e1 !important;
    font-weight: 400 !important;
    margin-top: 0;
}

/* Tabs */
.tab-nav button {
    background: rgba(30, 41, 59, 0.5) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 12px !important;
    margin: 0 4px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
}
.tab-nav button:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
}
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
}

/* Cards / panels / input wrappers */
.gr-block,
.gr-box,
.gr-form,
.gr-panel,
.block,
.form,
.gradio-container .form,
.gradio-container .block,
[data-testid="block-info"],
.wrap.svelte-1ipelgc,
.input-container {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
}
.gradio-container div.form,
.gradio-container .form > div,
.gradio-container fieldset {
    background: transparent !important;
}

/* Text inputs */
input[type="text"],
input[type="number"],
textarea,
.gr-textbox textarea,
.gr-textbox input,
.gr-text-input {
    background: rgba(15, 23, 42, 0.7) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 8px !important;
}
input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 0.8;
}
input:focus, textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    outline: none !important;
}

/* Primary buttons */
.gr-button-primary,
button.primary,
.primary {
    background: linear-gradient(90deg, #6366f1, #ec4899) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(236, 72, 153, 0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.gr-button-primary:hover,
button.primary:hover,
.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(236, 72, 153, 0.5) !important;
    color: #ffffff !important;
}

/* Secondary buttons */
.gr-button-secondary,
button.secondary,
.secondary {
    background: rgba(30, 41, 59, 0.7) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148, 163, 184, 0.4) !important;
    border-radius: 10px !important;
}
.gr-button-secondary:hover,
button.secondary:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: #818cf8 !important;
    color: #ffffff !important;
}

/* Markdown content */
.gr-markdown,
.gr-markdown p,
.gr-markdown li,
.gr-markdown span,
.markdown,
.markdown p {
    color: #e2e8f0 !important;
}
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4,
.markdown h1, .markdown h2, .markdown h3, .markdown h4 {
    color: #ffffff !important;
}
.gr-markdown strong, .markdown strong {
    color: #ffffff !important;
}
.gr-markdown em, .markdown em {
    color: #cbd5e1 !important;
}
.gr-markdown a, .markdown a {
    color: #93c5fd !important;
    text-decoration: underline;
}
.gr-markdown a:hover, .markdown a:hover {
    color: #dbeafe !important;
}
.gr-markdown code, .markdown code {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #c4b5fd !important;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Labels */
label,
.gr-label,
.gr-form > label,
.label-wrap,
.label-wrap span,
.gr-text-input + label,
[data-testid="textbox-label"],
span.svelte-1gfkn6j,
.block-label {
    color: #ffffff !important;
    font-weight: 700 !important;
    background: transparent !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}
.gr-radio label,
fieldset label {
    text-shadow: none !important;
}

/* Examples chips */
.gr-examples button,
.examples-table button,
.gr-examples td,
.examples-table td {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
.gr-examples button:hover,
.examples-table button:hover {
    background: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    border-color: #a78bfa !important;
}

/* Radio buttons */
.gr-radio,
fieldset.gr-radio {
    background: transparent !important;
}
.gr-radio label,
fieldset label {
    background: rgba(30, 41, 59, 0.5) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    margin: 4px !important;
    cursor: pointer !important;
}
.gr-radio label:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    color: #ffffff !important;
    border-color: #818cf8 !important;
}
.gr-radio input:checked + label,
.gr-radio input:checked ~ * {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
}

/* Audio player */
audio {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 8px !important;
    width: 100%;
}

/* Number/slider inputs */
.gr-number input,
.gr-slider input {
    color: #f1f5f9 !important;
    background: rgba(15, 23, 42, 0.7) !important;
}

/* Hide Gradio footer */
footer {
    display: none !important;
}
"""


# ============================================================
# Build the Gradio UI
# ============================================================

with gr.Blocks(
    title="ChemTutor AI",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as app:
    gr.HTML(
        """
        <div id="app-header">
            <h1>🧪 ChemTutor AI</h1>
            <h3>Your AI Chemistry Tutor for F.Sc Class 12 (Pakistan)</h3>
        </div>
        """
    )

    with gr.Tabs():

        # ============================================================
        # TAB 1 — Topic Explainer
        # ============================================================
        with gr.Tab("📖  Explain a Topic"):
            gr.Markdown(
                "### 💡 Get a textbook-grounded explanation\n"
                "Enter a chemistry topic from your F.Sc syllabus. "
                "You'll get a detailed explanation, an audio version you can listen to, "
                "and recommended YouTube videos from Pakistani educators."
            )

            with gr.Row():
                topic_input = gr.Textbox(
                    label="Chemistry topic",
                    placeholder="e.g., esterification, alkali metals, electrolysis",
                    lines=1,
                    scale=4,
                )
                explain_btn = gr.Button(
                    "✨  Explain",
                    variant="primary",
                    scale=1,
                )

            gr.Examples(
                examples=[
                    "esterification",
                    "alkali metals",
                    "electrolysis",
                    "noble gases",
                    "transition elements",
                ],
                inputs=topic_input,
                label="💡 Try an example",
            )

            explanation_output = gr.Markdown(
                value="_Your explanation will appear here..._",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    audio_output = gr.Audio(
                        label="🔊 Listen to the explanation",
                        type="filepath",
                    )
                with gr.Column(scale=1):
                    videos_output = gr.Markdown(
                        value="_Video links will appear here..._",
                    )

            explain_btn.click(
                fn=explain_topic,
                inputs=topic_input,
                outputs=[explanation_output, audio_output, videos_output],
            )

        # ============================================================
        # TAB 2 — Adaptive Quiz
        # ============================================================
        with gr.Tab("📝  Take a Quiz"):
            gr.Markdown(
                "### 🎯 Test your understanding\n"
                "Get questions adapted to your level. Choose **free-text** for "
                "deeper feedback, or **MCQ** for quick recall practice."
            )

            quiz_state = gr.State(value=None)

            # === Quiz controls ===
            with gr.Row():
                with gr.Column(scale=2):
                    quiz_topic_input = gr.Textbox(
                        label="Chemistry topic",
                        placeholder="e.g., esterification, alkali metals",
                        lines=1,
                    )
                with gr.Column(scale=2):
                    quiz_level_input = gr.Radio(
                        label="Your level",
                        choices=["beginner", "intermediate", "advanced"],
                        value="intermediate",
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    quiz_mode_input = gr.Radio(
                        label="Quiz format",
                        choices=[
                            ("📝 Free-text (deeper feedback)", "free_text"),
                            ("🔘 MCQ (faster)", "mcq"),
                        ],
                        value="free_text",
                    )
                with gr.Column(scale=2):
                    quiz_count_input = gr.Slider(
                        label="Number of questions",
                        minimum=QUIZ_MIN_COUNT,
                        maximum=QUIZ_MAX_COUNT,
                        value=QUIZ_FINAL_COUNT,
                        step=1,
                    )
                with gr.Column(scale=1):
                    start_quiz_btn = gr.Button(
                        "🚀  Start Quiz",
                        variant="primary",
                    )

            quiz_status = gr.Markdown(value="Click **Start Quiz** to begin.")

            # === Per-question lists ===
            question_displays: list[gr.Markdown] = []
            answer_textbox_inputs: list[gr.Textbox] = []
            answer_radio_inputs: list[gr.Radio] = []
            answer_textbox_groups: list[gr.Group] = []
            answer_radio_groups: list[gr.Group] = []
            result_displays: list[gr.Markdown] = []
            question_groups: list[gr.Group] = []

            for i in range(NUM_QUIZ_QUESTIONS):
                with gr.Group(visible=False) as q_group:
                    q_md = gr.Markdown(value="_Quiz not started yet._")

                    # Wrap textbox in its own group for clean visibility toggling
                    with gr.Group(visible=False) as text_group:
                        a_text = gr.Textbox(
                            label=f"Your answer to Q{i+1}",
                            placeholder="Type your answer here...",
                            lines=2,
                        )

                    # Wrap radio in its own group for clean visibility toggling
                    with gr.Group(visible=False) as radio_group:
                        a_radio = gr.Radio(
                            label=f"Select your answer for Q{i+1}",
                            choices=[],
                        )

                    r_md = gr.Markdown(value="")

                question_displays.append(q_md)
                answer_textbox_inputs.append(a_text)
                answer_radio_inputs.append(a_radio)
                answer_textbox_groups.append(text_group)
                answer_radio_groups.append(radio_group)
                result_displays.append(r_md)
                question_groups.append(q_group)

            # === Submit + Reset buttons ===
            with gr.Row():
                submit_quiz_btn = gr.Button(
                    "✅  Submit All Answers",
                    variant="primary",
                    visible=False,
                )
                reset_quiz_btn = gr.Button(
                    "🔄  Reset Quiz",
                    variant="secondary",
                    visible=False,
                )

            summary_output = gr.Markdown(value="")

            # === Wire up the buttons ===
            start_quiz_btn.click(
                fn=start_quiz,
                inputs=[
                    quiz_topic_input,
                    quiz_level_input,
                    quiz_count_input,
                    quiz_mode_input,
                ],
                outputs=[
                    quiz_state,
                    quiz_status,
                    *question_displays,
                    *answer_textbox_inputs,        # textbox values
                    *answer_radio_inputs,          # radio values+choices
                    *answer_textbox_groups,        # textbox group visibility
                    *answer_radio_groups,          # radio group visibility
                    *result_displays,
                    *question_groups,
                    summary_output,
                    submit_quiz_btn,
                    reset_quiz_btn,
                ],
            )

            submit_quiz_btn.click(
                fn=submit_quiz,
                inputs=[
                    quiz_state,
                    *answer_textbox_inputs,
                    *answer_radio_inputs,
                ],
                outputs=[
                    quiz_status,
                    *result_displays,
                    summary_output,
                ],
            )

            reset_quiz_btn.click(
                fn=reset_quiz,
                inputs=[],
                outputs=[
                    quiz_state,
                    quiz_status,
                    *question_displays,
                    *answer_textbox_inputs,
                    *answer_radio_inputs,
                    *answer_textbox_groups,
                    *answer_radio_groups,
                    *result_displays,
                    *question_groups,
                    summary_output,
                    submit_quiz_btn,
                    reset_quiz_btn,
                ],
            )

        # ============================================================
        # TAB 3 — About
        # ============================================================
        with gr.Tab("ℹ️  About"):
            gr.Markdown(ABOUT_MARKDOWN)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "7860"))
    # Hugging Face Spaces / Docker set PORT and expect binding on all interfaces.
    if os.environ.get("PORT"):
        host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
        open_browser = False
    else:
        host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
        open_browser = True

    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=open_browser,
    )