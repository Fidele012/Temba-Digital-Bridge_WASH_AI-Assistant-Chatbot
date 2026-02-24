# app.py
# ==============================================================================
# Temba Digital Bridge — WASH Assistant (Gradio 6 compatible)
# - Keeps the same UI layout + CSS styling you provided
# - Fixes Gradio 6 Chatbot history format (uses type="messages")
# - Loads a base model + optional LoRA adapter from Hugging Face Hub
# - Works on Hugging Face Spaces (Python 3.13)
# ==============================================================================

import os
import re
import time
from datetime import datetime

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional LoRA/PEFT support (recommended for LoRA adapters)
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# -----------------------------
# Config (HF Spaces friendly)
# -----------------------------
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Ndihokubwayo/Temba_Digital_Bridge_AI_Assistant_Chatbot",
)
# If your adapter is in a different repo, set ADAPTER_ID. If adapter files live in the same repo as MODEL_ID,
# leave ADAPTER_ID empty and the app will try to load adapters from MODEL_ID automatically.
ADAPTER_ID = os.environ.get("ADAPTER_ID", "").strip()

# Force CPU if needed
FORCE_CPU = os.environ.get("FORCE_CPU", "0").strip() == "1"

DEVICE = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"


# -----------------------------
# OOD (Out-of-domain) handling
# -----------------------------
OUT_OF_DOMAIN_RESPONSE = (
    "I’m specialized in water, sanitation, infrastructure, and public health topics. "
    "This question seems to be outside my area of expertise. Please contact our team for assistance with other topics. "
    "If your concern relates to water safety, sanitation, hygiene, or infrastructure, kindly rephrase your question and I’ll gladly assist you."
)

WASH_KEYWORDS = {
    "wash", "water", "drinking", "drinkable", "potable", "contamination", "contaminated",
    "chlorine", "chlorination", "bleach", "filtration", "filter", "boil", "boiling",
    "purify", "purification", "disinfect", "disinfection",
    "sanitation", "latrine", "toilet", "sewer", "sewage", "wastewater", "drainage",
    "handwashing", "hygiene", "soap",
    "well", "borehole", "pump", "pipeline", "tap", "standpipe", "storage", "tank",
    "fecal", "faecal", "diarrhea", "diarrhoea", "cholera", "typhoid",
    "public", "health", "disease", "waterborne"
}

def _is_in_domain(query: str) -> bool:
    q = (query or "").lower()
    words = set(re.findall(r"[a-z]+", q))
    return len(words.intersection(WASH_KEYWORDS)) > 0


# -----------------------------
# UI CSS (exactly as provided)
# -----------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--blue:#0369A1;--blue-light:#0EA5E9;--blue-pale:#E0F2FE;--teal:#0D9488;
      --teal-pale:#CCFBF1;--navy:#0C2340;--slate:#475569;--slate-lt:#94A3B8;
      --cream:#F0F9FF;--white:#fff;--red:#DC2626;--red-pale:#FEE2E2;
      --r:10px;--shadow:0 2px 12px rgba(12,35,64,.10);}
body,.gradio-container{font-family:'Plus Jakarta Sans',sans-serif!important;background:var(--cream)!important;color:var(--navy)!important;}
/* Header */
#hdr{background:linear-gradient(135deg,var(--navy) 0%,#0D4A8A 60%,#1564A8 100%);border-bottom:3px solid var(--blue-light);padding:22px 32px 18px;position:relative;overflow:hidden;}
#hdr::after{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 85% 50%,rgba(14,165,233,.18) 0%,transparent 60%);pointer-events:none;}
#hdr h1{font-size:1.75rem!important;font-weight:700!important;color:#fff!important;margin:0!important;letter-spacing:-.3px;}
#hdr p{font-size:.85rem;color:var(--blue-light);margin-top:4px;font-weight:300;}
.badge{display:inline-flex;align-items:center;background:rgba(14,165,233,.22);border:1px solid rgba(14,165,233,.4);border-radius:20px;padding:2px 10px;font-size:.7rem;color:var(--blue-light);font-weight:600;letter-spacing:.7px;text-transform:uppercase;margin-bottom:8px;}
/* Disclaimer */
#disc{background:var(--red-pale);border:1.5px solid #FECACA;border-left:4px solid var(--red);border-radius:6px;padding:9px 14px;font-size:.8rem;color:#7F1D1D;margin:10px 18px 0;}
/* Layout */
#wrap{padding:14px 18px;display:flex;gap:14px;}
/* Chat panel */
#chat-col{flex:8;background:var(--white);border:1px solid #BFDBFE;border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden;}
#chat-box{background:#F8FBFF!important;border:none!important;border-bottom:1px solid #E0F2FE!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.9rem!important;line-height:1.65!important;}
#chat-box .message.user{background:linear-gradient(135deg,var(--navy),#0D4A8A)!important;color:#ffffff!important;border-radius:16px 16px 3px 16px!important;padding:10px 15px!important;max-width:76%!important;margin-left:auto!important;}
#chat-box .message.user *,#chat-box .message.user p,#chat-box .message.user span{color:#ffffff!important;}
#chat-box .message.bot{background:var(--white)!important;color:var(--navy)!important;border:1px solid #BFDBFE!important;border-radius:3px 16px 16px 16px!important;padding:12px 15px!important;max-width:86%!important;}
#chat-box .message.bot em{color:var(--slate-lt)!important;font-size:.76rem!important;font-style:normal!important;font-family:'JetBrains Mono',monospace!important;}
#chat-box .message.bot hr{border-color:#E0F2FE!important;margin:6px 0 3px!important;}
/* Input */
#inp-area{padding:12px 16px;background:var(--white);}
#msg-box textarea{font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.9rem!important;border:1.5px solid #BAD4EF!important;border-radius:var(--r)!important;background:#F8FBFF!important;padding:10px 14px!important;transition:border-color .2s,box-shadow .2s!important;line-height:1.5!important;}
#msg-box textarea:focus{border-color:var(--blue)!important;box-shadow:0 0 0 3px rgba(3,105,161,.1)!important;background:#fff!important;outline:none!important;}
#msg-box label{font-size:.74rem!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:.7px!important;color:var(--slate)!important;}
/* Buttons */
#sub-btn{background:linear-gradient(135deg,var(--blue),var(--blue-light))!important;color:#fff!important;border:none!important;border-radius:var(--r)!important;font-weight:600!important;font-size:.88rem!important;padding:11px 20px!important;box-shadow:0 2px 8px rgba(3,105,161,.3)!important;transition:all .2s!important;width:100%!important;}
#sub-btn:hover{transform:translateY(-1px)!important;box-shadow:0 4px 14px rgba(3,105,161,.4)!important;filter:brightness(1.06)!important;}
#clr-btn{background:transparent!important;color:var(--slate)!important;border:1.5px solid #CBD5E1!important;border-radius:var(--r)!important;font-size:.85rem!important;padding:9px 16px!important;transition:all .2s!important;width:100%!important;}
#clr-btn:hover{background:#F1F5F9!important;color:var(--navy)!important;}
/* Sidebar */
#side{flex:3;display:flex;flex-direction:column;gap:12px;}
.card{background:var(--white);border:1px solid #BFDBFE;border-radius:var(--r);padding:14px;box-shadow:var(--shadow);}
.ctitle{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.9px;color:var(--slate);margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #EFF6FF;}
.srow{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F8FBFF;font-size:.81rem;}
.srow:last-child{border-bottom:none;}
.slbl{color:var(--slate);}
.sval{color:var(--navy);font-weight:600;font-family:'JetBrains Mono',monospace;font-size:.78rem;}
.sval.g{color:var(--teal);}
/* Examples */
.ecat{font-size:.69rem;font-weight:700;color:var(--blue);text-transform:uppercase;letter-spacing:.7px;padding:8px 0 4px;border-bottom:1px solid var(--blue-pale);margin-bottom:4px;}
.ex-btn button{background:#F8FBFF!important;border:1px solid #BFDBFE!important;border-radius:6px!important;color:var(--navy)!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.8rem!important;padding:6px 10px!important;text-align:left!important;width:100%!important;cursor:pointer!important;margin-bottom:3px!important;transition:all .15s!important;line-height:1.4!important;}
.ex-btn button:hover{background:var(--blue-pale)!important;border-color:var(--blue-light)!important;color:#075985!important;transform:translateX(3px)!important;}
/* Tips */
.tip{display:flex;gap:8px;align-items:flex-start;padding:5px 0;font-size:.8rem;color:#475569;border-bottom:1px solid #F8FBFF;line-height:1.4;}
.tip:last-child{border-bottom:none;}
.tic{flex-shrink:0;margin-top:1px;}
/* Footer */
#ftr{text-align:center;padding:13px;font-size:.75rem;color:var(--slate-lt);border-top:1px solid #BFDBFE;background:var(--white);}
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:3px;}
"""


# -----------------------------
# Example prompts (same as yours)
# -----------------------------
EXAMPLES = {
    "💧 Water Safety": [
        "How do I safely treat drinking water at home?",
        "How do I use chlorine to purify a well?",
        "How can I identify contaminated water?",
    ],
    "🧼 Hygiene & Sanitation": [
        "Why is handwashing important for public health?",
        "What are the best hygiene practices to prevent disease?",
        "How should household waste be disposed of safely?",
    ],
    "🦠 Waterborne Diseases": [
        "What are the early symptoms of Cholera?",
        "How is typhoid fever transmitted and prevented?",
        "What household steps can prevent diarrhoeal disease?",
    ],
}


# -----------------------------
# Model loading
# -----------------------------
def _load_model_and_tokenizer():
    """
    Loads:
    - Base model from MODEL_ID
    - Optional LoRA adapter from ADAPTER_ID (or from MODEL_ID if adapter files exist there)
    """
    print(f"Loading tokenizer from: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Ensure pad token exists
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model from: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # Try to attach LoRA adapter if available
    if _HAS_PEFT:
        adapter_source = ADAPTER_ID if ADAPTER_ID else MODEL_ID
        try:
            print(f"Trying to load LoRA adapter from: {adapter_source}")
            model = PeftModel.from_pretrained(model, adapter_source)
            print("LoRA adapter loaded successfully.")
        except Exception as e:
            print(f"No adapter loaded (this is OK if your repo is already merged). Reason: {type(e).__name__}: {e}")
    else:
        print("PEFT not installed; skipping LoRA adapter load.")

    model.eval()

    # Move to device if not using device_map
    if DEVICE == "cpu":
        model.to("cpu")

    return model, tok


optimized_model, tokenizer = _load_model_and_tokenizer()


# -----------------------------
# Inference function (Gradio 6 messages format)
# -----------------------------
def process_temba_query(user_message, chat_history, max_tokens, temperature, rep_penalty):
    """
    Gradio 6 Chatbot expects: list of dicts with {"role": "...", "content": "..."} when type="messages"
    Uses Alpaca prompt format to match training schema.
    Appends response-time metadata for transparency.
    Includes OOD refusal message.
    """
    if chat_history is None:
        chat_history = []

    if not user_message or not user_message.strip():
        return "", chat_history

    user_message = user_message.strip()

    # Add user message
    chat_history = chat_history + [{"role": "user", "content": user_message}]

    # OOD gate
    if not _is_in_domain(user_message):
        meta = f"\n\n---\n*⏱ 0.0s · 0 tokens · {datetime.now().strftime('%H:%M')} · Temba v1.0*"
        chat_history = chat_history + [{"role": "assistant", "content": OUT_OF_DOMAIN_RESPONSE + meta}]
        return "", chat_history

    formatted_input = f"### Instruction:\n{user_message}\n\n### Response:\n"
    inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=512)

    # Send tensors to device
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    output_ids = None

    try:
        with torch.no_grad():
            output_ids = optimized_model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                repetition_penalty=float(rep_penalty),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][prompt_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

        raw = re.sub(r"(?<!\n)(\d+\.\s)", r"\n\1", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

        bot_response = raw if raw else "I could not generate a response. Please rephrase your question."

    except Exception as e:
        bot_response = f"⚠️ Error: {type(e).__name__} — {str(e)}"

    elapsed = time.time() - t0
    tok_count = int(len(output_ids[0]) - prompt_len) if output_ids is not None else 0
    meta = f"\n\n---\n*⏱ {elapsed:.1f}s · {tok_count} tokens · {datetime.now().strftime('%H:%M')} · Temba v1.0*"

    chat_history = chat_history + [{"role": "assistant", "content": bot_response + meta}]
    return "", chat_history


# -----------------------------
# Build the Gradio interface (same elements/layout)
# -----------------------------
temba_theme = gr.themes.Base(font=gr.themes.GoogleFont("Plus Jakarta Sans"))

with gr.Blocks(
    title="Temba Digital Bridge — WASH Assistant",
    analytics_enabled=False,
) as temba_ui:

    # Header
    gr.HTML("""
    <div id="hdr">
        <div class="badge">💧 WASH · AI · Community Health</div>
        <h1>💧 Temba Digital Bridge</h1>
        <p>AI-powered guidance on Water, Sanitation &amp; Hygiene (WASH) to prevent waterborne diseases</p>
    </div>""")

    # Instructions (1)
    gr.Markdown("""
    ### 📖 How to use this Assistant:
    1. **Ask a specific question** about water safety, hygiene practices, or disease symptoms (e.g., Cholera).
    2. **Review the professional guidance** provided in the chat window.
    3. **Use the 'Clear' button** to start a new inquiry at any time.

    *Note: This AI is a support tool and does not replace professional medical diagnosis.*
    """)

    # Disclaimer
    gr.HTML("""
    <div id="disc">⚠️ <strong>Disclaimer:</strong> Temba is an educational AI tool.
    It does <strong>not</strong> replace professional medical or public-health advice.
    Always consult a qualified health worker for personal health decisions.</div>""")

    # Instructions (2)
    gr.Markdown("""
    ### How to Use This Chatbot
    1️⃣ **Click inside the input box below.** Place your cursor in the message field to begin typing.
    2️⃣ **Type your question about water, sanitation, or public health.** For example: *How can I make drinking water safe at home?*
    3️⃣ **Click the "Submit" button (or press Enter).** This sends your question to the chatbot.
    4️⃣ **Wait a few seconds for the response.** The chatbot will generate an answer automatically.
    5️⃣ **If needed, ask another question.**
    """)

    # Main layout (kept for CSS structure)
    gr.HTML('<div id="wrap">', visible=False)

    with gr.Row(equal_height=False):

        # Chat column
        with gr.Column(scale=8, elem_id="chat-col"):
            chatbot = gr.Chatbot(
                label="Temba Conversation",
                elem_id="chat-box",
                height=460,
                show_label=False,
                show_copy_button=True,
                type="messages",  # ✅ Gradio 6 messages format
            )

            with gr.Group(elem_id="inp-area"):
                msg = gr.Textbox(
                    placeholder="e.g., How can I safely treat drinking water at home?",
                    label="Your WASH Question",
                    lines=2,
                    max_lines=5,
                    elem_id="msg-box",
                    autofocus=True,
                )
                with gr.Row():
                    submit_btn = gr.Button("🚀  Submit Question", variant="primary", elem_id="sub-btn", scale=3)
                    clear_btn = gr.Button("🧹  Clear", variant="secondary", elem_id="clr-btn", scale=1)

        # Sidebar
        with gr.Column(scale=3):

            # Model info card (static text to match your UI)
            gr.HTML("""
            <div class="card">
                <div class="ctitle">⚙️ Model Info</div>
                <div class="srow"><span class="slbl">Base Model</span><span class="sval">TinyLlama-1.1B</span></div>
                <div class="srow"><span class="slbl">Fine-tuning</span><span class="sval g">LoRA + QLoRA</span></div>
                <div class="srow"><span class="slbl">Domain</span><span class="sval">WASH</span></div>
                <div class="srow"><span class="slbl">Prompt Format</span><span class="sval">Alpaca</span></div>
                <div class="srow"><span class="slbl">LoRA Rank</span><span class="sval">r = 16</span></div>
                <div class="srow"><span class="slbl">Status</span><span class="sval g">● Online</span></div>
            </div>""")

            # Generation settings
            with gr.Accordion("🎛️ Generation Settings", open=False):
                temperature = gr.Slider(
                    0.1, 1.0, value=0.2, step=0.05,
                    label="Temperature",
                    info="Lower = more factual. 0.1–0.3 recommended for WASH."
                )
                max_tokens = gr.Slider(
                    64, 512, value=256, step=32,
                    label="Max Tokens",
                    info="Maximum response length."
                )
                rep_penalty = gr.Slider(
                    1.0, 1.5, value=1.1, step=0.05,
                    label="Repetition Penalty",
                    info="Reduces repeated phrases. 1.1 recommended."
                )

            # Categorised examples
            with gr.Accordion("💡 Example Questions  (click to use)", open=True):
                for cat, qs in EXAMPLES.items():
                    gr.HTML(f"<div class='ecat'>{cat}</div>")
                    for q in qs:
                        b = gr.Button(q, elem_classes=["ex-btn"], size="sm")
                        b.click(fn=lambda x=q: x, inputs=[], outputs=[msg], queue=False)

            # Tips card
            gr.HTML("""
            <div class="card">
                <div class="ctitle">💡 How to Get the Best Results</div>
                <div class="tip"><span class="tic">🎯</span>
                    <span><strong>Be specific</strong> — mention the water source, disease, or hygiene practice.</span></div>
                <div class="tip"><span class="tic">🔄</span>
                    <span><strong>Rephrase</strong> if the answer seems off — small wording changes help.</span></div>
                <div class="tip"><span class="tic">🌡️</span>
                    <span>Use <strong>Temperature 0.1–0.2</strong> for precise, factual answers.</span></div>
                <div class="tip"><span class="tic">🧹</span>
                    <span>Use <strong>Clear</strong> to start a completely fresh conversation.</span></div>
                <div class="tip"><span class="tic">🚫</span>
                    <span>Do <strong>not</strong> use for personal medical diagnosis or emergencies.</span></div>
            </div>""")

    # Footer
    gr.HTML("""
    <div id="ftr">
        Temba Digital Bridge &nbsp;|&nbsp; Fine-tuned TinyLlama-1.1B · LoRA · WASH Domain
        &nbsp;|&nbsp; Educational use only &nbsp;|&nbsp; © 2025
    </div>""")

    # Event wiring
    gen_inputs = [msg, chatbot, max_tokens, temperature, rep_penalty]
    submit_btn.click(fn=process_temba_query, inputs=gen_inputs, outputs=[msg, chatbot], queue=True)
    msg.submit(fn=process_temba_query, inputs=gen_inputs, outputs=[msg, chatbot], queue=True)
    clear_btn.click(fn=lambda: ([], ""), inputs=[], outputs=[chatbot, msg], queue=False)


# -----------------------------
# Launch (Gradio 6: pass theme/css here)
# -----------------------------
if __name__ == "__main__":
    temba_ui.launch(
        share=True,
        debug=True,
        theme=temba_theme,
        css=CSS,
    )