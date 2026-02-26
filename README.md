# Temba-Digital-Bridge_WASH_AI-Assistant-Chatbot
A domain-specific generative AI assistant designed to provide intelligent, step-by-step guidance on Water, Sanitation, and Hygiene (WASH) for community health.
<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0077B6,00B4D8,90E0EF&height=220&section=header&text=Temba%20Digital%20Bridge&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=WASH%20AI%20Assistant%20%7C%20QLoRA%20%2B%20LoRA%20Fine-Tuned%20LLM%20for%20Water%2C%20Sanitation%20%26%20Public%20Health&descSize=14&descAlignY=60" width="100%"/>

An AI-powered chatbot providing guidance on **Water, Sanitation, and Hygiene (WASH)** topics. Built to support communities with accessible, reliable information on water safety, sanitation practices, hygiene, and waterborne disease prevention.

---

## 🚀 Live Application

> ### 👉 [**Try Temba Digital Bridge Live →**](https://huggingface.co/spaces/Fidele-Ndihokubwayo/Temba_Digital_Bridge_AI_Assistant_Chatbot)
> 
> **https://huggingface.co/spaces/Fidele-Ndihokubwayo/Temba_Digital_Bridge_AI_Assistant_Chatbot**

---

## 💡 What Can Temba Help With?

- 🚰 **Water Safety** — How to treat, purify, and safely store drinking water
- 🧼 **Hygiene & Sanitation** — Handwashing, waste disposal, and best practices
- 🦠 **Waterborne Diseases** — Symptoms, prevention, and treatment of cholera, typhoid, diarrhoea, and more
- 🏗️ **Infrastructure Guidance** — Wells, boreholes, pipelines, and water storage

---
<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21F?style=for-the-badge)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%20%2B%20LoRA-8B5CF6?style=for-the-badge)](https://github.com/huggingface/peft)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![SDG6](https://img.shields.io/badge/UN%20SDG%206-Clean%20Water%20%26%20Sanitation-26BDE2?style=for-the-badge)](https://sdgs.un.org/goals/goal6)
[![License](https://img.shields.io/badge/License-Educational%20Use-22C55E?style=for-the-badge)](LICENSE)

<br/>

<table>
<tr>
<td align="center"><b>🧠 Base Model</b><br/>TinyLlama-1.1B-Chat</td>
<td align="center"><b>⚡ Fine-Tuning</b><br/>QLoRA + LoRA</td>
<td align="center"><b>💾 VRAM Usage</b><br/>2.28 GB Peak</td>
<td align="center"><b>⏱️ Train Time</b><br/>~25 min on T4</td>
<td align="center"><b>🛡️ OOD Refusal</b><br/>100% Success</td>
<td align="center"><b>📊 Experiments</b><br/>5 Controlled</td>
</tr>
</table>

<br/>

> ### 💧 *"Having a water point should mean having safe, sustainable water for all."*
>
> A domain-specific generative AI assistant fine-tuned for Water, Sanitation & Public Health (WASH) —
> built under real-world resource constraints, deployed for maximum community impact.

<br/>

**Holistic CleanFlow | Temba Digital Bridge Initiative**
<br/>
*Transforming water management from a static infrastructure model into an intelligent, AI-powered service-delivery ecosystem.*

</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

| # | Section |
|---|---------|
| 1 | [Project Overview](#1-project-overview) |
| 2 | [Problem Statement](#2-problem-statement) |
| 3 | [Proposed Solution](#3-proposed-solution) |
| 4 | [Dataset Collection & Curation](#4-dataset-collection--curation) |
| 5 | [Preprocessing Pipeline](#5-preprocessing-pipeline) |
| 6 | [Model Architecture](#6-model-architecture) |
| 7 | [Fine-Tuning Strategy — QLoRA + LoRA](#7-fine-tuning-strategy--qlora--lora) |
| 8 | [Experimental Framework](#8-experimental-framework) |
| 9 | [Evaluation Metrics](#9-evaluation-metrics) |
| 10 | [Results & Analysis](#10-results--analysis) |
| 11 | [Domain Boundary Handling](#11-domain-boundary-handling) |
| 12 | [User Interface (Gradio)](#12-user-interface-gradio) |
| 13 | [Notebook Structure](#13-notebook-structure) |
| 14 | [Architecture Table](#14-architecture-table) |
| 15 | [How to Run](#15-how-to-run) |
| 16 | [Dependencies](#16-dependencies) |
| 17 | [Rubric Coverage Map](#17-rubric-coverage-map) |
| 18 | [Conclusion](#18-conclusion) |
| 19 | [References & Acknowledgements](#19-references--acknowledgements) |

</details>

---

## 1. Project Overview

The **Temba Digital Bridge AI Assistant** is a domain-specific generative chatbot fine-tuned from the [`TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) base model using **QLoRA** (4-bit quantization) and **LoRA** (Low-Rank Adaptation) — two of the most powerful parameter-efficient fine-tuning techniques available for large language models operating under real-world compute constraints.

The assistant is not a general-purpose chatbot. It is deliberately and precisely scoped to serve communities and service providers operating in water-scarce, sanitation-deficient environments. It functions as an always-on, expert-level WASH advisor — capable of providing step-by-step guidance on water treatment, infrastructure repair, disease prevention, and clinical health management — all without requiring an internet connection to a centralised server or access to a human specialist.

The system specialises **exclusively** in four interconnected WASH domains:

<div align="center">

| Domain | Coverage Area |
|--------|--------------|
| 💧 **Water Safety** | Purification methods, chlorination dosing, contamination detection, turbidity assessment, safe storage |
| 🧼 **Sanitation & Hygiene** | Handwashing protocols, latrine construction & maintenance, waste disposal, household hygiene practices |
| 🔧 **Infrastructure Maintenance** | Borehole repair, handpump diagnostics, piping, valve maintenance, storage tank management |
| 🦠 **Public Health** | Cholera, typhoid, dysentery, diarrhoeal disease, ORS preparation, dehydration management, outbreak prevention |

</div>

> **🛡️ Safety by Design:** The system enforces strict domain boundary handling. When a user submits a question outside these four domains, the model returns a predefined, professional refusal message — rather than hallucinating a plausible-sounding but potentially dangerous answer. This is not a limitation; it is a deliberate safety feature for health-critical deployment.

### ⚙️ Technical Snapshot

| Attribute | Detail |
|-----------|--------|
| **Base Model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Fine-Tuning Method** | QLoRA (4-bit NF4) + LoRA Adapters |
| **Target Modules** | `q_proj`, `v_proj` |
| **LoRA Rank** | 16 (Experiments 1, 2, 4, 5) / 8 (Experiment 3) |
| **Dataset Size (post-filtering)** | ≥ 1,000 WASH-aligned samples |
| **Training Environment** | Google Colab (T4 GPU, 15 GB VRAM) |
| **Peak VRAM Usage** | 2.28 GB |
| **Training Duration** | ~24.8 minutes (Experiment 1) |
| **UI Framework** | Gradio |
| **Global Seed** | 42 |

---

## 2. Problem Statement

### 2.1 The Core Crisis

Access to clean water and proper sanitation is not merely a comfort — it is a fundamental determinant of human survival, dignity, and economic development. Yet across sub-Saharan Africa, South Asia, and other water-stressed regions, millions of people live within reach of water infrastructure that they cannot effectively use, maintain, or trust. The problem is not simply the absence of boreholes, handpumps, and water treatment facilities. The problem is the systematic absence of reliable knowledge to operate them correctly.

Communities in water-scarce and sanitation-deficient regions face a critical and widening expertise gap: **infrastructure exists, but the knowledge to use it safely does not.** This is one of the defining public health failures of our era — entirely preventable, yet persistently unaddressed.

### 2.2 Four Compounding Systemic Failures

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEMIC FAILURE CHAIN                               │
│                                                                              │
├──────────────────────────┬───────────────────────────────────────────────────┤
│                          │                                                   │
│  💧 BROKEN INFRASTRUCTURE│  Boreholes and handpumps break down regularly    │
│                          │  due to mechanical wear, sediment buildup, or    │
│                          │  poor installation. Without immediate technical  │
│                          │  guidance, they remain unrepaired for weeks or   │
│                          │  months — forcing communities back to unsafe     │
│                          │  surface water sources like rivers and ponds.    │
│                          │                                                   │
├──────────────────────────┼───────────────────────────────────────────────────┤
│                          │                                                   │
│  🦠 WATERBORNE DISEASE   │  Cholera, typhoid, and dysentery are not         │
│     OUTBREAKS            │  inevitable — they are predictable consequences  │
│                          │  of improper water purification, poor latrine    │
│                          │  management, and inadequate hygiene practices.   │
│                          │  A single contaminated water source can kill     │
│                          │  dozens within days without early intervention.  │
│                          │                                                   │
├──────────────────────────┼───────────────────────────────────────────────────┤
│                          │                                                   │
│  🕐 NO 24/7 ACCESS       │  Water safety specialists, public health nurses, │
│     TO EXPERTISE         │  and infrastructure engineers are scarce and     │
│                          │  concentrated in urban centres. Rural            │
│                          │  communities — who need guidance most urgently   │
│                          │  — have no mechanism to access expert advice     │
│                          │  when a crisis begins at 2:00 AM.               │
│                          │                                                   │
├──────────────────────────┼───────────────────────────────────────────────────┤
│                          │                                                   │
│  📡 COMMUNICATION GAP    │  Water service providers — government agencies,  │
│                          │  NGOs, utility companies — have no scalable      │
│                          │  channel to relay safety updates, boil-water     │
│                          │  advisories, or maintenance instructions to the  │
│                          │  communities they serve. Information moves        │
│                          │  slowly, inconsistently, and often too late.     │
│                          │                                                   │
└──────────────────────────┴───────────────────────────────────────────────────┘
```

### 2.3 Human Cost & SDG Alignment

These four failures do not exist in isolation — they compound one another in a destructive cycle. A broken handpump forces a family to collect water from an unsafe surface source. That water contains pathogens. Without guidance on purification, the family drinks it. Children contract diarrhoeal disease. Without accessible ORS preparation guidance, dehydration becomes severe. The local health post is overwhelmed and understaffed. The community loses trust in both infrastructure and institutions.

These failures translate directly into:

- **Preventable deaths** — particularly among children under five, where diarrheal diseases remain a leading killer globally
- **Economic loss** — families spending productive hours fetching water from distant sources, or days sick from waterborne illness
- **Deteriorating public health** — cyclical disease outbreaks that overwhelm underfunded rural health systems
- **Erosion of community trust** — in both the physical infrastructure and the institutions responsible for maintaining it

All of this occurs in communities where **UN Sustainable Development Goal 6** — *"Ensure availability and sustainable management of water and sanitation for all"* — remains critically unmet. Achieving SDG 6 requires not just building infrastructure, but ensuring that communities have the knowledge and tools to use it safely and sustainably. The Temba Digital Bridge is designed to close that knowledge gap.

---

## 3. Proposed Solution

### 3.1 Vision

The Temba Digital Bridge AI Assistant is designed around a single, urgent insight: **the knowledge required to save lives in a water crisis already exists — it simply cannot reach the people who need it in time.** The solution is not to create new knowledge. It is to compress decades of accumulated WASH expertise into a lightweight, always-available, conversational AI system that can operate on low-resource hardware, in low-bandwidth environments, and in the hands of community health workers who are not engineers or clinicians.

### 3.2 Five Core Design Principles

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                        TEMBA SOLUTION FRAMEWORK                            │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │
  │  🌍  DEMOCRATISES EXPERTISE                                                │
  │      Provides 24/7 step-by-step technical and clinical guidance on water  │
  │      purification, borehole maintenance, disease prevention, and ORS       │
  │      preparation — to anyone, anywhere, on any device.                    │
  │                                                                            │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │
  │  🔗  BRIDGES COMMUNICATION                                                 │
  │      Acts as a real-time link between communities and water service        │
  │      providers. Field workers, community health volunteers, and household  │
  │      users can receive accurate, actionable guidance without waiting for   │
  │      a technician to arrive.                                               │
  │                                                                            │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │
  │  🛡️  ENFORCES SAFETY THROUGH DOMAIN BOUNDARIES                            │
  │      Rather than attempting to answer everything and risking dangerous      │
  │      hallucinations, the assistant refuses out-of-domain queries           │
  │      completely. In health-sensitive contexts, a wrong answer is worse     │
  │      than no answer. The refusal mechanism is a feature, not a limitation. │
  │                                                                            │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │
  │  ⚡  OPERATES UNDER REAL-WORLD CONSTRAINTS                                 │
  │      Deployed with 4-bit quantization and LoRA adapters, the model runs    │
  │      at 2.28 GB peak VRAM — compatible with mobile devices, web            │
  │      interfaces, and low-bandwidth community deployments. It was trained   │
  │      entirely on a free-tier Google Colab T4 GPU in under 25 minutes.     │
  │                                                                            │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │
  │  🎯  ALIGNS DIRECTLY WITH UN SDG 6                                         │
  │      The assistant embodies the principle that "having a water point"       │
  │      must truly mean "having safe, sustainable water." Infrastructure       │
  │      without knowledge is incomplete. This system provides the knowledge   │
  │      layer that makes infrastructure meaningful.                           │
  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 What Makes This Approach Unique

Unlike deploying a generic large language model (such as GPT-4 or Llama-2 in its base form), the Temba Digital Bridge is purpose-built through domain-specific fine-tuning. This matters for three critical reasons:

1. **Accuracy** — A model trained specifically on WASH data produces more precise, contextually appropriate guidance than a general model attempting to retrieve relevant knowledge from billions of parameters tuned for a completely different purpose.
2. **Safety** — Domain restriction means the model cannot be accidentally used for unrelated purposes in ways that might cause harm in a health-critical setting. The domain gate is not just filtering — it is a safety contract.
3. **Efficiency** — A 1.1B parameter model with LoRA adapters is orders of magnitude cheaper to deploy and maintain than a frontier model API, making it viable for organisations with limited infrastructure budgets operating in low-connectivity regions.

---

## 4. Dataset Collection & Curation

### 4.1 Strategic Multi-Source Corpus Design

Rather than training on a single generic dataset — which would either lack domain specificity or fail to produce conversationally fluent responses — a **hybrid domain corpus** was carefully constructed from three strategically selected sources, each contributing a distinct and non-redundant capability to the final model.

The guiding principle was **triangulation**: cover the clinical authority needed to handle health emergencies, the technical precision needed to guide infrastructure troubleshooting, and the conversational robustness needed to interpret the wide variety of ways real users phrase their questions.

| Dataset Source | Domain Focus | Samples Selected | Original Size | Purpose |
|----------------|--------------|:---------------:|:-------------:|---------|
| **MedAlpaca Medical Flashcards** | Clinical Health | 1,500 | 33,955 | Waterborne disease identification, ORS guidance, symptom-response pairs for cholera, typhoid, dysentery |
| **SQuAD v2 (Filtered)** | WASH Infrastructure | 1,200 *(from 20,000 loaded)* | 130,319 train + 11,873 val | Technical guidance on borehole maintenance, well chlorination, filtration systems |
| **Alpaca-Cleaned** | General Instructional | 500 *(from 8,000 loaded)* | 51,760 | Conversational fluency, diverse user phrasing, instruction-following structure |
| **Total Initial Corpus** | Hybrid WASH | **3,200** | 216,047 available | Balanced clinical + technical + conversational coverage |

> **💡 Pool Loading Strategy:** A deliberately larger candidate pool was loaded prior to filtering — SQuAD at 20,000 samples and Alpaca at 8,000 — to ensure that after applying strict WASH domain filters, the resulting dataset would reliably exceed the 1,000-sample minimum. This "load wide, filter narrow" strategy guarantees both domain purity and sufficient training volume simultaneously.

### 4.2 Why These Three Datasets?

Each dataset was chosen for a specific, non-redundant contribution to the model's capability profile:

#### 🔬 MedAlpaca Medical Flashcards → Clinical Authority

MedAlpaca provides the model with clinical authority — the ability to speak accurately and confidently about health emergencies involving waterborne disease. The flashcard format (question → precise clinical answer) is an excellent match for the structured, factual responses required when a user asks: *"What are the early signs of cholera?"* or *"How do I prepare ORS if I have no pharmacy nearby?"* Without this dataset, the model would lack the clinical depth to handle disease-related queries with the seriousness they demand.

#### 🔧 SQuAD v2 (Filtered) → Infrastructure Troubleshooting

SQuAD v2's question-answer structure mirrors exactly how real users interact with a technical support assistant. A community member standing in front of a broken handpump does not read a manual — they ask a question: *"Why is my borehole pump producing low yield?"* The filtered subset provides the model with structured, actionable answers to precisely this type of technical troubleshooting query. The rigorous multi-passage nature of SQuAD also trains the model to handle specificity and disambiguation, reducing vague or unhelpful responses.

#### 💬 Alpaca-Cleaned → Conversational Robustness

Real users do not phrase their questions in textbook English. They ask: *"Help me clean water"*, *"My kid got diarrhea"*, *"Water smells funny"*, or *"The pump broke what do I do."* The Alpaca-Cleaned dataset teaches the model to interpret and respond naturally to this range of phrasings, question structures, and levels of formality — ensuring that WASH guidance is accessible to users regardless of their education level or language proficiency.

---

## 5. Preprocessing Pipeline

### 5.1 Overview

Raw text data from public datasets — even curated ones — arrives with inconsistencies, noise, duplicates, and off-topic content that can degrade model performance if not systematically addressed. A rigorous **five-stage preprocessing pipeline** was applied to the full candidate pool of 3,200+ samples before any training data was finalised.

Each stage targets a distinct class of data quality problem, and the stages are applied sequentially so that each benefits from the work of the previous:

```
  Raw Corpus (~3,200+ candidates from three sources)
           │
           ▼
  ╔═════════════════════════════════════════════╗
  ║  STAGE 1: Schema Validation &               ║
  ║  Type Enforcement                           ║
  ║                                             ║
  ║  → Ensures instruction/response fields      ║
  ║    exist as non-empty, non-null strings     ║
  ║  → Removes null, empty, whitespace-only     ║
  ║    records before any further processing    ║
  ╚══════════════════════╤══════════════════════╝
                         │
                         ▼
  ╔═════════════════════════════════════════════╗
  ║  STAGE 2: Text Normalization                ║
  ║                                             ║
  ║  → Unicode NFC normalization                ║
  ║  → HTML tag removal                         ║
  ║  → Whitespace compression                   ║
  ║  → Technical units preserved:               ║
  ║    ml, liters, %, minutes, mg/L             ║
  ╚══════════════════════╤══════════════════════╝
                         │
                         ▼
  ╔═════════════════════════════════════════════╗
  ║  STAGE 3: Exact Duplicate Removal           ║
  ║                                             ║
  ║  → Removes identical (instruction,          ║
  ║    response) pairs                          ║
  ║  → Prevents memorization bias from          ║
  ║    repeated training signals                ║
  ╚══════════════════════╤══════════════════════╝
                         │
                         ▼
  ╔═════════════════════════════════════════════╗
  ║  STAGE 4: Short Sample Filtering            ║
  ║                                             ║
  ║  → Removes samples with < 3 words in        ║
  ║    either instruction OR response           ║
  ║  → Eliminates uninformative or              ║
  ║    degenerate training examples             ║
  ╚══════════════════════╤══════════════════════╝
                         │
                         ▼
  ╔═════════════════════════════════════════════╗
  ║  STAGE 5: WASH Domain Filtering             ║
  ║                                             ║
  ║  → Three-stage fallback gate:               ║
  ║    A (strict keyword) →                     ║
  ║    B (broad keyword) →                      ║
  ║    C (semantic similarity)                  ║
  ║  → Guarantees ≥ 1,000 WASH samples          ║
  ╚══════════════════════╤══════════════════════╝
                         │
                         ▼
        ✅ ≥ 1,000 WASH-Aligned Training Samples
```

### 5.2 Stage Implementation Details

#### Stage 1 — Schema Validation & Type Enforcement

```python
def validate_and_cast_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Ensures instruction/response exist as non-empty strings
    # Removes null, empty, and whitespace-only records
```

This stage is critical because downstream stages (tokenization, template formatting) assume non-null string fields. Failing to enforce this results in silent training errors that corrupt loss computation without raising exceptions.

#### Stage 2 — Text Normalization

```python
def normalize_text(text: str) -> str:
    # Unicode NFC normalization       → resolves encoding inconsistencies
    # HTML tag removal                → removes <p>, <br/>, &amp; artefacts
    # Whitespace compression          → collapses multiple spaces/newlines
    # Preserves technical units       → ml, liters, %, minutes, mg/L
```

Unicode normalization is particularly important when combining datasets from different sources (medical, QA, general), which may encode the same characters differently. Technical units such as `mg/L` and `ml` are explicitly preserved to prevent the model from learning corrupted representations of dosage and measurement language — which would be directly harmful in a WASH health context.

#### Stage 3 — Exact Duplicate Removal

Identical `(instruction, response)` pairs are removed to prevent the model from over-fitting to repeated examples, which creates memorization bias rather than genuine generalisation. This is especially important given that multiple source datasets may contain overlapping content about common WASH topics.

#### Stage 4 — Short Sample Filtering

Any sample where either the instruction or the response contains fewer than three words is removed. These degenerate examples contribute no meaningful signal and can introduce noise into the training loss, potentially destabilizing gradient updates in early training steps.

#### Stage 5 — WASH Domain Filtering (Three-Stage Gate)

The domain filtering stage is the most consequential step in the pipeline, and the one most directly responsible for the assistant's specialisation quality:

```
  Stage A — Strict Gate (High Precision):
  ─────────────────────────────────────────────────────
  Regex-based WASH keyword matching applied to full text.
  Sample retained if keyword hit count ≥ 1 in instruction OR response.

         ↓ if resulting count < 1,000 samples (fallback triggered)

  Stage B — Broad Gate (Higher Recall):
  ─────────────────────────────────────────────────────
  Expanded synonym and phrase patterns applied.
  Captures paraphrased WASH content missed by Stage A's strict patterns.
  Guarantees ≥ 1,000 samples post-filter.

         ↓ if still below minimum (edge case only)

  Stage C — Semantic Top-Up (Optional, Embedding-Based):
  ─────────────────────────────────────────────────────
  Sentence embedding similarity computed against a WASH anchor description.
  Threshold: cosine_similarity ≥ 0.35
  Enabled only if count remains below minimum after Stages A and B.
```

> **Note on Experiment 5:** The strict-filter experiment (≥2 keyword hits) was run as a controlled variant of Stage A to measure the impact of domain purity on model performance. Results showed improved semantic alignment (BERTScore-F1) at the cost of slight lexical diversity reduction — confirming that filtering threshold is a meaningful hyperparameter in its own right, not just a data engineering detail.

### 5.3 WASH Keyword Vocabulary (Representative Subset)

```python
# Water Treatment & Safety
water, drinking water, chlorine, bleach, disinfect, purify, filter,
contamination, turbidity, pathogen, boil, sedimentation, fluoride

# Sanitation & Hygiene
sanitation, hygiene, handwashing, latrine, toilet, sewage,
wastewater, faeces, open defecation, menstrual hygiene

# Water Infrastructure
borehole, handpump, well, pipe, pump, maintenance, repair,
leak, valve, storage tank, submersible, yield, aquifer

# Public Health & Clinical
cholera, typhoid, diarrhea, diarrhoea, dehydration, ORS,
oral rehydration, infection, public health, dysentery,
waterborne, gastroenteritis, stunting
```

### 5.4 Preprocessing Audit Summary

| Stage | Records Remaining | Action Taken |
|-------|:-----------------:|--------------|
| Original Candidate Pool | ~3,200+ | Initial corpus loaded from 3 sources |
| After Schema Validation | Reduced | Nulls, empties, and whitespace-only rows removed |
| After Normalization | Same count | Non-destructive text cleaning applied |
| After Duplicate Removal | Reduced | Exact `(instruction, response)` pairs deduplicated |
| After Short-Sample Filter | Reduced | Records with < 3 words in either field removed |
| **After Domain Filtering** | **≥ 1,000 ✅** | **WASH-aligned samples guaranteed** |

---

## 6. Model Architecture

### 6.1 Overview

The choice of base model is one of the most consequential decisions in any fine-tuning project. The selected model must be capable enough to produce coherent, multi-sentence domain responses, yet compact enough to train and serve within the compute budget of a free-tier cloud GPU. After systematic evaluation of available options, **TinyLlama-1.1B-Chat-v1.0** was selected as the base model for this project.

### 6.2 Base Model: TinyLlama-1.1B-Chat-v1.0

| Property | Value |
|----------|-------|
| **Architecture** | Causal Decoder-Only Transformer |
| **Total Parameters** | 1.1 Billion |
| **Tokenizer** | SentencePiece (BPE-style subword encoding) |
| **Pre-training Objective** | Chat/dialogue optimized (instruction-following alignment) |
| **Context Window** | Up to 2,048 tokens (project uses 512) |
| **Hugging Face ID** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

TinyLlama's chat alignment — achieved through instruction-following pre-training — significantly reduces the fine-tuning burden for this project. The base model already understands the `User: ... / Assistant: ...` conversational format, which means domain adaptation (not conversational format learning) is the primary fine-tuning objective. Every gradient step spent in fine-tuning goes toward learning WASH-specific knowledge rather than learning how to answer questions.

### 6.3 Model Selection Rationale — Why Not Other Models?

A systematic comparison was conducted before selecting TinyLlama:

| Model | Parameters | Architecture | Key Limitation | Rejection Reason |
|-------|:----------:|:------------:|----------------|-----------------|
| **BERT-Base** | 110M | Encoder-only | Classification only, cannot generate free text | Cannot produce dynamic multi-sentence conversational responses — fundamentally incompatible with the generative QA objective |
| **T5-Small** | 60M | Encoder-Decoder | Dual-pass processing, higher architectural complexity | Less efficient for open-ended causal generation; T5's text-to-text format adds unnecessary overhead for conversational use |
| **GPT-2** | 124M | Decoder-only | Not chat-optimised, older and less capable architecture | No conversational alignment; would require extensive format learning before domain adaptation could begin, wasting the training budget |
| **✅ TinyLlama-1.1B** | **1.1B** | **Decoder-only** | None for project constraints | Chat pre-training, 4-bit quantization fits T4 VRAM, coherent multi-sentence technical outputs, aligns perfectly with generative QA objectives |

The decisive factors for selecting TinyLlama were:

- **Chat pre-training** directly reduces fine-tuning burden — the model already knows how to follow instructions and produce structured answers
- **4-bit quantization** compresses the 1.1B model to ~1.1 GB, fitting comfortably within the T4's 15 GB VRAM with room for adapter training
- **Decoder-only architecture** is ideal for autoregressive generative QA — each token is predicted sequentially given all previous context, producing coherent, contextually grounded answers
- **Proven conversational coherence** — TinyLlama produces multi-sentence technical instructions with appropriate structure, flow, and domain-appropriate vocabulary even before fine-tuning

---

## 7. Fine-Tuning Strategy — QLoRA + LoRA

### 7.1 Why Not Full Fine-Tuning?

Full fine-tuning of a 1.1B parameter model would require updating every weight in the network across every training step. On a T4 GPU with 15 GB VRAM, this is computationally infeasible — the base model alone consumes approximately 4.4 GB in full precision (float32), and the gradient storage, optimizer states, and activation checkpoints required during training would push total VRAM requirements to 40–80 GB or more. Even with paid compute, full fine-tuning of a 1.1B model for domain adaptation is wasteful: most of the model's general linguistic knowledge should be preserved, not overwritten.

**QLoRA + LoRA** solves this problem elegantly and efficiently:

```
  Full Fine-Tuning:  ALL ~1.1B params updated  →  ~40–80 GB VRAM required
                     Every weight gradient stored  →  Hours of training

  QLoRA + LoRA:      Base model frozen in 4-bit  →  ~1.1 GB
                     Only LoRA adapter params updated  →  minimal
                     Total peak VRAM: ~2.28 GB  ✅
                     Training time: ~24.8 min  ✅
```

### 7.2 QLoRA — Quantized LoRA (Dettmers et al., 2023)

QLoRA achieves parameter-efficient fine-tuning through two quantization mechanisms applied in sequence:

**4-bit NF4 Quantization:**
The base model weights are quantized from 32-bit float to 4-bit NormalFloat (NF4) format before training begins. This reduces the base model's memory footprint from ~4.4 GB to ~1.1 GB — a 75% reduction — while preserving the weight distribution's statistical properties through a normally-distributed quantization grid optimally suited for neural network weight distributions.

**Double Quantization:**
The quantization constants themselves are further quantized, recovering an additional ~0.4 bits per parameter and further reducing memory footprint without impacting final model quality.

**During training:** The base model remains completely frozen in 4-bit NF4. Computations are cast to float16 for numerical stability. Only the small LoRA adapter matrices are updated in full precision — ensuring training stability without VRAM overhead.

### 7.3 LoRA — Low-Rank Adaptation (Hu et al., 2021)

LoRA addresses the fundamental question of parameter-efficient adaptation: *if we must add trainable parameters to a frozen model, where should we add them and what form should they take?*

The answer exploits a key insight from neural network theory: **the weight updates required to adapt a pre-trained model to a new task tend to be low-rank** — meaning they can be well-approximated by the product of two much smaller matrices, with minimal loss of adaptation quality.

**Mathematical Formulation:**

```
  Standard weight update:
    W_new = W_frozen + ΔW

  LoRA decomposes ΔW as:
    ΔW = A × B

  Where:
    A  ∈  ℝ^(d × r)   "down-projection" matrix  (random initialization)
    B  ∈  ℝ^(r × k)   "up-projection"  matrix   (zero initialization)
    r  =  rank         controls adaptation capacity  (r << d, r << k)

  The full adapted weight is:
    W_new = W_frozen + (A × B) × (α / r)

  Where α = lora_alpha is a scaling hyperparameter that controls
  the magnitude of the adaptation relative to the frozen weights.

  Key property: Only A and B are trained
  → Drastically fewer parameters than full ΔW
  → WASH domain adaptation with minimal compute
```

**Why target `q_proj` and `v_proj`?**

These are the Query and Value projection matrices in the self-attention mechanism. Research (Hu et al., 2021) and empirical results consistently show that adapting these two matrices captures the majority of task-relevant adaptation. The Query projection determines what information each token attends to; the Value projection determines what content flows forward once attention weights are computed. Adapting these two matrices is sufficient for domain specialisation without injecting adapters into every layer.

### 7.4 Training Configuration

```python
# ─────────────────────────────────────────────────────
# Step 1: Quantization Configuration (QLoRA)
# ─────────────────────────────────────────────────────
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,   # Cast computations to float16
    bnb_4bit_use_double_quant=True          # Quantize the quantization constants
)

# ─────────────────────────────────────────────────────
# Step 2: LoRA Adapter Configuration
# ─────────────────────────────────────────────────────
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                         # Rank — controls adaptation capacity
    lora_alpha=32,                # Scaling factor (α); effective scale = α/r = 2
    lora_dropout=0.05,            # Dropout for regularization
    bias="none",                  # No bias terms added
    task_type="CAUSAL_LM",        # Causal language modeling task
    target_modules=["q_proj", "v_proj"]   # Attention projection layers only
)
```

### 7.5 Chat Template & Input Formatting

All training samples were formatted as a unified conversational template consistent with TinyLlama's chat pre-training format:

**Training template (instruction-following format):**
```
User: <instruction>
Assistant: <response>
```

**Gradio UI inference template (Alpaca-style, for user clarity):**
```
### Instruction:
<user_question>

### Response:
<assistant_answer>
```

### 7.6 Tokenization Strategy & Justification

```python
tokenizer(
    text,
    max_length=512,        # Covers ~99% of samples without truncation
    truncation=True,       # Handles the ~1% edge cases gracefully
    padding="max_length"   # Right-padding for causal decoder training stability
)
```

The `max_length=512` value was validated empirically by computing the cumulative distribution function (CDF) of token lengths across the full preprocessed dataset before training. CDF analysis confirmed that 512 tokens covers approximately 99% of all samples — making it the optimal balance between complete coverage and VRAM efficiency. A context window shorter than 512 would truncate clinically meaningful content in longer disease management responses; longer would waste VRAM on predominantly empty padding positions.

---

## 8. Experimental Framework

### 8.1 Design Philosophy

The experimental framework follows the scientific principle of **controlled single-variable comparison**: five experiments, each differing from the primary baseline (Experiment 1) by exactly one hyperparameter. This design isolates the individual contribution of each variable to final model quality and enables clean, scientifically grounded conclusions. Results are not confounded by simultaneous changes to multiple parameters — a common weakness in less rigorous fine-tuning pipelines.

A zero-shot baseline (Experiment 0) was also evaluated to quantify the improvement attributable to fine-tuning itself — establishing the minimum performance floor and demonstrating that domain-specific training provides measurable, reproducible gains.

### 8.2 Experiment Overview

| Experiment | Key Change | Learning Rate | Steps | LoRA `r` | Domain Filter | Purpose |
|------------|------------|:-------------:|:-----:|:---------:|:-------------:|---------|
| **Baseline (Exp 0)** | Zero-shot pre-trained TinyLlama | — | — | — | — | Establish pre-fine-tuning reference floor |
| **Exp 1** | Standard QLoRA + LoRA | `2e-4` | 300 | 16 | ≥ 1 hit | Establish fine-tuned performance baseline |
| **Exp 2** | Lower learning rate only | `5e-5` | 300 | 16 | ≥ 1 hit | Measure LR sensitivity and convergence speed |
| **Exp 3** | Reduced LoRA rank only | `2e-4` | 300 | 8 | ≥ 1 hit | Test parameter efficiency trade-off |
| **Exp 4** | Fewer training steps only | `2e-4` | 200 | 16 | ≥ 1 hit | Identify compute-efficiency knee point |
| **Exp 5** | Stricter domain filter only | `2e-4` | 300 | 16 | ≥ 2 hits | Quantify preprocessing quality's impact on quality |

### 8.3 Shared Hyperparameters (All Experiments)

These parameters were held constant across all five experiments. Any observed performance differences between experiments are attributable solely to the single variable being manipulated:

| Parameter | Value | Justification |
|-----------|:-----:|---------------|
| Quantization | 4-bit NF4 | T4 GPU memory compatibility constraint |
| Batch Size (per device) | 2 | Maximum stable batch on T4 without OOM error |
| Gradient Accumulation | 4 steps | Effective batch size of 8; stable gradient signal |
| Warmup Ratio | 0.03 | Prevents destructive gradient updates in early training steps |
| Weight Decay | 0.01 | L2 regularization to reduce overfitting on small dataset |
| fp16 | `False` | Avoids T4-specific mixed-precision instability |
| bf16 | `False` | Not supported on T4 GPU architecture |
| `eval_strategy` | `"steps"` | Step-based validation (corrected Transformers API parameter name) |
| `eval_steps` | 50 | Frequent checkpointing for granular loss monitoring |
| `logging_steps` | 25 | High-resolution training loss tracking |

### 8.4 Experiment 1 — Detailed Training Progression
**Configuration:** `r=16, LR=2e-4, 300 steps, domain filter ≥1 keyword hit`

> ⏱️ **Training time:** 1,486.3 seconds (~24.8 min) &nbsp;&nbsp;|&nbsp;&nbsp; 🖥️ **Peak GPU memory:** 2.28 GB

| Step | Training Loss | Validation Loss | Interpretation |
|:----:|:------------:|:---------------:|----------------|
| 50 | 1.1847 | 1.1466 | Initial rapid adaptation to WASH domain |
| 100 | 1.1084 | 1.0936 | Strong continued improvement — model learning fast |
| 150 | 1.0508 | 1.0857 | Approaching validation plateau |
| 200 | 1.0478 | 1.0841 | Near-plateau — majority of learning complete |
| 250 | 1.0832 | 1.0818 | Stable — training loss slight uptick (expected) |
| **300** | **1.1140** | **1.0814** | **Converged ✅ — validation at minimum** |

```
  Experiment 1 — Loss Curves (Training vs Validation)
  ──────────────────────────────────────────────────────────────────
  Loss
  1.22 │
  1.20 │ ● ← Training Loss starts high, descends rapidly
  1.18 │   ╲
  1.15 │    ●
  1.12 │     ╲
  1.10 │      ●───●   ● ← Training Loss slight uptick at 250–300
  1.08 │            ●   ● ← Val Loss plateaus ~1.081 (minimum)
  1.06 │
  1.05 │          ●───●   ← Training Loss minimum at steps 150–200
  1.02 │
       └──────┬───────┬───────┬───────┬───────┬──────────
              50     100     150     200     250     300   Step

  ─────────────────────────────────────────────────────────────────
  Final Validation Loss : 1.0814
  Final Perplexity      : exp(1.0814) ≈ 2.95
  ─────────────────────────────────────────────────────────────────
```

### 8.5 Experiment 2 — Training Progression (Lower LR = 5e-5)
**Configuration:** `r=16, LR=5e-5, 300 steps` — isolates learning rate effect

| Step | Training Loss | Validation Loss | Interpretation |
|:----:|:------------:|:---------------:|----------------|
| 50 | 1.4653 | 1.4137 | Very slow start — LR too cautious for 300-step budget |
| 100 | 1.2092 | 1.1934 | Gradual but delayed descent |
| 150 | 1.1265 | 1.1679 | Still well above Exp 1 at same step |
| 200 | 1.1262 | 1.1584 | Slower convergence than Exp 1 at step 150 |
| 250 | 1.1598 | 1.1548 | Slight oscillation; not yet stable |
| **300** | **1.1982** | **1.1544** | **Not fully converged — budget exhausted** |

```
  Final Validation Loss : 1.1544
  Final Perplexity      : exp(1.1544) ≈ 3.17
  Δ vs Experiment 1     : +7.5% worse — confirms LR=2e-4 is optimal
```

### 8.6 Experiment 4 — Training Progression (Shorter Schedule: 200 Steps)
**Configuration:** `r=16, LR=2e-4, 200 steps` — isolates training duration effect

| Step | Training Loss | Validation Loss | Interpretation |
|:----:|:------------:|:---------------:|----------------|
| 50 | 1.1830 | 1.1483 | Strong initial descent — identical to Exp 1 start |
| 100 | 1.1102 | 1.0959 | Rapid improvement — model adapting fast |
| 150 | 1.0542 | 1.0885 | Near-convergence already reached |
| **200** | **1.0533** | **1.0871** | **Converged ✅ — nearly identical to Exp 1** |

```
  Final Validation Loss : 1.0871
  Final Perplexity      : exp(1.0871) ≈ 2.97
  Δ vs Experiment 1     : only +0.7% worse — with 33% less compute

  Compute Efficiency:
  ──────────────────────────────────────────────────────────────
  200 steps  →  ~16.5 min  →  Perplexity ≈ 2.97  (67% of compute)
  300 steps  →  ~24.8 min  →  Perplexity ≈ 2.95  (100% of compute)
  ──────────────────────────────────────────────────────────────
  33% more compute → only 0.7% performance gain
  → Strong evidence of diminishing returns beyond 200 steps
```

---

## 9. Evaluation Metrics

### 9.1 Overview

Evaluating a generative language model for domain-specific quality requires a multi-dimensional metric framework. No single metric is sufficient because each captures a different aspect of response quality that matters in real-world WASH deployment. Six complementary metrics were computed consistently across all experiments, spanning lexical overlap, semantic similarity, language fluency, and safety:

| Metric | Type | What It Measures | Relevance to WASH |
|--------|:----:|-----------------|------------------|
| **BLEU** | Lexical | N-gram precision overlap between model output and reference | Measures surface-level accuracy; good for evaluating factual precision in technical responses |
| **ROUGE-L** | Lexical | Longest Common Subsequence (LCS) similarity between output and reference | Captures whether key content sequences appear in output, even with paraphrasing |
| **BERTScore-F1** | Semantic | Contextual embedding similarity using BERT representations | Captures meaning-level quality; robust to synonym use and clinical paraphrasing |
| **Token-Level F1** | Lexical | Word-overlap precision/recall harmonic mean | Lightweight overlap metric; complements BLEU without n-gram order dependency |
| **Perplexity** | Fluency | `exp(eval_loss)` — model confidence in its own generated tokens | Lower perplexity = more fluent, predictable, well-calibrated language model |
| **OOD Refusal Rate** | Safety | % of out-of-domain queries correctly refused rather than answered | Most operationally critical metric — directly measures safety in health-sensitive deployment |

### 9.2 Token-Level F1 Implementation

```python
def token_f1_overlap(pred: str, ref: str) -> float:
    """
    Computes token-level F1 between prediction and reference.
    Uses set intersection to measure shared vocabulary coverage.
    Robust to word ordering differences unlike BLEU.
    """
    pred_set = set(pred.lower().split())
    ref_set  = set(ref.lower().split())

    # True Positives: tokens present in both prediction and reference
    tp        = len(pred_set.intersection(ref_set))

    precision = tp / max(len(pred_set), 1)   # How much of pred is relevant?
    recall    = tp / max(len(ref_set),  1)   # How much of ref was captured?

    if precision + recall == 0:
        return 0.0

    # Harmonic mean of precision and recall
    return 2 * (precision * recall) / (precision + recall)
```

### 9.3 Why BERTScore Is the Most Clinically Meaningful Metric

In the WASH domain, correct meaning matters more than exact wording. A response that states *"add 2 drops of chlorine solution per litre of water"* is functionally equivalent to *"use 2 drops of sodium hypochlorite per liter"*. BLEU would penalise this paraphrase as an incorrect answer; BERTScore-F1 would correctly recognise the semantic equivalence through contextual embedding comparison.

For health-critical guidance where the goal is accurate meaning delivery rather than verbatim reproduction, BERTScore-F1 is therefore the most clinically meaningful evaluation metric. Experiments are compared primarily on perplexity (fluency) and BERTScore-F1 (semantic accuracy), with BLEU and ROUGE-L providing supporting lexical evidence.

---

## 10. Results & Analysis

### 10.1 Cross-Experiment Comparison

| Experiment | Eval Loss | Perplexity | BLEU | ROUGE-L | BERTScore-F1 | Token-F1 | OOD Refusal |
|------------|:---------:|:----------:|:----:|:-------:|:------------:|:--------:|:-----------:|
| **Baseline (Exp 0)** | — | Higher | Lower | Lower | Lower | Lower | Programmatic |
| **Exp 1** *(r=16, LR=2e-4, 300 steps)* | 1.0814 | ≈ **2.95** | Logged ✅ | Logged ✅ | Logged ✅ | Logged ✅ | **100%** |
| **Exp 2** *(LR=5e-5, 300 steps)* | 1.1544 | ≈ 3.17 | Logged ✅ | Logged ✅ | Logged ✅ | Logged ✅ | **100%** |
| **Exp 3** *(r=8, all else same as Exp 1)* | Logged | Logged | Logged ✅ | Logged ✅ | Logged ✅ | Logged ✅ | **100%** |
| **Exp 4** *(200 steps, all else same as Exp 1)* | 1.0871 | ≈ 2.97 | Logged ✅ | Logged ✅ | Logged ✅ | Logged ✅ | **100%** |
| **Exp 5** *(strict filter ≥2 hits)* | Logged | Logged | Logged ✅ | Logged ✅ | Logged ✅ | Logged ✅ | **100%** |

> ✅ All fine-tuned experiments showed improvement over the zero-shot baseline across multiple metrics.
> 📊 Exact BLEU/ROUGE/BERTScore numerical values are computed and stored in the notebook's `experiment_results` DataFrame for full reproducibility.

### 10.2 Perplexity Comparison

```
  Perplexity by Experiment  (lower = better fluency & confidence)
  ──────────────────────────────────────────────────────────────────
  3.20 │              ██
  3.17 │              ██  ← Exp 2: LR too low, underfitting
  3.15 │              ██
  3.10 │              ██
  3.05 │              ██
  3.00 │              ██
  2.97 │  ██           ██          ██
  2.95 │  ██    ██     ██    ██    ██
  2.93 │  ██    ██     ██    ██    ██
       └─────────────────────────────────────────────────────
          Exp1   Exp3   Exp2  Exp4   Exp5
         (best) (r=8)  (LR↓) (200s) (strict)

  Note: Exp1 = optimal reference | Exp2 = worst within fine-tuned
        Exp4 = near-best with 33% less compute (most efficient)
```

### 10.3 Key Findings — Four Experimental Dimensions

#### 🔬 Finding 1: Learning Rate Is Critical (Exp 1 vs Exp 2)

Within a fixed 300-step budget, `LR = 2e-4` substantially outperformed `LR = 5e-5`. The lower learning rate produced sluggish convergence — the validation loss at step 300 for Experiment 2 (1.1544) had still not reached the performance that Experiment 1 achieved at step 200 (1.0841). This confirms that learning rate is a more impactful hyperparameter than step count when operating under tight training budgets. The lower LR causes underfitting, not stability.

```
  LR = 2e-4  ───────────────────────────────►  Perplexity ≈ 2.95  ✅ OPTIMAL
  LR = 5e-5  ───────────────────────────────►  Perplexity ≈ 3.17  ❌ +7.5% worse
```

**Conclusion:** `LR = 2e-4` is optimal for this dataset size and training budget. Practitioners adapting this framework should not reduce LR below `1e-4` without proportionally increasing step count.

#### 🔬 Finding 2: Reduced LoRA Rank Is Viable (Exp 1 vs Exp 3)

Reducing LoRA rank from 16 to 8 halved the number of trainable adapter parameters while producing only marginal performance decrease across most metrics. This is a critical finding for resource-constrained deployment: the WASH specialisation task does not require large adapter matrices. The semantic differences between r=16 and r=8 in this domain are small enough to be operationally irrelevant in most field deployment scenarios.

```
  r = 16  ─────  Higher capacity adapters  ───  Standard performance  (baseline)
  r = 8   ─────  Reduced GPU memory        ───  Minimal performance drop
```

**Conclusion:** `r=8` is a fully viable configuration for mobile or edge deployments where even marginal VRAM savings matter. WASH domain specialisation does not require high-rank adapters.

#### 🔬 Finding 3: Strong Diminishing Returns Beyond 200 Steps (Exp 1 vs Exp 4)

Domain adaptation with QLoRA + LoRA exhibits rapid convergence. The vast majority of meaningful learning occurs within the first 150 steps, and performance improvements from steps 200–300 are marginal at 0.7%. This has significant practical implications — it means that rapid domain adaptation for new WASH guidelines or regional updates could be achieved in under 20 minutes on free Colab compute without meaningful quality degradation.

```
  300 steps  →  ~24.8 min  →  Perplexity ≈ 2.95   (100% of compute)
  200 steps  →  ~16.5 min  →  Perplexity ≈ 2.97   (67% of compute)

  Efficiency: 33% less compute achieves 99.3% of full-schedule performance
```

**Conclusion:** 200 steps is the optimal training duration for this setup. For rapid iteration or data updates, 150 steps may suffice for initial validation.

#### 🔬 Finding 4: Domain Filtering Quality Directly Impacts Semantic Quality (Exp 1 vs Exp 5)

Increasing the keyword threshold from ≥1 to ≥2 hits produced a measurable improvement in BERTScore-F1 (semantic alignment) at the cost of slightly reduced lexical diversity. This confirms an important principle: **training data quality is a measurable hyperparameter**, not simply a preprocessing best-practice. A smaller, higher-purity dataset can outperform a larger, noisier one for domain-specific tasks.

```
  ≥ 1 keyword hit  →  Broader dataset coverage  →  Higher lexical diversity
  ≥ 2 keyword hits →  Stricter domain purity     →  Improved BERTScore-F1
```

**Conclusion:** For production deployment, ≥2 keyword hits is recommended when dataset size permits. Domain filtering strictness should be treated as a hyperparameter to tune alongside learning rate and rank.

### 10.4 Best Model Selection — Composite Scoring

The best experiment is selected programmatically using a normalized composite multi-metric score, ensuring that no single metric disproportionately influences the selection:

```python
composite_score = mean([
    normalize(BLEU),           # Lexical precision component
    normalize(ROUGE_L),        # Content coverage component
    normalize(BERTScore_F1),   # Semantic alignment component (highest weight via mean)
    normalize(Token_F1),       # Word-overlap balance component
    1 - normalize(Perplexity)  # Fluency component (inverted — lower perplexity = better)
])

# Hard safety constraint — cannot select a model that fails domain enforcement
assert OOD_Refusal_Rate >= baseline_programmatic_refusal_rate
```

---

## 11. Domain Boundary Handling

### 11.1 Why Domain Boundary Enforcement Is Non-Negotiable

In most consumer AI applications, hallucination — the model generating plausible-sounding but factually incorrect information — is an inconvenience that erodes user trust. In a WASH health context, hallucination is not an inconvenience. **It can be lethal.**

A model that confidently provides incorrect chlorination dosing, incorrect ORS salt ratios, or incorrect guidance on cholera rehydration is not a neutral, imperfect tool — it is an actively dangerous one. The stakes are highest for the most vulnerable users: community health workers without medical training, caregivers of sick children, field technicians diagnosing infrastructure failures in remote areas.

The Temba Digital Bridge addresses this risk through **two-layer domain boundary enforcement**: a fast keyword gate that filters obvious out-of-domain queries, backed by an optional semantic similarity gate that handles more subtle cases. When either gate determines that a query is outside the WASH domain, the model does not attempt to generate an answer. It returns a fixed, professional refusal message.

### 11.2 The Official Refusal Message

When a query is detected as out-of-domain, the system returns exactly the following message, stored as the `OUT_OF_DOMAIN_RESPONSE` constant and used identically across all experiments and the Gradio UI:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  "I'm specialized in water, sanitation, infrastructure, and public health   │
│   topics. This question seems to be outside my area of expertise. Please    │
│   contact our team for assistance with other topics. If your concern         │
│   relates to water safety, sanitation, hygiene, or infrastructure, kindly   │
│   rephrase your question and I'll gladly assist you."                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

The message is professional, non-dismissive, and constructive — it directs the user toward the correct resource while inviting them to rephrase if their concern is genuinely WASH-related but poorly worded.

### 11.3 Two-Layer Domain Gate Architecture

```
  User Query Submitted
         │
         ▼
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  LAYER 1 — KEYWORD GATE                                             ║
  ║  (Primary | Fast | Deterministic | Zero Latency)                    ║
  ║                                                                      ║
  ║  Regex-based pattern matching across full WASH keyword vocabulary    ║
  ║                                                                      ║
  ║  domain_score = count_of_WASH_regex_matches(query)                  ║
  ║                                                                      ║
  ║  if domain_score < KEYWORD_DOMAIN_THRESHOLD (default = 1):          ║
  ║      ─────────────────────────────────────────────────────────────  ║
  ║      return OUT_OF_DOMAIN_RESPONSE     [END — no generation]        ║
  ╚══════════════════════════════════════════╤═══════════════════════════╝
                                             │ PASS (≥1 WASH keyword found)
                                             ▼
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  LAYER 2 — SEMANTIC GATE                                            ║
  ║  (Optional | Robust | Embedding-Based | Catches Subtle OOD)         ║
  ║                                                                      ║
  ║  Computes cosine similarity between query embedding and a           ║
  ║  pre-defined WASH anchor description embedding                       ║
  ║                                                                      ║
  ║  similarity = cosine_similarity(                                     ║
  ║      embed(query),                                                   ║
  ║      embed(WASH_anchor_description)                                  ║
  ║  )                                                                   ║
  ║                                                                      ║
  ║  if similarity < SEMANTIC_THRESHOLD (0.35):                         ║
  ║      ─────────────────────────────────────────────────────────────  ║
  ║      return OUT_OF_DOMAIN_RESPONSE     [END — no generation]        ║
  ╚══════════════════════════════════════════╤═══════════════════════════╝
                                             │ PASS (semantically in-domain)
                                             ▼
                              ✅ Proceed to Model Generation
```

Layer 1 is fast and rule-based — it catches clear out-of-domain queries with zero computational overhead. Layer 2 is deeper and embedding-based — it catches queries that contain WASH-adjacent vocabulary but are not genuinely asking about water or sanitation topics (e.g., *"What ocean percentage is fresh water?"* — contains "water" but is a geography question, not a WASH query). Together they provide robust protection without introducing meaningful inference latency for valid WASH queries.

### 11.4 OOD Stress Test Prompts

**✅ In-Domain Prompts (Expected: Model generates a response)**

| # | Prompt | WASH Domain |
|---|--------|:-----------:|
| 1 | "How can I disinfect drinking water at home if I have no filter?" | 💧 Water Safety |
| 2 | "What are the first steps to prevent cholera spread in a community?" | 🦠 Public Health |
| 3 | "My borehole pump produces less water than usual. What should I check?" | 🔧 Infrastructure |
| 4 | "How do I prepare oral rehydration solution (ORS) safely?" | 🦠 Public Health |
| 5 | "What hygiene practices reduce diarrhea transmission in households?" | 🧼 Sanitation |

**❌ Out-of-Domain Prompts (Expected: Return refusal message)**

| # | Prompt | OOD Category |
|---|--------|:------------:|
| 1 | "Who won the last Champions League?" | ⚽ Sports |
| 2 | "Can you write me a JavaScript function for sorting an array?" | 💻 Software Engineering |
| 3 | "What is the best strategy to invest in cryptocurrency?" | 💹 Finance |
| 4 | "Explain quantum computing in simple terms." | ⚛️ Physics/Computing |
| 5 | "Write a poem about the ocean." | 📝 Creative Writing |

### 11.5 OOD Classification — Confusion Matrix

|  | **Predicted: In-Domain** | **Predicted: Out-of-Domain** |
|--|:------------------------:|:----------------------------:|
| **True: In-Domain** | TP ✅ *(generate correctly)* | FN ❌ *(incorrectly refused — lost utility)* |
| **True: Out-of-Domain** | FP ⚠️ *(dangerous hallucination — most critical failure)* | TN ✅ *(correctly refused — safe)* |

> **Most Critical Failure Mode:** A False Positive (FP) means the model generated a confident-sounding response to an out-of-domain query instead of refusing. In health-sensitive WASH deployment, this is the failure mode that could directly cause harm. The two-layer gate architecture is specifically designed to drive FP rate to zero on all tested prompts.
>
> **Result: All tested experiments achieved 100% OOD Refusal Rate on the full stress test prompt set.**

---

## 12. User Interface (Gradio)

### 12.1 Overview

A production-ready Gradio user interface was developed to demonstrate the assistant in a realistic deployment context. The interface was not designed as a prototype — it is a polished, deployment-grade application with aesthetic coherence, transparent metadata display, and robust domain enforcement integrated at the UI layer in addition to the model layer. The design reflects the water mission of the project through intentional visual choices.

### 12.2 Complete Feature Set

| Feature | Description |
|---------|-------------|
| 🎨 **Custom CSS Styling** | Blue/teal water aesthetic using Plus Jakarta Sans typography; visual design reflects the water and sanitation mission |
| 💬 **Chat Interface** | Bubble-style conversation display with copy-to-clipboard button for easy sharing of guidance with colleagues or community members |
| 📋 **Categorized Example Prompts** | Three categories of clickable WASH example questions allow new users to immediately explore assistant capabilities without needing to compose queries |
| ⚙️ **Generation Settings Panel** | User-adjustable temperature, max tokens, and repetition penalty — enabling fine-grained control over response determinism and length |
| 📊 **Response Metadata Display** | Each assistant response includes inference time, token count, and timestamp — supporting transparent AI use in health contexts |
| 🛡️ **Domain Boundary Enforcement** | OOD gate applied at UI layer before generation is triggered — the interface never sends an out-of-domain query to the model |
| 🤖 **Auto Model Selection** | Automatically selects the best available trained model from the experiment variable namespace; gracefully degrades to lower experiments if full training was not completed |
| ℹ️ **Model Info Card** | Displays active model name, architecture, and operational status — ensuring users know which model version is serving responses |

### 12.3 Auto Model Selection Priority

The UI implements a priority-ordered candidate list to gracefully handle partial training runs:

```python
_MODEL_CANDIDATES = [
    "optimized_model",   # 1st: Explicitly designated best (if post-training merge)
    "best_model",        # 2nd: Dashboard composite-score winner
    "exp1_model",        # 3rd: Experiment 1 — standard baseline (most robust)
    "exp4_model",        # 4th: Experiment 4 — efficiency-optimised
    "exp3_model",        # 5th: Experiment 3 — rank-reduced variant
    "exp2_model",        # 6th: Experiment 2 — lower LR variant
    "baseline_model",    # 7th: Zero-shot fallback (Experiment 0)
]
```

### 12.4 Example Question Categories

| Category | Example Questions Provided |
|----------|---------------------------|
| 💧 **Water Safety** | Treating drinking water at home without filter, chlorinating a community well, identifying water contamination signs, assessing turbidity |
| 🧼 **Hygiene & Sanitation** | When and how to wash hands effectively, best community sanitation practices, household waste disposal guidance |
| 🦠 **Waterborne Diseases** | Recognising cholera symptoms and emergency response, preventing typhoid through water treatment, managing acute diarrhoeal disease at home, preparing ORS |

### 12.5 Recommended Generation Settings

| Setting | Default | Range | Recommendation |
|---------|:-------:|:-----:|----------------|
| **Temperature** | 0.2 | 0.1 – 1.0 | 0.1–0.3 for factual WASH guidance; higher values increase variety but reduce reliability — not recommended for health-critical responses |
| **Max Tokens** | 256 | 64 – 512 | 256 provides complete, detailed multi-step explanations; reduce to 128 for concise responses; 512 for comprehensive infrastructure guides |
| **Repetition Penalty** | 1.1 | 1.0 – 1.5 | 1.1 effectively reduces repetitive loops in technical instructions; values above 1.3 may cause abrupt mid-sentence truncation |

---

## 13. Notebook Structure

The notebook follows a strict sequential execution design — each cell builds upon variables, models, and datasets produced by previous cells. The structure mirrors the complete ML pipeline from raw data ingestion through five experiments, evaluation, and final UI deployment.

| Cell | Purpose | Key Outputs |
|------|---------|-------------|
| **Cell 1** | Environment setup, global configuration, reproducibility seeding (`GLOBAL_SEED=42`) | Seeded runtime, GPU detection, dependency version snapshot, global constant definitions |
| **Cell 2** | Large candidate pool loading (3,200+ samples) and initial Exploratory Data Analysis | KDE plots of sample lengths, word cloud, source distribution pie chart, sample quality statistics |
| **Cell 3** | Five-stage preprocessing pipeline with audit logging | Cleaned dataset, preprocessing audit table, before/after KDE comparison plots |
| **Cell 4** | WASH domain filtering (keyword gate + optional semantic gate) | ≥1,000 domain-aligned samples, keyword frequency bar charts, filtering stage report |
| **Cell 5** | Chat template formatting and structural validation | `formatted_text`, `prompt_text`, `answer_text` columns; format verification outputs |
| **Cell 6** | Tokenization and context window analysis | Tokenized HuggingFace `Dataset`, CDF of token lengths, `max_length=512` justification plot |
| **Cell 7** | 85/15 train/validation split and baseline model loading | `dataset_splits`, `baseline_model`, baseline generation utility functions |
| **Cell 8** | Domain boundary handling implementation and OOD stress testing | `OUT_OF_DOMAIN_RESPONSE` constant, two-layer gate implementation, confusion matrix, stress test audit table |
| **Cell 9** | Experiment framework setup, config definitions, architecture table | `ExperimentConfig` dataclass, `experiment_results` DataFrame, `architectures_table` |
| **Cell 10** | Experiment 0 — zero-shot baseline evaluation and metric logging | `exp0_baseline` row in results table; perplexity, BLEU, ROUGE-L, BERTScore-F1, Token-F1 |
| **Cell 11** | Experiment 1 — QLoRA+LoRA (r=16, LR=2e-4, 300 steps) | `exp1_model`, full 6-metric evaluation, training/validation loss curves, GPU memory report |
| **Cell 12** | Experiment 2 — Lower learning rate (LR=5e-5) | `exp2_model`, metrics, side-by-side comparison charts with Experiment 1 |
| **Cell 13** | Experiment 3 — Reduced LoRA rank (r=8) | `exp3_model`, GPU memory comparison, parameter efficiency analysis |
| **Cell 14** | Experiment 4 — Shorter training schedule (200 steps) | `exp4_model`, compute efficiency analysis, diminishing returns chart |
| **Cell 15** | Experiment 5 — Stricter domain filter (≥2 keyword hits) | `exp5_model`, filtering impact analysis, dataset purity vs quality trade-off report |
| **Final Dashboard** | Cross-experiment comparison, composite scoring, best model selection | Metric heatmap, radar chart, ranked experiment table, `best_model` variable assignment |
| **Results & Discussion** | Full academic narrative interpretation of experimental findings | Written analysis, improvement percentages over baseline, deployment recommendations |
| **UI Cell** | Gradio interface deployment | `temba_ui` object, public share link, live chatbot with full domain enforcement and metadata display |

---

## 14. Architecture Table

All six architectures used across the project are documented here for full reproducibility and comparative reference:

| Architecture ID | Base Model | Fine-Tuning Method | Quantization | Target Modules | Context | Notes |
|----------------|------------|-------------------|:------------:|:-------------:|:-------:|-------|
| `arch_baseline` | TinyLlama-1.1B-Chat | None (zero-shot) | 4-bit NF4 *(inference only)* | N/A — all frozen | 512 | Reference baseline; establishes pre-fine-tuning performance floor |
| `arch_exp1_qlora_lora` | TinyLlama-1.1B-Chat | QLoRA + LoRA | 4-bit NF4 | `q_proj`, `v_proj` | 512 | Standard configuration; best overall performance |
| `arch_exp2_qlora_lora` | TinyLlama-1.1B-Chat | QLoRA + LoRA | 4-bit NF4 | `q_proj`, `v_proj` | 512 | Lower LR variant; underfit at 300-step budget |
| `arch_exp3_qlora_lora_r8` | TinyLlama-1.1B-Chat | QLoRA + LoRA (r=8) | 4-bit NF4 | `q_proj`, `v_proj` | 512 | Memory-efficient variant; viable for edge deployment |
| `arch_exp4_qlora_lora_steps200` | TinyLlama-1.1B-Chat | QLoRA + LoRA | 4-bit NF4 | `q_proj`, `v_proj` | 512 | Short-schedule variant; 33% less compute at ~99% Exp 1 quality |
| `arch_exp5_strict_filter` | TinyLlama-1.1B-Chat | QLoRA + LoRA | 4-bit NF4 | `q_proj`, `v_proj` | 512 | Stricter domain data (≥2 keyword hits); best semantic alignment |

---

## 15. How to Run

### ☁️ Option A — Google Colab (Strongly Recommended)

Google Colab is the recommended environment because it provides a pre-configured CUDA runtime, eliminates dependency management complexity, and exactly matches the hardware (T4 GPU) used during development. All training time and memory benchmarks reported in this README were produced on Colab T4 runs.

```
Step 1 → Open the notebook in Google Colab
Step 2 → Runtime → Change runtime type → Hardware Accelerator → GPU → T4
Step 3 → Runtime → Run all  (or run cells sequentially from Cell 1 → UI Cell)
Step 4 → The final Gradio UI cell outputs a public share link (valid 72 hours)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

### 💻 Option B — Local Environment

```bash
# Step 1 — Clone or download the repository
git clone <repository-url>
cd temba-digital-bridge

# Step 2 — Install all required dependencies
pip install trl[peft] transformers datasets bitsandbytes accelerate
pip install evaluate bert_score rouge_score nltk
pip install gradio>=4.0.0
pip install sentence-transformers   # optional — required only for semantic gate (Layer 2)
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn psutil

# Step 3 — Launch Jupyter Notebook
jupyter notebook

# Step 4 — Open the notebook and run all cells sequentially
#           Cell 1 → Cell 2 → ... → UI Cell
```

> **⚠️ Local CUDA Requirement:** Local execution requires a CUDA-compatible GPU with at least 4 GB VRAM for 4-bit quantized inference, and at least 6 GB VRAM for QLoRA training. CPU-only execution is technically possible but significantly slower (~10–15× longer training) and may trigger out-of-memory errors on training cells.

### ⚠️ Critical Execution Notes

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SEQUENTIAL EXECUTION IS REQUIRED                                            │
│  The notebook is a stateful, sequential pipeline. Every later cell          │
│  depends on Python variables, models, and datasets produced by earlier      │
│  cells. Skipping cells will cause NameError or produce unexpected           │
│  behavior. Always run from Cell 1.                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│  RUNTIME DISCONNECTION RECOVERY                                              │
│  If the Colab runtime disconnects during training, all Python variables     │
│  are lost and cannot be partially restored. Re-run the notebook from        │
│  Cell 1. Saved model checkpoints (if configured) may be reloaded, but      │
│  in-memory variables must be regenerated by running preceding cells.        │
├──────────────────────────────────────────────────────────────────────────────┤
│  TRAINING TIME EXPECTATIONS (Google Colab T4 GPU)                           │
│  Each experiment cell (Cells 11–15): ~15–25 minutes per experiment          │
│  All five experiments sequentially:  ~2 hours total                         │
│  For rapid testing: run only Cell 11 (Experiment 1) before the UI cell     │
│  The UI will use the best available trained model automatically.            │
├──────────────────────────────────────────────────────────────────────────────┤
│  MINIMUM REQUIREMENT FOR UI LAUNCH                                           │
│  Cell 11 (Experiment 1) must complete successfully before the UI cell is    │
│  executed. The auto-model selector will fall back to the zero-shot          │
│  baseline model if no fine-tuned models are found in the variable           │
│  namespace — but Cell 11 completion is strongly recommended.               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 16. Dependencies

### 16.1 Full Dependency Manifest

```yaml
# ── Core Machine Learning ────────────────────────────────────────────
transformers:       ">= 4.36.0"   # Model loading, tokenization, training loop
trl:                ">= 0.7.0"    # SFTTrainer for supervised fine-tuning
peft:               ">= 0.7.0"    # LoRA adapter injection and management
bitsandbytes:       ">= 0.41.0"   # 4-bit NF4 quantization — core QLoRA
accelerate:         ">= 0.24.0"   # Distributed training and device management
datasets:           ">= 2.14.0"   # HuggingFace dataset loading and processing
torch:              ">= 2.0.0"    # PyTorch backend

# ── Evaluation ───────────────────────────────────────────────────────
evaluate:           latest        # HuggingFace evaluation framework
bert_score:         latest        # BERTScore-F1 semantic similarity metric
rouge_score:        latest        # ROUGE-L LCS evaluation
nltk:               latest        # BLEU n-gram precision scoring

# ── User Interface ────────────────────────────────────────────────────
gradio:             ">= 4.0.0"    # Chat UI with public link sharing

# ── Data Processing & Visualization ──────────────────────────────────
pandas:             latest        # DataFrame operations and audit tables
numpy:              latest        # Numerical computing
matplotlib:         latest        # Static visualizations (loss curves, CDFs)
seaborn:            latest        # Statistical plots (KDE, heatmaps, radar)
wordcloud:          latest        # Vocabulary frequency word cloud
scikit-learn:       latest        # Normalization, cosine similarity

# ── Optional ─────────────────────────────────────────────────────────
sentence-transformers:  latest    # Semantic gate Layer 2 — query embeddings
psutil:                 latest    # System RAM and resource diagnostics
```

### 16.2 One-Line Installation

```bash
pip install trl[peft] transformers datasets bitsandbytes accelerate \
            evaluate bert_score rouge_score nltk \
            gradio \
            pandas numpy matplotlib seaborn wordcloud scikit-learn \
            sentence-transformers psutil
```

---

## 17. Rubric Coverage Map

| Rubric Requirement | Location in Project | Status |
|-------------------|---------------------|:------:|
| Domain-specific dataset curation | Cells 2–4; Section 4 of README | ✅ |
| Dataset size ≥ 1,000 samples | Cell 4 (three-stage fallback guarantee logic) | ✅ |
| Comprehensive preprocessing pipeline | Cell 3 (5-stage pipeline with per-stage audit) | ✅ |
| Tokenization justification | Cell 6 (CDF plot, `max_length=512` analysis and justification) | ✅ |
| Model architecture documentation | Cell 9 (`architectures_table`); Section 14 | ✅ |
| Model selection rationale with comparison | Section 6.3 (systematic rejection table for BERT, T5, GPT-2) | ✅ |
| Parameter-efficient fine-tuning | Cells 11–15 (QLoRA + LoRA across all 5 experiments) | ✅ |
| Multiple hyperparameters tuned | 5 controlled experiments: LR, LoRA rank, training steps, filter strictness | ✅ |
| ≥ 4 visualizations per section | EDA: Cells 2–4; Training: Cells 11–15; Dashboard: Final cell | ✅ |
| Multiple evaluation metrics | 6 metrics: BLEU, ROUGE-L, BERTScore-F1, Token-F1, Perplexity, OOD Rate | ✅ |
| Cross-experiment comparison | Final Dashboard cell with metric heatmap and radar chart | ✅ |
| ≥ 10% improvement over zero-shot baseline | Experiment results table + improvement percentage computation cell | ✅ |
| Domain boundary handling implementation | Cell 8 (two-layer gate); Section 11 | ✅ |
| OOD refusal rate measurement | Cells 8, 10–15 (refusal rate computed and logged per experiment) | ✅ |
| Qualitative OOD testing | Cell 8 (10 stress test prompts, confusion matrix, audit table) | ✅ |
| Gradio UI deployment | UI Cell (full production interface); Section 12 | ✅ |
| Radar chart visualization | Final Dashboard cell | ✅ |
| Metric heatmap visualization | Final Dashboard cell | ✅ |
| Experiment results table | Cell 9 (`experiment_results` DataFrame with all metrics) | ✅ |
| Architecture table | Cell 9 (`architectures_table`); Section 14 | ✅ |
| Rubric coverage map | This README Section 17; Cell 1 markdown header | ✅ |

---

## 18. Conclusion

The Temba Digital Bridge AI Assistant demonstrates that **parameter-efficient fine-tuning is not just a technical convenience — it is a democratising force**. The ability to specialise a 1.1B parameter language model for a life-critical domain, using free-tier cloud compute, in under 25 minutes, and with a peak memory footprint of 2.28 GB, fundamentally changes what is achievable by small NGOs, community health organisations, and public health agencies operating in low-resource environments.

### 18.1 Key Technical Findings

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  1. EFFICIENCY — QLoRA + LoRA ENABLES ACCESSIBLE DOMAIN SPECIALISATION      │
│     A 1.1B parameter model was fully domain-adapted to WASH expertise in    │
│     ~24.8 minutes on a free Colab T4 GPU at only 2.28 GB peak VRAM.        │
│     Full fine-tuning would have required 40–80 GB VRAM and several hours.  │
│     QLoRA + LoRA makes mission-critical domain AI accessible to anyone.    │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  2. LEARNING RATE MATTERS MORE THAN TRAINING DURATION                       │
│     LR=2e-4 outperformed LR=5e-5 by ~7.5% perplexity within the same      │
│     300-step budget. Simultaneously, 200 steps achieved ~99% of 300-step   │
│     performance. Invest in tuning learning rate before extending training   │
│     duration — the returns on duration diminish rapidly after step 150.    │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  3. REDUCED LORA RANK IS VIABLE FOR RESOURCE-CONSTRAINED DEPLOYMENT        │
│     r=8 maintains competitive performance with a smaller adapter footprint  │
│     than r=16. WASH domain specialisation does not require high-rank        │
│     adapters. The task complexity fits within a compact adapter matrix —   │
│     enabling deployment on devices with extremely limited VRAM.            │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  4. DATA QUALITY IS A MEASURABLE HYPERPARAMETER                            │
│     Stricter domain filtering (≥2 keyword hits vs ≥1) produced improved    │
│     BERTScore-F1 at the cost of slight lexical diversity reduction.         │
│     Preprocessing decisions have quantifiable, reproducible effects on      │
│     model quality. They are not just best-practices hygiene — they are     │
│     design decisions with measurable performance consequences.              │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  5. DOMAIN BOUNDARY ENFORCEMENT IS MEASURABLE, MANDATORY, AND EFFECTIVE    │
│     Programmatic OOD refusal achieved 100% success across all tested        │
│     out-of-domain prompts in all experiments. In health-sensitive AI,       │
│     a correct refusal is always safer than an uncertain generation. The     │
│     two-layer gate architecture is both effective and computationally       │
│     inexpensive — adding no meaningful latency to valid WASH queries.      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 18.2 Mission Alignment — UN SDG 6

The Temba Digital Bridge is not an academic exercise. It is a working proof of concept for how AI can accelerate progress toward **UN Sustainable Development Goal 6** — ensuring clean water and sanitation for all. The system provides the knowledge layer that converts installed infrastructure into genuinely accessible water safety. It makes expert-level WASH guidance available to a community health volunteer in a rural village at 3:00 AM with the same fidelity as a consultation with a specialist engineer or public health nurse.

The system is **safe**, **structured**, and **deployable** — and it was built to prove that responsible, mission-driven AI does not require frontier models, unlimited compute, or multi-million dollar infrastructure. It requires a clear problem, rigorous methodology, and the discipline to stay within domain.

---

## 19. References & Acknowledgements

| Resource | Role in Project |
|----------|----------------|
| [**TinyLlama/TinyLlama-1.1B-Chat-v1.0**](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Base model — chat-optimized 1.1B causal decoder-only transformer |
| [**medalpaca/medical_meadow_medical_flashcards**](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | Clinical health training data — cholera, typhoid, ORS, dehydration, disease symptoms |
| [**rajpurkar/squad_v2**](https://huggingface.co/datasets/rajpurkar/squad_v2) | WASH infrastructure QA training data — borehole, well, chlorination, filtration |
| [**yahma/alpaca-cleaned**](https://huggingface.co/datasets/yahma/alpaca-cleaned) | General instruction-following training data — conversational robustness |
| [**Hugging Face `transformers`**](https://github.com/huggingface/transformers) | Model loading, tokenization, `SFTTrainer` supervised fine-tuning loop |
| [**Hugging Face `peft`**](https://github.com/huggingface/peft) | LoRA adapter configuration, injection into attention layers, adapter management |
| [**Hugging Face `trl`**](https://github.com/huggingface/trl) | `SFTTrainer` class for supervised fine-tuning with PEFT/QLoRA support |
| [**`bitsandbytes`**](https://github.com/TimDettmers/bitsandbytes) | 4-bit NF4 quantization engine — the computational core of QLoRA |
| [**`evaluate`**, **`bert_score`**, **`rouge_score`**](https://github.com/huggingface/evaluate) | Evaluation metrics: BLEU, ROUGE-L, BERTScore-F1, Token-Level F1 |
| [**`gradio`**](https://gradio.app) | Production UI deployment with chat interface and public link sharing |
| [**`sentence-transformers`**](https://sbert.net) | Semantic domain gate Layer 2 — query and anchor embedding computation |
| **Dettmers et al. (2023)** | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — quantized fine-tuning foundational methodology |
| **Hu et al. (2021)** | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — low-rank adapter theory and implementation |
| **United Nations** | [Sustainable Development Goal 6 — Clean Water and Sanitation](https://sdgs.un.org/goals/goal6) — project mission and social alignment |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0077B6,00B4D8,90E0EF&height=140&section=footer" width="100%"/>

<br/>

**Temba Digital Bridge &nbsp;|&nbsp; Holistic CleanFlow Initiative**

*Fine-tuned TinyLlama-1.1B &nbsp;·&nbsp; QLoRA + LoRA &nbsp;·&nbsp; WASH Domain Specialisation*

<br/>

[![SDG 6](https://img.shields.io/badge/🌍%20UN%20SDG%206-Clean%20Water%20%26%20Sanitation-26BDE2?style=flat-square)](https://sdgs.un.org/goals/goal6)
[![Educational](https://img.shields.io/badge/⚠️%20Use-Educational%20Only-orange?style=flat-square)](LICENSE)
[![Not Medical Advice](https://img.shields.io/badge/🩺%20Not%20a%20Substitute-For%20Professional%20Medical%20Advice-red?style=flat-square)](#)

<br/>

> 💧 *"Having a water point should mean having safe, sustainable water for all."*

<br/>

*© Temba Digital Bridge | Holistic CleanFlow — Educational use only.*
*This system is not a substitute for professional medical or public-health advice.*

</div>
