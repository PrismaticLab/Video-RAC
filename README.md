<div align="center">

# 🪄🎓 **VideoRAC**: *Retrieval-Adaptive Chunking for Lecture Video RAG*

</div>

<div align="center">

<img src="https://github.com/PrismaticLab/Video-RAC/blob/main/docs/assets/logo.png?raw=true" alt="VideoRAC Logo" width="300"/>

### 🏛️ *Official CSICC 2025 Implementation*

#### "Adaptive Chunking for VideoRAG Pipelines with a Newly Gathered Bilingual Educational Dataset"

*(Presented at the 30th International Computer Society of Iran Computer Conference — CSICC 2025)*

[![Paper](https://img.shields.io/badge/Paper-CSICC%202025-blue)](https://ieeexplore.ieee.org/document/10967455)
[![Dataset](https://img.shields.io/badge/Dataset-EduViQA-orange)](https://huggingface.co/datasets/UIAIC/EduViQA)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

</div>

---

## 📊 Project Pipeline

<div align="center">

<!-- ✨ Placeholder for horizontal pipeline image ✨ -->

<img src="https://github.com/PrismaticLab/Video-RAC/blob/main/docs/assets/fig-2.png?raw=true" alt="VideoRAC Pipeline" width="900"/>

</div>

---

## 📖 Overview

We present **Video-RAC**, an adaptive chunking methodology for lecture videos within Retrieval-Augmented Generation (RAG) pipelines. Using **CLIP embeddings** and **SSIM** to detect coherent slide transitions, plus **entropy-based keyframe selection**, we construct multimodal chunks that align audio transcripts and visual frames.

Alongside the method, we release **EduViQA**, a slide-centric, bilingual (Persian/English) lecture dataset containing 20 videos from 5 professors across STEM and education topics. Each lecture is paired with 50 synthetic QA items and categorized by duration (40% mid-length, ~20–40 minutes) to support controlled RAG benchmarking.

This repository is the **official implementation** of the CSICC 2025 paper by *Hemmat et al.*

> **Hemmat, A., Vadaei, K., Shirian, M., Heydari, M.H., Fatemi, A.**
> *“Adaptive Chunking for VideoRAG Pipelines with a Newly Gathered Bilingual Educational Dataset.”*
> *Proceedings of the 30th International Computer Society of Iran Computer Conference (CSICC 2025), University of Isfahan.*

---

## 🧠 Research Background

This framework underpins the **EduViQA bilingual dataset**, designed for evaluating lecture-based RAG systems in both Persian and English. The dataset and code form a unified ecosystem for multimodal question generation and retrieval evaluation.

**Key Contributions:**

* 🎥 Adaptive Hybrid Chunking — Combines CLIP cosine similarity with SSIM-based visual comparison.
* 🧮 Entropy-Based Keyframe Selection — Extracts high-information frames for retrieval.
* 🗣️ Transcript–Frame Alignment — Synchronizes ASR transcripts with visual semantics.
* 🔍 Multimodal Retrieval — Integrates visual and textual embeddings for RAG.
* 🧠 Benchmark Dataset — 20 bilingual educational videos with 50 QA pairs each.

---

## 📊 Dataset

### EduViQA: Bilingual Educational Video QA Dataset

![Dataset Composition](docs/assets/fig-1.png)
*Dataset composition highlighting topic distribution and lecture duration proportions.*

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Videos** | 20 (10 Persian, 10 English) |
| **Professors** | 5 |
| **Duration Focus** | 40% mid-length (20–40 minutes) |
| **QA Pairs per Video** | 50 synthetic QA pairs |
| **Format** | JSON annotations |

### Topics Covered
- Computer Architecture
- Data Structures
- System Dynamics and Control
- Teaching Skills
- Descriptive Research
- Regions in Human Geography
- Differentiated Instruction
- Business

The dataset also captures slide transitions and keyframes extracted via CLIP+SSIM chunking, enabling multimodal retrieval experiments with aligned visuals and transcripts.

**📥 Access Dataset:** [Hugging Face - EduViQA](https://huggingface.co/datasets/UIAIC/EduViQA)

---

## ⚙️ Installation

```bash
pip install VideoRAC
```

---

## 🚀 Usage Example

### 1️⃣ Hybrid Chunking

```python
from VideoRAC.Modules import HybridChunker

chunker = HybridChunker(
    clip_model='openai/clip-vit-base-patch32',
    alpha=0.6,
    threshold_embedding=0.85,
    threshold_ssim: float=0.8,
    interval: int=1,
)
chunks, timestamps, duration = chunker.chunk("lecture.mp4")
chunker.evaluate()
```

### 2️⃣ Q&A Generation

```python
from VideoRAC.Modules import VideoQAGenerator

def my_llm_fn(messages):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

urls = ["https://www.youtube.com/watch?v=2uYu8nMR5O4"]
qa = VideoQAGenerator(video_urls=urls, llm_fn=my_llm_fn)
qa.process_videos()
```

---

## 📈 Results Summary (CSICC 2025)

| Method                   | AR       | CR       | F        | Notes                        |
| ------------------------ | -------- | -------- | -------- | ---------------------------- |
| **VideoRAC (CLIP+SSIM)** | **0.87** | **0.82** | **0.91** | Best performance overall     |
| CLIP-only                | 0.80     | 0.75     | 0.83     | Weaker temporal segmentation |
| Simple Slicing           | 0.72     | 0.67     | 0.76     | Time-based only              |

> Evaluated using RAGAS metrics: *Answer Relevance (AR)*, *Context Relevance (CR)*, and *Faithfulness (F)*.

---

## 🧾 License

Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You may share and adapt this work with attribution. Please cite our paper when using VideoRAC or EduViQA:

```bibtex
@INPROCEEDINGS{10967455,
  author={Hemmat, Arshia and Vadaei, Kianoosh and Shirian, Melika and Heydari, Mohammad Hassan and Fatemi, Afsaneh},
  booktitle={2025 29th International Computer Conference, Computer Society of Iran (CSICC)}, 
  title={Adaptive Chunking for VideoRAG Pipelines with a Newly Gathered Bilingual Educational Dataset}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  keywords={Measurement;Visualization;Large language models;Pipelines;Retrieval augmented generation;Education;Question answering (information retrieval);Multilingual;Standards;Context modeling;Video QA;Datasets Preparation;Academic Question Answering;Multilingual},
  doi={10.1109/CSICC65765.2025.10967455}}
```

---

## 👥 Authors

**University of Isfahan — Department of Computer Engineering**

* **Kianoosh Vadaei** — [kia.vadaei@gmail.com](mailto:kia.vadaei@gmail.com)
* **Melika Shirian** — [mel.shirian@gmail.com](mailto:mel.shirian@gmail.com)
* **Arshia Hemmat** — [amirarshia.hemmat@kellogg.ox.ac.uk](mailto:amirarshia.hemmat@kellogg.ox.ac.uk)
* **Mohammad Hassan Heydari** — [heidary0081@gmail.com](mailto:heidary0081@gmail.com)
* **Afsaneh Fatemi** — [a.fatemi@eng.ui.ac.ir](mailto:a.fatemi@eng.ui.ac.ir)

---

<div align="center">

**⭐ Official CSICC 2025 Implementation — Give it a star if you use it in your research! ⭐**
*Made with ❤️ at University of Isfahan*

</div>
