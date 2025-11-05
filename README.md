# 🚗 Smart Driving Feedback System (RAG + AI)

A **Retrieval-Augmented Generation (RAG)** based AI assistant that provides **personalized driving safety feedback** using real-time contextual data.  
This project leverages **LLMs**, **web retrieval**, and **data-driven insights** to generate **human-like driving recommendations** through an interactive **Streamlit dashboard**.

---

## 🧠 Overview

Traditional driver feedback systems rely purely on rule-based sensor data.  
This project introduces a hybrid approach — combining **LLM-based reasoning (Flan-T5)** with **web retrieval via DuckDuckGo API**, and **domain-specific rules** to deliver *context-aware* insights and natural language recommendations to the driver.

---

## 🧩 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue?logo=python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/LLM-Flan-T5-green?logo=huggingface" />
  <img src="https://img.shields.io/badge/RAG-LangChain-orange?logo=langchain" />
  <img src="https://img.shields.io/badge/Web%20Search-DuckDuckGo%20API-purple?logo=duckduckgo" />
</p>

---

## ⚙️ Features

✅ **Real-time feedback** — RAG-based AI model analyzes driving context and suggests safety improvements.  
✅ **Hybrid reasoning system** — Combines sensor-logic rules with natural-language generation.  
✅ **Web retrieval** — Uses DuckDuckGo API to fetch the latest safety insights and contextual driving tips.  
✅ **Interactive dashboard** — Clean and simple Streamlit UI for seamless driver interaction.  
✅ **Multilingual support** — Generates feedback in multiple languages for regional drivers.  
✅ **Extensible architecture** — Easy to integrate with vehicle telemetry or IoT-based data pipelines.

---

## 🧱 Project Structure
```
├── app.py → Main Streamlit application
├── rag_engine/ → Core RAG and retrieval modules
├── utils/ → Helper functions for data processing and formatting
├── assets/ → Icons, text, and static content
└── requirements.txt → Dependency list
```
## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ChaitanyaGore28/Smart-Driver-Recommendation.git
cd Smart-Driver-Recommendation
pip install -r requirements.txt
streamlit run app.py
```
## 4️⃣ Access the app

Open your browser and visit:
👉 http://localhost:8501
