# Multi-Agent Task Automation System

## Overview
An AI-powered task automation application built with Microsoft AutoGen, Streamlit, and Ollama. It uses multiple AI agents to plan tasks, generate responses, and review outputs.

## Features
- Planner, Writer, and Reviewer agents
- Follow-up questions and chat history
- SQLite database for conversation storage
- Document question answering
- PDF export
- Local AI inference using Ollama and Qwen 2.5

## Technologies
- Python
- Microsoft AutoGen
- Streamlit
- Ollama
- SQLite
- ReportLab
- PyMuPDF, pypdf, python-docx, and Tesseract OCR

## Requirements
- Python
- Ollama
- The `qwen2.5:1.5b` model
- Tesseract OCR, if using OCR features

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the model:

```bash
ollama pull qwen2.5:1.5b
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Limitations
- AI-generated answers may contain errors.
- Response quality depends on the selected model.
- Ollama must be accessible to the application.
- OCR requires a working Tesseract installation.

## Project Purpose
This project demonstrates multi-agent task processing, document question answering, conversation history, and PDF generation for a final-year engineering project.