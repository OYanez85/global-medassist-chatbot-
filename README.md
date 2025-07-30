# global-medassist-chatbot-

## Global MedAssist

A multi-agent simulation for medical assistance built with Gradio, LangGraph, and LangChain. This app simulates a conversation between a patient and various medical-assessment agents, generates speech audio, stitches together a full playback, and exports a PDF summary and ZIP bundle.

## Features

Multi-Patient Support: Predefined profiles for Anne, Liam, and Priya.

LangGraph Workflow: Orchestrates a sequence of agent nodes with customizable edges.

RAG Knowledge Bases: Retrieval-Augmented Generation for hospital and policy info.

SSML & gTTS TTS: Google Cloud Text-to-Speech or gTTS fallback with emotion-based prosody.

Audio Stitching: Combines per-agent clips into a single MP3.

PDF Export: Generates a conversation log PDF via ReportLab.

ZIP Package: Bundles audio clips, log, PDF, and full-audio.

Hugging Face Integration: Gradio app ready to deploy as a Space.

## File Structure

app.py               # Main application script
tts_audio/           # Auto-generated audio clips
rag_docs/            # Mocked knowledge-base text files
sounds/ringtone.mp3  # Optional ringtone audio
case_log.txt         # Generated conversation log
case_export.zip      # Exported ZIP bundle
README.md            # This file
requirements.txt     # Python dependencies
apt-packages.txt     # System-level dependencies (e.g., ffmpeg)

# Requirements

## Python Packages (requirements.txt)

gradio==3.42.1
pydub==0.25.1
gTTS==2.3.0
google-cloud-texttospeech==2.24.0
langgraph==0.1.5
langchain==0.0.350
langchain-openai==0.0.13
openai>=0.27.0
faiss-cpu==1.7.4
reportlab==3.8.0

System Packages (apt-packages.txt)

ffmpeg

Note: pydub requires the ffmpeg binary on the system path.

# Environment Variables

OPENAI_API_KEY – your OpenAI API key (required for embeddings and chat).

GOOGLE_APPLICATION_CREDENTIALS – path to your Google Cloud service account JSON (optional, for Cloud TTS).

Set these in your Space settings (HF Secrets) or your local environment before launching.

# Installation & Local Run

Clone this repo:

git clone <repo-url>
cd <repo-directory>

# Install Python dependencies:

pip install -r requirements.txt

# Install system packages (Ubuntu/Debian):

sudo apt-get update && sudo apt-get install ffmpeg -y

# Export environment variables:

export OPENAI_API_KEY="your-key"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"

# Run the app:

python app.py

Open http://localhost:7860 in your browser.

Deployment to Hugging Face Spaces

Create a new Space on Hugging Face with Gradio as the SDK.

Add app.py, requirements.txt, and apt-packages.txt to the repo.

In Settings & secrets, add OPENAI_API_KEY (and optionally GOOGLE_APPLICATION_CREDENTIALS).

Push your code; the Space will build and launch automatically.

# Usage

Select a patient from the dropdown (Anne, Liam, or Priya).

View the live conversation log in the UI.

Download the ZIP bundle containing:

Individual agent audio clips

Full conversation MP3

Conversation log PDF

Play back the full conversation audio directly in the browser.

# Customization

Patients: Extend get_patient_by_name and patient_scripts for new profiles.

Graph Workflow: Modify edges in build_workflow() to adjust agent order.

TTS: Toggle between Google Cloud TTS and gTTS via the tts_client variable.

License

MIT
