## Overview

This project implements an Advanced RAG (Retrieval-Augmented Generation) Pipeline that enables intelligent question-answering over PDF documents. The system combines state-of-the-art embedding models, efficient vector search, cross-encoder reranking, and multiple LLM providers to deliver accurate, citation-backed responses.

## Creat .env file and enter basic details as follows


LLM_PROVIDER=azure  #Options: "azure", "gemini", "openai"
  

AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_DEPLOYMENT_NAME= 
   
GEMINI_API_KEY=
GEMINI_MODEL_NAME=


## Install pip install -r requirements.txt

## Quick Start

### Option 1: Command Line Interface (CLI)
python main.py (or) python main_converstaion.py

### Option 2: FastAPI Server  
python src/api/server.py