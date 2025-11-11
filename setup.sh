#!/bin/bash

# RAG Poisoning Research - Setup Script
# This script sets up the environment for the Hidden Parrot attack demonstration

set -e

# Parse command line arguments
NO_LOCAL=false
for arg in "$@"; do
    case $arg in
        --no-local)
            NO_LOCAL=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

echo "🦜 Setting up RAG Poisoning Research Environment"
echo "================================================"

if [ "$NO_LOCAL" = true ]; then
    echo "🔧 Running in --no-local mode (skipping LLM download)"
fi

# Look for environment configuration files
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env file..."
    source .env
else
    echo "📄 No environment file found, will create defaults..."
fi

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv is installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv --python=3.11
source .venv/bin/activate
# Upgrade pip
echo "⬆️  Upgrading pip..."
uv pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
uv pip install -r requirements.txt

# Define default values for environment variables if not set
VECTOR_DB_PATH=${VECTOR_DB_PATH:-"./data/chroma_db"}
SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME:-"./models/embedding"}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"./models/embedding"}
LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-"./models/llm/llama-2-7b-chat.Q4_K_M.gguf"}
LOG_FILE=${LOG_FILE:-"./logs/rag_demo.log"}

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p "$VECTOR_DB_PATH"
mkdir -p "$SENTENCE_TRANSFORMERS_HOME"
mkdir -p "$(dirname "$LLAMA_MODEL_PATH")"
mkdir -p "$(dirname "$LOG_FILE")"

# Set local model cache to project directory for self-sustainability
export SENTENCE_TRANSFORMERS_HOME="$SENTENCE_TRANSFORMERS_HOME"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"

# Define embedding model name from environment or default
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"sentence-transformers/all-MiniLM-L6-v2"}

# Check and download embedding model locally to project if needed
echo "� Checking for embedding model: $EMBEDDING_MODEL..."
python3 -c "
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pathlib
import sys

# Set environment variables
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.environ.get('SENTENCE_TRANSFORMERS_HOME', './models/embedding')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', './models/embedding')

# Check if model files exist
model_name = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
# Extract the model name for path creation
model_parts = model_name.split('/')
if len(model_parts) > 1:
    model_path_name = f'models--{model_parts[0]}--{model_parts[1]}'
    model_path = pathlib.Path(os.environ.get('SENTENCE_TRANSFORMERS_HOME', './models/embedding') + '/' + model_path_name)
else:
    print('⚠️  Invalid model name format, using default path')
    model_path = pathlib.Path('./models/embedding/models--sentence-transformers--all-MiniLM-L6-v2')

if model_path.exists() and list(model_path.glob('*')):
    print(f'✅ Embedding model {model_name} already exists locally')
else:
    print(f'🔄 Downloading {model_name} to {os.environ.get(\"SENTENCE_TRANSFORMERS_HOME\", \"./models/embedding\")}...')
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        print(f'✅ Embedding model {model_name} downloaded locally')
    except Exception as e:
        print(f'❌ Error downloading model: {e}')
        sys.exit(1)
"

# Download a local LLM for complete self-sustainability
if [ "$NO_LOCAL" = false ]; then
    echo "🤖 Downloading local LLM for inference..."
    if [ ! -f "$LLAMA_MODEL_PATH" ]; then
        echo "Downloading Llama-2-7B-Chat (Q4_K_M quantized, ~4GB) to $LLAMA_MODEL_PATH..."
        LLAMA_DOWNLOAD_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        if command -v wget >/dev/null 2>&1; then
            wget -O "$LLAMA_MODEL_PATH" "$LLAMA_DOWNLOAD_URL"
        elif command -v curl >/dev/null 2>&1; then
            curl -L -o "$LLAMA_MODEL_PATH" "$LLAMA_DOWNLOAD_URL"
        else
            echo "❌ Neither wget nor curl is available. Please install one of them to download the LLM."
            exit 1
        fi
        echo "✅ Local LLM downloaded"
    else
        echo "✅ Local LLM already exists at $LLAMA_MODEL_PATH"
    fi
else
    echo "⏭️  Skipping local LLM download (--no-local mode)"
fi

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the test: python3 test_setup.py (--no-local for remote inference only)"
if [ "$NO_LOCAL" = false ]; then
    echo "3. Run the demo with one of the following options:"
    echo "   - Local LLM: python3 src/rag_poisoning_demo.py"
    echo "   - Ollama: python3 src/rag_poisoning_demo.py --infer ollama"
    echo "   - DeepSeek: python3 src/rag_poisoning_demo.py --infer deepseek"
    echo ""
    echo "📝 Notes:"
    echo "   - For Ollama: Ensure Ollama is running and configured in .env"
    echo "   - For DeepSeek: Ensure API key is set in .keys file"
    echo "   - All embedding models are downloaded locally for self-sustainability"
    echo "   - Embedding model: $SENTENCE_TRANSFORMERS_HOME"
    echo "   - Local LLM model: $(dirname "$LLAMA_MODEL_PATH")"
    echo "   - Configuration: .env (environment variables loaded from this file)"
else
    echo "3. Run the demo with one of the following options:"
    echo "   - Ollama: python3 src/rag_poisoning_demo.py --infer ollama"
    echo "   - DeepSeek: python3 src/rag_poisoning_demo.py --infer deepseek"
    echo ""
    echo "📝 Notes:"
    echo "   - Local LLM was skipped (--no-local mode)"
    echo "   - For Ollama: Ensure Ollama is running and configured in .env"
    echo "   - For DeepSeek: Ensure API key is set in .keys file"
    echo "   - All embedding models are downloaded locally for self-sustainability"
    echo "   - Embedding model: $SENTENCE_TRANSFORMERS_HOME"
    echo "   - Configuration: .env (environment variables loaded from this file)"
fi
echo ""
echo "🏴‍☠️ Ready to demonstrate the Hidden Parrot attack!"
