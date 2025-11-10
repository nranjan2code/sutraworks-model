#!/bin/bash

# Model Download Helper Script
# Downloads real AI models for testing the SutraWorks system

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║        🌐 SutraWorks Model Downloader 🌐           ║"
echo "║      Download real models for end-to-end testing     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo

# Function to download with git-lfs if available
download_model() {
    local repo=$1
    local model_name=$2
    local cache_dir="$HOME/.cache/sutraworks/models/$repo"
    
    echo "📥 Downloading $model_name..."
    echo "Repository: $repo"
    echo "Cache directory: $cache_dir"
    echo
    
    if command -v git-lfs >/dev/null 2>&1; then
        echo "✅ Git LFS detected - using for efficient download"
        mkdir -p "$cache_dir"
        
        if [ ! -d "$cache_dir/.git" ]; then
            git clone https://huggingface.co/$repo "$cache_dir"
        else
            echo "📁 Repository already exists, updating..."
            cd "$cache_dir"
            git pull
            cd - >/dev/null
        fi
        
        echo "✅ Downloaded $model_name to $cache_dir"
    else
        echo "⚠️  Git LFS not found - using Rust downloader"
        echo "Install git-lfs for faster downloads: brew install git-lfs"
        echo "Then run: cargo run --example real_world_test --release"
    fi
    echo
}

# Check available disk space
available_space=$(df -h . | tail -1 | awk '{print $4}')
echo "💾 Available disk space: $available_space"
echo

# Ask user which models to download
echo "📋 Available models for download:"
echo "1. RWKV-4 169M (~700MB) - Lightweight RNN language model"
echo "2. RWKV-4 430M (~1.7GB) - Medium RNN language model"  
echo "3. Mamba 130M (~500MB) - State-space model"
echo "4. Mamba 370M (~1.5GB) - Larger state-space model"
echo "5. All small models (169M + 130M = ~1.2GB)"
echo "6. All models (~4GB total)"
echo "0. Skip download (test with synthetic data only)"
echo

read -p "Select models to download (0-6): " choice

case $choice in
    1)
        download_model "BlinkDL/rwkv-4-pile-169m" "RWKV-4 169M"
        ;;
    2)
        download_model "BlinkDL/rwkv-4-pile-430m" "RWKV-4 430M"
        ;;
    3)
        download_model "state-spaces/mamba-130m" "Mamba 130M"
        ;;
    4)
        download_model "state-spaces/mamba-370m" "Mamba 370M"
        ;;
    5)
        download_model "BlinkDL/rwkv-4-pile-169m" "RWKV-4 169M"
        download_model "state-spaces/mamba-130m" "Mamba 130M"
        ;;
    6)
        download_model "BlinkDL/rwkv-4-pile-169m" "RWKV-4 169M"
        download_model "BlinkDL/rwkv-4-pile-430m" "RWKV-4 430M"
        download_model "state-spaces/mamba-130m" "Mamba 130M"
        download_model "state-spaces/mamba-370m" "Mamba 370M"
        ;;
    0)
        echo "⏭️  Skipping downloads - will use synthetic data for testing"
        echo
        ;;
    *)
        echo "❌ Invalid selection"
        exit 1
        ;;
esac

echo "🎯 Next steps:"
echo "1. Run manual tests (no downloads needed):"
echo "   cargo run --example manual_test --release"
echo
echo "2. Run real-world tests with downloaded models:"
echo "   cargo run --example real_world_test --release"
echo
echo "3. Run specific model examples:"
echo "   cargo run --example rwkv_inference --release"
echo "   cargo run --example mamba_inference --release"
echo
echo "4. Run full end-to-end pipeline:"
echo "   cargo run --example end_to_end --release"
echo

# Check if models were downloaded
cache_dir="$HOME/.cache/sutraworks/models"
if [ -d "$cache_dir" ]; then
    echo "📁 Downloaded models:"
    find "$cache_dir" -name "*.safetensors" -o -name "*.bin" | while read -r file; do
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  • $(basename "$(dirname "$file")"): $size"
    done
    echo
fi

echo "✅ Setup complete! Ready for real-world AI testing."