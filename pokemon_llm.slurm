#!/bin/bash
#SBATCH --job-name=pokemon_llm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --account=cs2731_2025f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-user=kes298@pitt.edu
#SBATCH --mail-type=END,FAIL

cd /ihome/cs2731_2025f/kes298/pokemon

# Initialize pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate pokemon-llm

# Hugging Face authentication and caching
export HUGGINGFACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)

python pokemon_llm/pokemon_llm.py