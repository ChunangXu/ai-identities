# Better if you run these in terminal, one follwed by the other
# This script is important, installs ollama and sets up the path
# One time setup, you don't need to run this everytime
# SHOULD BE RUN FROM $SCRATCH/ai-identities DIRECTORY
wget https://github.com/ollama/ollama/releases/download/v0.5.11/ollama-linux-amd64.tgz -O ./ollama-linux-amd64.tgz
tar -xvzf ollama-linux-amd64.tgz
mkdir ollama
mv bin ollama
mv lib ollama
export PATH="$PATH:$SCRATCH/ai-identities/ollama/bin"