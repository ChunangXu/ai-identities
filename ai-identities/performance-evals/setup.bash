pip install -r requirements.txt
ollama serve &

ollama_models=(
    llama3.2:3b
    deepseek-r1:1.5b
    qwen:1.8b
    gemma2:2b
    phi3:3.8b
    mistral
)

for item in "${ollama_models[@]}"; do
    ollama pull $item
done
# IF pip-installing using this dont forget to activate your venv first before all this
# also once you pull the models, its gonna be populated in your .ollama directory in your home/login
# so you must copy .ollama/models to your scratch/ directory
# these might not work within the script, but make sure to ren below 2 commands after all models have been pulled
# mkdir -p $SCRATCH/ollama_home
# cp -r ~/.ollama/models $SCRATCH/ollama_home/

