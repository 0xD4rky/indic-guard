# indic-guard

Post-training llms act as a guard against offensive prompts against Indian Languages

## Process to run the script

1. Login to your huggingface hub locally to access gated repos/models
```huggingface-cli login```
2. ```
   git clone https://github.com/0xD4rky/indic-guard.git
   cd indic-guard
   ```
3. ```pip install -r requirements.txt```
4. ```cd indic-guard```
5. ```chmod +x run.sh``` to give bash script the execution permissions
6. ```bash run.sh``` to run the script
7. Give the arguements (model which you wanna guard, your HF username and whether you wanna push this model to hub or not) when asked in the cli.
