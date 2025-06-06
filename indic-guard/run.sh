#!/bin/bash

echo "SFT run and model merging script"

echo "Enter the base model ID (e.g., google/gemma-3-1b-it):"
read MODEL_ID

echo "Enter your Hugging Face username:"
read HF_USERNAME

echo "Push to Hugging Face Hub? (y/n):"
read PUSH_CHOICE

if [[ $PUSH_CHOICE == "y" || $PUSH_CHOICE == "Y" ]]; then
    PUSH_TO_HUB="True"
else
    PUSH_TO_HUB="False"
fi

echo "Starting SFT run with model: $MODEL_ID"

# Run SFT with model ID as argument
python3 sft.py --model_id "$MODEL_ID"

if [ $? -eq 0 ]; then
    echo "SFT run completed successfully!"
    
    echo "Starting adapter merging"
    
    # Run adapter merging with arguments
    python3 merge_adapters.py --base_model_id "$MODEL_ID" --hub_username "$HF_USERNAME" --push_to_hub "$PUSH_TO_HUB"
    
    if [ $? -eq 0 ]; then
        echo "Adapter merging completed successfully"
    else
        echo "Adapter merging failed!"
        exit 1
    fi
    
else
    echo "SFT Training failed!"
    exit 1
fi