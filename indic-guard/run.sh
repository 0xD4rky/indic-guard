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

# updates sft.py with the new model id
sed "s|\"model_id\" : \".*\"|\"model_id\" : \"$MODEL_ID\"|g" sft.py > sft_temp.py

python3 sft_temp.py

if [ $? -eq 0 ]; then
    echo "SFT run completed successfully!"
    
    echo "Starting adapter merging"
    
    # Update merge_adapters.py with user parameters
    sed -e "s|base_model_id=\".*\"|base_model_id=\"$MODEL_ID\"|g" \
        -e "s|hub_username=\".*\"|hub_username=\"$HF_USERNAME\"|g" \
        -e "s|push_to_hub=.*|push_to_hub=$PUSH_TO_HUB|g" \
        merge_adapters.py > merge_temp.py
    
    # Run adapter merging
    python3 merge_temp.py
    
    if [ $? -eq 0 ]; then
        echo "Adapter merging completed successfully"
        # Clean up temporary files
        rm sft_temp.py merge_temp.py
    else
        echo "Adapter merging failed!"
        rm sft_temp.py merge_temp.py
        exit 1
    fi
    
else
    echo "SFT Training failed!"
    rm sft_temp.py
    exit 1
fi