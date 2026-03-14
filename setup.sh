#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Creating directories..."
mkdir -p reports checkpoints data
echo "Setup complete."
echo "Steps: 1) python data_generation/generate_dataset_from_reports.py"
echo "       2) python training/train.py"
echo "       3) streamlit run ui/streamlit_app.py"
