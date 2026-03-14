# PHI De-identification System

A production-ready AI system for detecting and anonymizing Protected Health Information (PHI) from clinical reports using a **BiLSTM + CRF** sequence labeling model.

## Architecture

```
Embedding Layer → BiLSTM (2-layer, bidirectional) → Dropout → Linear → CRF (Viterbi)
```

## Detected Entities (BIO tagging)

| Entity | Description |
|--------|-------------|
| NAME | Patient name |
| DOCTOR | Physician/doctor name |
| HOSPITAL | Hospital or clinic name |
| LOCATION | City or address |
| DATE | Any date |
| PHONE | Phone number |
| EMAIL | Email address |
| ID | Patient ID / MRN |

## Project Structure

```
phi_deidentification/
├── config/config.py                  # Hyperparameters and paths
├── data_generation/
│   ├── generate_synthetic_dataset.py # 55,000+ synthetic BIO sentences
│   └── generate_dataset_from_reports.py # PDF → BIO pipeline + split
├── data/
│   ├── train.txt / dev.txt / test.txt  # CoNLL-format datasets
│   └── vocab.txt / tags.txt
├── dataset/
│   ├── vocab_builder.py              # Vocabulary construction
│   └── dataset_loader.py             # PyTorch Dataset + DataLoader
├── models/
│   └── bilstm_crf_model.py           # BiLSTM + CRF model (PyTorch)
├── training/
│   ├── train.py                      # Training loop
│   └── evaluate.py                   # Entity-level evaluation
├── utils/
│   ├── tokenizer.py                  # Text tokenization
│   ├── metrics.py                    # NER span-level metrics
│   ├── pdf_parser.py                 # PDF text extraction
│   └── phi_extractor.py              # Regex PHI fallback
├── inference/
│   ├── predict.py                    # PHIPredictor class
│   └── redact_text.py                # PHIRedactor (text + PDF)
├── ui/
│   └── streamlit_app.py              # Streamlit web interface
├── reports/                          # Place PDF reports here
├── checkpoints/                      # Saved model checkpoints
├── requirements.txt
├── setup.sh
└── run_pipeline.py                   # End-to-end pipeline runner
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset (55,000+ synthetic + real PDF sentences)
python data_generation/generate_dataset_from_reports.py

# 3. Train model
python training/train.py

# 4. Evaluate on test set
python training/evaluate.py

# 5. Launch Streamlit UI
streamlit run ui/streamlit_app.py

# Or run everything at once:
python run_pipeline.py --ui
```

## PDF Reports

Place `.pdf` medical reports in the `reports/` folder before running dataset generation. The pipeline will:
1. Extract text using `pdfminer.six`
2. Detect PHI using regex/field labels
3. Convert to BIO-tagged sentences
4. Merge with synthetic data

## Hyperparameters (config/config.py)

| Parameter | Value |
|-----------|-------|
| embedding_dim | 128 |
| hidden_dim | 256 |
| dropout | 0.3 |
| batch_size | 32 |
| learning_rate | 0.001 |
| epochs | 10 |
| max_sequence_length | 50 |

## Example

**Input:**
```
Patient Ravi Kumar visited Apollo Hospital in Bangalore on 12 March 2024.
Dr. Priya Sharma prescribed Metformin 500mg. Contact: ravi@gmail.com
```

**Output:**
```
Patient [NAME] visited [HOSPITAL] in [LOCATION] on [DATE].
Dr. [DOCTOR] prescribed Metformin 500mg. Contact: [EMAIL]
```

## Model Details

- **Loss:** CRF negative log-likelihood
- **Optimizer:** Adam with gradient clipping (clip=5.0)
- **Decoding:** Viterbi algorithm
- **Evaluation:** Entity-level span F1 (not token accuracy)
- **Dataset:** 70% train / 15% dev / 15% test split
