# Donut OCR for Receipts

This project implements an OCR (Optical Character Recognition) and Information Extraction system for receipts using the **Donut** (Document Understanding Transformer) model. It is fine-tuned on the `mychen76/ds_receipts_v2_train` dataset to directly convert receipt images into structured JSON data without an intermediate OCR step (OCR-free).

## Features
- **End-to-End**: Maps images directly to JSON structure.
- **Apple Silicon Optimized**: Configured to use Metal Performance Shaders (`mps`) for GPU acceleration on Mac M1/M2/M3 chips.
- **Custom Tokenization**: Dynamically handles special tokens for receipt fields (e.g., `<s_total>`, `<s_date>`).

## Installation

1. **Clone the repository** (if applicable) or navigate to the project folder:
   ```bash
   cd ocr_project
   ```

2. **Set up the virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project requires `torch`, `transformers`, `datasets`, and `sentencepiece`.*

## Usage

### 1. Training
To fine-tune the model on the receipts dataset, run:

```bash
python3 -m src.train
```

**Configuration**:
You can adjust training parameters in `src/config.py`:
- `BATCH_SIZE`: Defaults to 1 to fit in memory on standard Macs.
- `IMAGE_SIZE`: Set to `(1280, 960)` for performance.
- `NUM_EPOCHS`: Defaults to 3.

**Output**:
The trained model and processor will be saved in the `result/final_model` directory.

### 2. Inference
To extract data from a new receipt image:

```bash
python3 -m src.inference --image path/to/receipt.jpg
```

**Arguments**:
- `--image`: Path to the input image file (JPG/PNG).
- `--model`: Path to the trained model directory (defaults to `result/final_model`).

**Example Output**:
```json
{
  "store_name": "SAFEWAY",
  "date": "12/22/17",
  "total": "$89.09",
  "line_items": [
    {
      "item_name": "Apple",
      "item_value": "1.99"
    }
  ]
}
```

## Project Structure

```text
ocr_project/
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py          # Global configuration (paths, hyperparameters)
│   ├── dataset.py         # Data loading and preprocessing logic
│   ├── train.py           # Main training script (Teacher Forcing)
│   ├── inference.py       # Prediction script for new images
│   └── utils.py           # Helpers for JSON <-> Token conversion
└── result/                # Directory where trained models are saved
```

## Troubleshooting

- **Out of Memory (OOM)**: If you encounter memory errors, try reducing `IMAGE_SIZE` or `BATCH_SIZE` in `src/config.py`.
- **MPS Issues**: If `mps` acceleration causes crashes, you can switch the `DEVICE` to `cpu` in `src/config.py`, though training will be significantly slower.
