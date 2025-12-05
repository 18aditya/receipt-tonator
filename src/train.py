import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
from src.config import Config
from src.dataset import ReceiptDataset
import os

def train():
    print(f"Loading processor and model: {Config.MODEL_NAME}")
    processor = DonutProcessor.from_pretrained(Config.MODEL_NAME)
    # Resize image processing to save memory
    processor.image_processor.size = {"height": Config.IMAGE_SIZE[0], "width": Config.IMAGE_SIZE[1]}
    
    model = VisionEncoderDecoderModel.from_pretrained(Config.MODEL_NAME)

    # Prepare Dataset
    print(f"Loading dataset: {Config.DATASET_NAME}")
    # We load the dataset first to update the tokenizer with special tokens
    train_dataset = ReceiptDataset(
        dataset_name_or_path=Config.DATASET_NAME,
        max_length=Config.MAX_LENGTH,
        processor=processor,
        split="train"
    )
    
    # IMPORTANT: Resize model embeddings to match new tokenizer size
    print(f"Resizing model embeddings to {len(processor.tokenizer)}")
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Configure model
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_receipt>"])[0]
    
    # Move to device
    print(f"Moving model to {Config.DEVICE}")
    model.to(Config.DEVICE)
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        max_steps=getattr(Config, "MAX_STEPS", -1), # Use max_steps if defined
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no", # We don't have a val split right now
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none", # Disable wandb/mlflow for now
        fp16=False, # MPS doesn't support fp16 well yet
        dataloader_pin_memory=False, # Sometimes causes issues on MPS
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer, # Use processing_class instead of tokenizer
        data_collator=default_data_collator,
    )
    
    # Start Training
    print("Starting training...")
    trainer.train()
    
    # Save Final Model
    print(f"Saving model to {Config.OUTPUT_DIR}/final_model")
    trainer.save_model(os.path.join(Config.OUTPUT_DIR, "final_model"))
    processor.save_pretrained(os.path.join(Config.OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    train()
