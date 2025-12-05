import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from src.utils import token2json
from src.config import Config
import argparse
import re

def predict(image_path, model_path):
    print(f"Loading model from {model_path}...")
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    device = Config.DEVICE if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load Image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare Input
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Prepare Decoder Input
    task_prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    decoder_input_ids = decoder_input_ids.to(device)
    
    # Generate
    print("Generating...")
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=Config.MAX_LENGTH,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # Post-process
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    
    return token2json(sequence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to receipt image")
    parser.add_argument("--model", type=str, default=f"{Config.OUTPUT_DIR}/final_model", help="Path to trained model")
    args = parser.parse_args()
    
    result = predict(args.image, args.model)
    print("\nPredicted JSON:")
    print(result)
