
class Config:
    # Model & Data
    MODEL_NAME = "naver-clova-ix/donut-base"
    DATASET_NAME = "mychen76/ds_receipts_v2_train"
    
    # Image processing
    IMAGE_SIZE = (1280, 960)  # Reduced from (2560, 1920) to save memory
    ALIGN_LONG_AXIS = False
    IGNORE_ID = -100
    
    # Tokenization
    MAX_LENGTH = 768
    ADDED_TOKENS = [] # Will be populated dynamically or we can list common ones
    
    # Training
    BATCH_SIZE = 1 # Reduced from 2 to save memory
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 3
    # MAX_STEPS = 10 # Commented out for full training
    SEED = 42
    
    # Output
    OUTPUT_DIR = "result"
    
    # Hardware
    DEVICE = "mps" # Apple Silicon
