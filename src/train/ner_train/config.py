import torch
import logging
import os

# --- General ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Removed device print statement, will be logged elsewhere if needed
SEED = 42

# --- Dataset ---
DATASET_NAME = "conll2003"
SPACY_MODEL = "en_core_web_trf"
MAX_SEQ_LEN = 128 # Max sequence length after spaCy/transformer tokenization. Adjust as needed.
MAX_WORD_LEN = 20 # Max word length for Char CNN

# --- Model ---
# Embeddings
SPACY_EMBEDDING_DIM = 768 # Depends on 'en_core_web_trf' model (RoBERTa-base)
POS_EMBEDDING_DIM = 50
DEP_EMBEDDING_DIM = 50
CHAR_EMBEDDING_DIM = 30
UNFREEZE_TRANSFORMER_LAYERS = 1 # Number of transformer layers to unfreeze (0 to freeze all)

# Char CNN
CHAR_CNN_FILTERS = 50 # Filters per kernel size
CHAR_CNN_KERNELS = [3, 4, 5] # Kernel sizes

# BiLSTM
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.33 # Recurrent dropout

# Attention (simple dot-product self-attention) - Currently unused
# ATTENTION_DIM = LSTM_HIDDEN_DIM * 2 # BiLSTM output dimension

# General Dropout (applied before the final layer)
DROPOUT_RATE = 0.5

# --- Training ---
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 30
LEARNING_RATE = 1e-4 # Initial learning rate for AdamW
TRANSFORMER_LEARNING_RATE = 2e-5 # Lower learning rate for transformer layers
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_VAL = 1.0
EARLY_STOPPING_PATIENCE = 3
USE_AMP = True # Use Mixed Precision Training

# --- Paths ---
# Define project structure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "models", "ner_model")
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "ner_model_best.pt")
CACHE_DIR = os.path.join(DATA_DIR, "ner_cache")  # Changed to src/data/ner_cache
LOG_DIR = os.path.join(SCRIPT_DIR, "runs")  # Directory for TensorBoard logs

# --- Mappings (will be filled in data_utils) ---
pos_vocab = {}
dep_vocab = {}
ner_vocab = {}
char_vocab = {}

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def get_logger(name):
    """Gets a logger instance."""
    return logging.getLogger(name)
