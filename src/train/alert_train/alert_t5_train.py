"""
This script trains a T5 model for automatic alert generation using NER and Sentiment Analysis results.
It optimizes model storage by saving only essential components for inference.
"""
import os
import json
import random
import numpy as np
import torch
import pandas as pd
import shutil
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.cache_utils import EncoderDecoderCache
import evaluate
from tqdm import tqdm
import warnings
import types
from functools import wraps
warnings.filterwarnings("ignore")


# Configuration parameters
MODEL_NAME = "google/flan-t5-small"
INCLUDE_TEXT = False
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 32
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 150
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EVAL_STEPS = 20
SAVE_STEPS = 100

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "data", "alert_data.txt")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "alert_generator_model")
# Removed INFERENCE_MODEL_DIR to use a single folder

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_from_jsonl(file_path):
    """
    Load and parse dataset from a JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON examples with non-empty NER results
    """
    data = []
    skipped_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                if 'ner_result' in example and example['ner_result']:
                    data.append(example)
                else:
                    skipped_count += 1
            except json.JSONDecodeError:
                print(f"Error parsing line {line_number}: {line}")
                continue
    
    print(f"Total examples loaded: {len(data)}")
    print(f"Examples skipped due to empty ner_result: {skipped_count}")
    
    return data


def prepare_datasets(data):
    """
    Create structured datasets and split them into train, validation, and test sets
    
    Args:
        data: List of parsed examples
        
    Returns:
        Dictionary containing train, validation, and test datasets
    """
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
    return {
        'train': train_test['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    }


def format_input(example):
    """
    Format the input for the model
    
    Args:
        example: Dictionary containing NER results, sentiment, and text
        
    Returns:
        Formatted input string
    """
    ner_parts = [f"{entity}:{label}" for entity, label in example['ner_result'].items() if label and entity]
    ner_text = ", ".join(ner_parts)
    return f"NER: {ner_text} | Sentiment: {example['sentiment']} | Text: {example['text']}" if INCLUDE_TEXT else f"NER: {ner_text} | Sentiment: {example['sentiment']}"


def preprocess_function(examples, tokenizer):
    """
    Preprocess examples for the model
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer for the model
        
    Returns:
        Processed inputs with tokenized text and labels
    """
    inputs = [format_input({
        'ner_result': examples['ner_result'][i],
        'sentiment': examples['sentiment'][i],
        'text': examples['text'][i]
    }) for i in range(len(examples['ner_result']))]
    
    targets = examples['alert']
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """
    Compute ROUGE metrics for model evaluation
    
    Args:
        eval_preds: Model predictions and labels
        tokenizer: Tokenizer for decoding predictions
        
    Returns:
        Dictionary of ROUGE metrics
    """
    rouge = evaluate.load("rouge")
    preds, labels = eval_preds
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}


def train_model(datasets, tokenizer, model, output_dir):
    """
    Train the model with the given datasets
    
    Args:
        datasets: Dictionary containing tokenized datasets
        tokenizer: Tokenizer for the model
        model: Model to train
        output_dir: Directory to save the model
        
    Returns:
        Trained model and tokenizer
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        fp16=False,
        gradient_accumulation_steps=4,
    )
    
    # Initialize trainer with a metric computation function that includes tokenizer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer


def optimize_model_for_inference(model_dir):
    """
    Optimize the model for inference, removing training artifacts
    and saving only essential components in the same folder.
    
    Args:
        model_dir: Model directory
    """
    print(f"Optimizing model for inference in {model_dir}")
    
    # Create a temporary directory for optimization
    temp_dir = os.path.join(model_dir, "temp_optimization")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Save model in optimized format in temporary directory
        print(f"Saving model in temporary directory: {temp_dir}")
        model.save_pretrained(
            temp_dir,
            is_main_process=True,
            save_function=torch.save,
            push_to_hub=False,
            safe_serialization=False  # Disable safetensors to avoid file access issues
        )
        
        # Save tokenizer in temporary directory
        tokenizer.save_pretrained(temp_dir)
        
        # Remove training artifacts from original directory
        training_artifacts = [
            "optimizer.pt", 
            "rng_state.pth", 
            "scheduler.pt", 
            "trainer_state.json", 
            "training_args.bin"
        ]
        
        for file in training_artifacts:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"Removing training artifact: {file_path}")
                os.remove(file_path)
        
        # Remove checkpoints from original directory
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(model_dir, item)
                if os.path.isdir(checkpoint_dir):
                    print(f"Removing checkpoint: {checkpoint_dir}")
                    shutil.rmtree(checkpoint_dir)
        
        # Copy files from temporary directory to original directory
        print(f"Copying optimized files from {temp_dir} to {model_dir}")
        for item in os.listdir(temp_dir):
            source = os.path.join(temp_dir, item)
            destination = os.path.join(model_dir, item)
            
            if os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        
        print("Model optimized and checkpoints successfully removed")
    except Exception as e:
        print(f"Error during model optimization: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    # Reload the optimized model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer


def run_inference(datasets, model, tokenizer, device, dataset_type="test", num_examples=7):
    """
    Run inference on random examples from the dataset
    
    Args:
        datasets: Dictionary containing datasets
        model: Model for inference
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        dataset_type: Type of dataset to use (train or test)
        num_examples: Number of examples to run inference on
    """
    print(f"\n=== INFERENCE ON {num_examples} RANDOM {dataset_type.upper()} EXAMPLES ===")
    sample_indices = random.sample(range(len(datasets[dataset_type])), num_examples)
    model.eval()

    for i, idx in enumerate(sample_indices):
        example = datasets[dataset_type][idx]
        input_text = format_input(example)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=MAX_TARGET_LENGTH, 
                num_beams=4, 
                early_stopping=True,
                use_cache=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference = example["alert"]
        
        print(f"\nExample {i+1}:")
        print(f"  Input: {input_text}")
        print(f"  Reference: {reference}")
        print(f"  Prediction: {prediction}")
        print("-" * 50)


def main():
    """Main function to run the training pipeline"""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare dataset
    print(f"Loading data from {DATA_PATH}")
    raw_data = load_dataset_from_jsonl(DATA_PATH)
    datasets = prepare_datasets(raw_data)
    
    # Load model and tokenizer
    print(f"Loading model {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    
    # Preprocess datasets
    print("Preprocessing datasets")
    tokenized_datasets = {
        split: dataset.map(
            lambda examples: preprocess_function(examples, tokenizer), 
            batched=True, 
            remove_columns=datasets[split].column_names
        )
        for split, dataset in datasets.items()
    }
    
    # Train model
    model, tokenizer = train_model(tokenized_datasets, tokenizer, model, MODEL_DIR)
    
    # Optimize model for inference
    inference_model, inference_tokenizer = optimize_model_for_inference(MODEL_DIR)
    
    # Run inference with the optimized model
    print("\nRunning inference with optimized model...")
    inference_model = inference_model.to(device)
    run_inference(datasets, inference_model, inference_tokenizer, device, "train", 3)
    run_inference(datasets, inference_model, inference_tokenizer, device, "test", 3)


def load_optimized_model_for_inference(model_dir=MODEL_DIR):
    """
    Load an optimized model for inference only
    
    Args:
        model_dir: Directory containing the optimized model
        
    Returns:
        Model and tokenizer ready for inference
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        device_map="auto",        # Automatically choose best device
        torch_dtype=torch.float16  # Use half precision for faster inference
    )
    model.eval()  # Set model to evaluation mode
    
    # Apply memory optimization
    if hasattr(model, "config"):
        model.config.use_cache = True  # Enable caching for faster inference
    
    return model, tokenizer


if __name__ == "__main__":
    main()









