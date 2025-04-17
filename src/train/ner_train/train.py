import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import time
import os
import datetime # For run names
import logging # Import logging

# Modificando los imports para usar rutas relativas al paquete
from src.train.ner_train import config
from src.train.ner_train import utils
# Import data_utils here, ensuring config is loaded first
from src.train.ner_train import data_utils
from src.train.ner_train.model import NERModel

logger = config.get_logger(__name__) # Get logger instance

def train_epoch(model, dataloader, optimizer, scaler, device):
    """Performs one training epoch."""
    model.train() # Set model to training mode
    total_loss = 0
    start_time = time.time()

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        # Move batch data to the target device
        # Use non_blocking=True for potentially faster transfer if pin_memory=True in DataLoader
        try:
            embeddings = batch['embeddings'].to(device, non_blocking=True)
            pos_ids = batch['pos_ids'].to(device, non_blocking=True)
            dep_ids = batch['dep_ids'].to(device, non_blocking=True)
            char_ids = batch['char_ids'].to(device, non_blocking=True)
            ner_ids = batch['ner_ids'].to(device, non_blocking=True) # Ground truth tags
            attention_mask = batch['attention_mask'].to(device, non_blocking=True) # Mask for CRF/padding
        except Exception as e:
            logger.error(f"Error moving batch to device: {e}", exc_info=True)
            continue # Skip batch if error occurs

        optimizer.zero_grad() # Reset gradients

        # Automatic Mixed Precision (AMP) context
        with autocast(enabled=config.USE_AMP):
            # Forward pass to calculate loss
            loss = model(
                embeddings=embeddings,
                pos_ids=pos_ids,
                dep_ids=dep_ids,
                char_ids=char_ids,
                attention_mask=attention_mask,
                tags=ner_ids # Provide true tags for loss calculation
            )

        # Check for NaN loss
        if torch.isnan(loss):
            logger.warning("NaN loss detected! Skipping batch.")
            continue

        # Backward pass with gradient scaling (for AMP)
        scaler.scale(loss).backward()

        # Unscale gradients before clipping (required by AMP)
        scaler.unscale_(optimizer)

        # Gradient Clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)

        # Optimizer step (updates model parameters)
        scaler.step(optimizer)

        # Update the scaler for the next iteration
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item()) # Update progress bar description


    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    epoch_time = time.time() - start_time
    logger.info(f"Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    return avg_loss

def evaluate_epoch(model, dataloader, device, id_to_ner, return_examples=False, num_examples=10):
    """Performs one evaluation epoch."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds_ids = [] # Store predicted tag IDs
    all_trues_ids = [] # Store true tag IDs
    example_data = [] # Store examples for visualization

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch data to the target device
            try:
                embeddings = batch['embeddings'].to(device, non_blocking=True)
                pos_ids = batch['pos_ids'].to(device, non_blocking=True)
                dep_ids = batch['dep_ids'].to(device, non_blocking=True)
                char_ids = batch['char_ids'].to(device, non_blocking=True)
                ner_ids = batch['ner_ids'].to(device, non_blocking=True) # Ground truth tags
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            except Exception as e:
                logger.error(f"Error moving batch to device during evaluation: {e}", exc_info=True)
                continue # Skip batch

            # Original text data remains on CPU
            true_labels_batch_str = batch['original_ner_tags'] # List[List[str]]
            original_tokens_batch = batch['original_tokens'] # List[List[str]]

            # Forward pass for loss (optional but useful) and predictions
            with autocast(enabled=config.USE_AMP):
                 # Calculate loss
                 loss = model(
                     embeddings=embeddings,
                     pos_ids=pos_ids,
                     dep_ids=dep_ids,
                     char_ids=char_ids,
                     attention_mask=attention_mask,
                     tags=ner_ids
                 )
                 # Get predictions (decoding)
                 predictions_ids = model(
                     embeddings=embeddings,
                     pos_ids=pos_ids,
                     dep_ids=dep_ids,
                     char_ids=char_ids,
                     attention_mask=attention_mask
                 ) # Returns List[List[int]]

            if loss is not None and not torch.isnan(loss):
                 total_loss += loss.item()

            # Store predictions and true labels (as IDs) for metric calculation
            # Move true IDs to CPU for processing
            ner_ids_cpu_list = ner_ids.cpu().numpy().tolist()

            for i in range(len(predictions_ids)): # Iterate through examples in the batch
                true_tags_str = true_labels_batch_str[i]
                original_tokens = original_tokens_batch[i]
                seq_len_true = len(true_tags_str) # Use original length before padding

                # Get predicted IDs for this example, truncated to original length
                pred_ids_seq = predictions_ids[i][:seq_len_true]
                # Get true IDs for this example, truncated and ignoring padding
                true_ids_seq_no_pad = [tid for tid in ner_ids_cpu_list[i][:seq_len_true]]

                # Append ID sequences for metric calculation
                all_preds_ids.append(pred_ids_seq)
                all_trues_ids.append(true_ids_seq_no_pad)

                # Store data for visualization examples if requested
                if return_examples and len(example_data) < num_examples:
                    # Convert predicted IDs to tags
                    pred_tags_str = [id_to_ner.get(pid, "<UNK>") for pid in pred_ids_seq]
                    # Ensure all lists have the same length for zipping
                    min_vis_len = min(len(original_tokens), len(true_tags_str), len(pred_tags_str))
                    example_data.append({
                        "tokens": original_tokens[:min_vis_len],
                        "true_tags": true_tags_str[:min_vis_len],
                        "pred_tags": pred_tags_str[:min_vis_len]
                    })

    # Calculate average loss and metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    metrics = utils.calculate_metrics(all_trues_ids, all_preds_ids, id_to_ner)

    logger.info(f"Eval Loss: {avg_loss:.4f}, F1 (macro): {metrics['f1']:.4f}")

    # Return results
    if return_examples:
        return avg_loss, metrics['f1'], metrics['report'], example_data
    else:
        return avg_loss, metrics['f1']


def main():
    """Main training and evaluation script."""
    # Set random seed for reproducibility
    utils.set_seed(config.SEED)
    logger.info(f"Using device: {config.DEVICE}")

    # --- TensorBoard Setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"NERModel_LR{config.LEARNING_RATE}_BS{config.BATCH_SIZE}_{timestamp}"
    log_dir = os.path.join(config.LOG_DIR, run_name)
    try:
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    except Exception as e:
        logger.error(f"Error initializing TensorBoard SummaryWriter: {e}", exc_info=True)
        return # Exit if TensorBoard setup fails

    # --- Load Data ---
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader = data_utils.get_dataloaders(config.BATCH_SIZE)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return # Exit if data loading fails

    # --- Initialize Model ---
    logger.info("Initializing model...")
    # Ensure vocabs are loaded (get_dataloaders should handle this via preprocess_data)
    if not config.pos_vocab or not config.dep_vocab or not config.ner_vocab or not config.char_vocab:
        logger.error("Vocabularies not loaded after data loading. Exiting.")
        return

    try:
        model = NERModel(
            num_ner_tags=len(config.ner_vocab),
            pos_vocab_size=len(config.pos_vocab),
            dep_vocab_size=len(config.dep_vocab),
            char_vocab_size=len(config.char_vocab)
        ).to(config.DEVICE)
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        return

    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    logger.info(f"Optimizer: AdamW (LR={config.LEARNING_RATE}, WeightDecay={config.WEIGHT_DECAY})")

    # --- GradScaler for Mixed Precision ---
    scaler = GradScaler(enabled=config.USE_AMP)
    logger.info(f"Automatic Mixed Precision (AMP) enabled: {config.USE_AMP}")

    # --- Training Loop Setup ---
    best_val_f1 = -1.0 # Initialize best F1 score
    epochs_no_improve = 0 # Counter for early stopping
    id_to_ner = data_utils.get_id_to_ner() # Get ID to tag mapping for evaluation
    if not id_to_ner:
        logger.error("Failed to get id_to_ner mapping. Exiting.")
        return

    logger.info("Starting training...")
    final_epoch = 0 # Track the last completed epoch
    training_start_time = time.time()

    for epoch in range(config.EPOCHS):
        final_epoch = epoch # Update last completed epoch
        logger.info(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")

        # --- Training Step ---
        train_loss = train_epoch(model, train_loader, optimizer, scaler, config.DEVICE)

        # --- Validation Step ---
        val_loss, val_f1 = evaluate_epoch(model, val_loader, config.DEVICE, id_to_ner)

        # --- Logging Metrics ---
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('F1/validation', val_f1, epoch)

        # --- Checkpoint Saving & Early Stopping ---
        if val_f1 > best_val_f1:
            logger.info(f"Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving model...")
            best_val_f1 = val_f1
            try:
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            except Exception as e:
                logger.error(f"Error saving model checkpoint: {e}", exc_info=True)
            epochs_no_improve = 0 # Reset counter
        else:
            epochs_no_improve += 1
            logger.info(f"Validation F1 did not improve. ({epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE})")

        # Check for early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    training_duration = time.time() - training_start_time
    logger.info(f"Training finished after {final_epoch + 1} epochs. Total time: {training_duration:.2f}s")

    # --- Final Evaluation on Test Set ---
    logger.info("--- Evaluating on Test Set using Best Model ---")
    # Load the best model saved during training
    if os.path.exists(config.MODEL_SAVE_PATH):
        try:
            # Ensure model structure is defined before loading state_dict
            # Re-initialize model structure (in case it was deleted)
            model = NERModel(
                num_ner_tags=len(config.ner_vocab),
                pos_vocab_size=len(config.pos_vocab),
                dep_vocab_size=len(config.dep_vocab),
                char_vocab_size=len(config.char_vocab)
            )
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
            model.to(config.DEVICE) # Move loaded model to device
            model.eval() # Set to evaluation mode
            logger.info(f"Loaded best model from {config.MODEL_SAVE_PATH} for final evaluation.")
        except Exception as e:
            logger.error(f"Error loading best model state dict: {e}. Evaluating with the last state.", exc_info=True)
            model.eval() # Ensure it's in eval mode anyway
    else:
        logger.warning("Best model checkpoint not found. Evaluating with the model's last state.")
        model.eval() # Ensure it's in eval mode

    # Perform evaluation on the test set, requesting examples
    try:
        test_loss, test_f1, test_report, test_examples = evaluate_epoch(
            model, test_loader, config.DEVICE, id_to_ner, return_examples=True, num_examples=10
        )
        logger.info(f"Final Test Results - Loss: {test_loss:.4f}, F1 (macro): {test_f1:.4f}")
        logger.info(f"Final Test Classification Report:\n{test_report}")

        # --- Log Final Metrics and Report to TensorBoard ---
        writer.add_scalar('F1/test', test_f1, 0) # Use step 0 for final test metrics
        # Format report for TensorBoard text display
        report_text_tb = f"<pre>{test_report}</pre>"
        writer.add_text('Report/test', report_text_tb, 0)

        # Log hyperparameters and final metrics
        hparams = {
            'lr': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'epochs_run': final_epoch + 1,
            'lstm_hidden': config.LSTM_HIDDEN_DIM,
            'dropout': config.DROPOUT_RATE
        }
        # Metrics should be simple scalar values
        metrics_dict = {'hparam/test_f1': test_f1, 'hparam/test_loss': test_loss}
        # Filter out any non-scalar hparams if necessary before logging
        scalar_hparams = {k: v for k, v in hparams.items() if isinstance(v, (int, float, bool, str))}

        writer.add_hparams(scalar_hparams, metrics_dict, run_name=log_dir) # Associate with the current run

        # --- Display Prediction Examples ---
        logger.info("--- Prediction Examples (Test Set) ---")
        for i, example in enumerate(test_examples):
            example_log = f"\nExample {i+1}:\n"
            header = f"{'Token':<15} | {'True NER':<10} | {'Predicted NER':<10}\n"
            separator = "-" * len(header.strip()) + "\n"
            example_log += header + separator
            for token, true_tag, pred_tag in zip(example['tokens'], example['true_tags'], example['pred_tags']):
                example_log += f"{token:<15} | {true_tag:<10} | {pred_tag:<10}\n"
            logger.info(example_log)

    except Exception as e:
        logger.error(f"Error during final test evaluation: {e}", exc_info=True)

    # --- Cleanup ---
    writer.close() # Close the TensorBoard writer
    logger.info("TensorBoard writer closed.")
    logger.info("Cleaning up resources...")
    # Explicitly delete large objects to potentially help GC
    del model
    del optimizer
    del scaler
    del train_loader, val_loader, test_loader
    # Clear CUDA cache if using GPU
    if config.DEVICE.type == 'cuda':
        try:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {e}")

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
