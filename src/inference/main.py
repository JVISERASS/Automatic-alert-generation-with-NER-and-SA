"""
Pipeline de Inferencia Completo

Este script carga los tres modelos del proyecto (NER, SA y Generador de Alertas) 
y realiza inferencias en un archivo de texto, mostrando los resultados de manera integrada.
"""

import os
import sys
import json
import torch
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from colorama import Fore, Style, init

# Inicializar colorama
init()

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Añadir la ruta al directorio src para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.ner_train.predictor import PipelinePredictor
from train.ner_train import config as ner_config
from train.sa_train.model import SentimentFromTextNerLSTM

def load_ner_model():
    """Carga el modelo NER y su configuración."""
    model_path = os.path.join(ner_config.MODEL_OUTPUT_DIR, "ner_model_best.pt")
    vocab_cache_dir = os.path.join(ner_config.DATA_DIR, "ner_cache")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: No se encontró el modelo NER en {model_path}")
    
    predictor = PipelinePredictor(model_path=model_path, vocab_cache_dir=vocab_cache_dir)
    logger.info(f"Modelo NER cargado correctamente desde {model_path}")
    
    return predictor

def load_sa_model():
    """Carga el modelo de análisis de sentimiento y sus vocabularios."""
    # Definir rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_path = os.path.join(project_root, "src", "models", "sa_model", "best_model_twitter_sentiment_ner.pth")
    vocab_dir = os.path.join(project_root, "src", "data", "sa_vocabs")
    
    text_vocab_path = os.path.join(vocab_dir, "twitter_text_vocab.pkl")
    ner_vocab_path = os.path.join(vocab_dir, "twitter_ner_vocab.pkl")
    sentiment_vocab_path = os.path.join(vocab_dir, "twitter_sentiment_vocab.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: No se encontró el modelo SA en {model_path}")
    
    # Cargar vocabularios
    import pickle
    from collections import defaultdict
    
    with open(text_vocab_path, 'rb') as f:
        text_vocab_dict = pickle.load(f)
        text_vocab = defaultdict(lambda: text_vocab_dict.get('<UNK>', 1))
        for k, v in text_vocab_dict.items():
            text_vocab[k] = v
    
    with open(ner_vocab_path, 'rb') as f:
        ner_vocab_dict = pickle.load(f)
        ner_vocab = defaultdict(lambda: ner_vocab_dict.get('O', 1))
        for k, v in ner_vocab_dict.items():
            ner_vocab[k] = v
    
    with open(sentiment_vocab_path, 'rb') as f:
        sentiment_vocab_dict = pickle.load(f)
        sentiment_vocab = defaultdict(lambda: sentiment_vocab_dict.get('NEU', 1))
        for k, v in sentiment_vocab_dict.items():
            sentiment_vocab[k] = v
    
    # Crear el mapeo inverso para las etiquetas de sentimiento
    sentiment_vocab_inverse = {v: k for k, v in sentiment_vocab.items()}
    
    # Cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentFromTextNerLSTM(
        text_vocab_size=len(text_vocab),
        text_embedding_dim=200,
        ner_vocab_size=len(ner_vocab),
        ner_embedding_dim=50,
        hidden_dim=256,
        sentiment_vocab_size=len(sentiment_vocab),
        n_layers=2,
        bidirectional=True,
        dropout=0.5
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Modelo SA cargado correctamente desde {model_path}")
    
    return model, text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse, device

def load_alert_generator_model():
    """Carga el modelo generador de alertas basado en T5."""
    # Definir rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(project_root, "src", "models", "alert_generator_model")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Error: No se encontró el directorio del modelo de alertas en {model_dir}")
    
    # Cargar modelo y tokenizador
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"Modelo generador de alertas cargado correctamente desde {model_dir}")
        
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error al cargar el modelo generador de alertas: {e}")
        raise

def predict_sentiment(sa_model, text, ner_tags, text_vocab, ner_vocab, sentiment_vocab_inverse, device):
    """Predice el sentimiento de un texto usando el modelo SA."""
    # Tokenizar el texto
    tokens = text.lower().strip().split()
    
    # Codificar tokens
    encoded_text = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in text_vocab:
            encoded_text.append(text_vocab[token_lower])
        else:
            encoded_text.append(text_vocab['<UNK>'])
    
    # Preparar etiquetas NER
    # Si las etiquetas NER son un diccionario (token -> etiqueta), convertirlas a lista
    if isinstance(ner_tags, dict):
        encoded_ner = []
        for token in tokens:
            tag = ner_tags.get(token, "O")  # Usar 'O' como valor predeterminado
            encoded_ner.append(ner_vocab[tag])
    else:
        # Si ya es una lista, usarla directamente
        encoded_ner = [ner_vocab.get(tag, ner_vocab['O']) for tag in ner_tags]
    
    # Recortar si es necesario
    max_len = 128
    if len(encoded_text) > max_len:
        encoded_text = encoded_text[:max_len]
        encoded_ner = encoded_ner[:max_len]
    
    # Convertir a tensores
    text_tensor = torch.tensor([encoded_text], dtype=torch.long).to(device)
    ner_tensor = torch.tensor([encoded_ner], dtype=torch.long).to(device)
    
    # Realizar predicción
    with torch.no_grad():
        output = sa_model(text_tensor, ner_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        pred_class_idx = pred_class.item()
        confidence_val = confidence.item()
    
    # Obtener etiqueta de sentimiento
    sentiment = sentiment_vocab_inverse.get(pred_class_idx, "NEU")
    
    return sentiment, confidence_val

def generate_alert(alert_model, tokenizer, text, ner_result, sentiment, device):
    """Genera una alerta usando el modelo generador."""
    # Formato de entrada para el modelo
    ner_text = " ".join([f"{token}:{tag}" for token, tag in ner_result.items()])
    input_text = f"text: {text} ner: {ner_text} sentiment: {sentiment}"
    
    # Tokenizar entrada
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generar alerta
    with torch.no_grad():
        outputs = alert_model.generate(
            input_ids,
            max_length=100,
            num_beams=4,
            early_stopping=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            do_sample=True
        )
    
    # Decodificar resultado
    alert = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return alert

def print_colored_result(text, ner_result, sentiment, confidence, alert):
    """Imprime resultados con formato y colores para mejor visualización."""
    # Definir colores según el sentimiento
    sentiment_color = Fore.GREEN
    if sentiment == "NEG":
        sentiment_color = Fore.RED
    elif sentiment == "NEU":
        sentiment_color = Fore.BLUE
    
    # Cabecera
    print("\n" + "="*80)
    print(f"{Fore.CYAN}TEXTO ANALIZADO:{Style.RESET_ALL}")
    print(f"{text}")
    print("-"*80)
    
    # Resultados NER
    print(f"{Fore.YELLOW}ENTIDADES RECONOCIDAS:{Style.RESET_ALL}")
    for token, tag in ner_result.items():
        tag_color = Fore.WHITE
        if "PER" in tag:
            tag_color = Fore.MAGENTA
        elif "LOC" in tag:
            tag_color = Fore.GREEN
        elif "ORG" in tag:
            tag_color = Fore.BLUE
        elif "MISC" in tag:
            tag_color = Fore.CYAN
        
        if tag != "O":
            print(f"  {token}: {tag_color}{tag}{Style.RESET_ALL}")
    
    # Resultados sentimiento
    print(f"{Fore.YELLOW}ANÁLISIS DE SENTIMIENTO:{Style.RESET_ALL}")
    print(f"  {sentiment_color}{sentiment}{Style.RESET_ALL} (Confianza: {confidence:.2f})")
    
    # Alerta generada
    print(f"{Fore.YELLOW}ALERTA GENERADA:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}{Style.BRIGHT}{alert}{Style.RESET_ALL}")
    
    print("="*80)

def main():
    """Función principal para realizar el pipeline completo de inferencia."""
    try:
        # Cargar los tres modelos
        logger.info("Cargando modelos...")
        ner_predictor = load_ner_model()
        sa_model, text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse, sa_device = load_sa_model()
        alert_model, alert_tokenizer, alert_device = load_alert_generator_model()
        logger.info("Todos los modelos cargados correctamente")
        
        # Cargar datos de entrada
        input_file = os.path.join(ner_config.DATA_DIR, "news_tweets.txt")
        if not os.path.exists(input_file):
            logger.error(f"Error: No se encontró el archivo de entrada en {input_file}")
            return
        
        # Leer archivo de texto
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"Se han cargado {len(lines)} líneas para procesar")
        
        # Procesar cada línea
        num_examples = min(10, len(lines))  # Limitar a 10 ejemplos
        
        for i, text in enumerate(lines[:num_examples], 1):
            logger.info(f"Procesando texto {i}/{num_examples}: '{text[:50]}...'")
            
            # 1. Reconocimiento de entidades (NER)
            ner_result = ner_predictor._predict_ner(text)
            
            # 2. Análisis de sentimiento (SA)
            sentiment, confidence = predict_sentiment(
                sa_model, text, ner_result, text_vocab, ner_vocab, 
                sentiment_vocab_inverse, sa_device
            )
            
            # 3. Generación de alerta
            alert = generate_alert(
                alert_model, alert_tokenizer, text, ner_result, 
                sentiment, alert_device
            )
            
            # 4. Mostrar resultados con formato
            print_colored_result(text, ner_result, sentiment, confidence, alert)
            
            # Resultado como diccionario JSON (opcional para exportar)
            result = {
                "text": text,
                "ner_result": ner_result,
                "sentiment": {
                    "label": sentiment,
                    "confidence": confidence
                },
                "alert": alert
            }
        
        logger.info("Procesamiento completado")
    
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Error durante la inferencia: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()