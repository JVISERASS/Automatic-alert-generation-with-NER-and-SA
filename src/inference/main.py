"""
NER Inference Script

Este script carga el modelo NER entrenado y realiza inferencias en un archivo de texto,
mostrando los resultados con la estructura de entidades reconocidas.
"""

import os
import sys
import json
import torch

# Añadir la ruta al directorio src para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.ner_train.predictor import PipelinePredictor
from train.ner_train import config

def main():
    """Función principal para realizar inferencias con el modelo NER."""
    print("Cargando el modelo NER...")
    
    # Rutas de archivos
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, "ner_model_best.pt")
    vocab_cache_dir = os.path.join(config.DATA_DIR, "ner_cache")
    input_file = os.path.join(config.DATA_DIR, "news_tweets.txt")
    
    # Verificar que los archivos existen
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return
    
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo de entrada en {input_file}")
        return
    
    try:
        # Cargar el modelo
        predictor = PipelinePredictor(model_path=model_path, vocab_cache_dir=vocab_cache_dir)
        print("Modelo cargado correctamente.")
        
        # Leer el archivo de texto
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Se han cargado {len(lines)} líneas para procesar.")
        
        # Procesar cada línea
        for i, text in enumerate(lines, 1):
            print(f"\nProcesando texto {i}/{len(lines)}:")
            print(f"Texto: {text}")
            
            # Realizar inferencia
            ner_result = predictor._predict_ner(text)
            
            # Crear resultado en formato similar al de muestra
            result = {
                "text": text,
                "ner_result": ner_result
            }
            
            # Mostrar resultado
            print("Resultado:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Limitar el número de resultados mostrados para evitar sobrecarga
            if i >= 5:  # Mostrar solo los primeros 5 resultados
                print("\n... (se muestran solo los primeros 5 resultados)")
                break
        
        print("\nProcesamiento completado.")
    
    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()