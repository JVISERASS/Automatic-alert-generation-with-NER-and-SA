# Automatic Alert Generation with NER and Sentiment Analysis

## 🚀 Project Overview
This project aims to develop an automatic alert generation system from news articles and social media posts. The system leverages Named Entity Recognition (NER) and Sentiment Analysis (SA) techniques to produce contextual alerts relevant to reputation monitoring, economic updates, and geopolitical risks.

## 🔍 How It Works
1. **Named Entity Recognition (NER):** Identifies key entities such as people, organizations, monetary values, and locations using a custom LSTM-based model.
2. **Sentiment Analysis (SA):** Classifies text sentiment as positive, neutral, or negative using a neural network model.
3. **Alert Generation (AG):** Combines NER and SA results to generate meaningful alerts.

## 📌 Example
**Input (Text):**  
"Musk accused of making a Nazi salute during Trump’s inauguration..."

**Output:**  
"Reputation Risk: Elon Musk"

## 🔥 Project Levels
The project can be developed at different levels of complexity:
- **Basic (up to 7.0 points):** Separate NER and SA models, rule-based alert generation.
- **Intermediate (up to 9.0 points):** Joint architecture for NER and SA with a combined loss function and AI-based alert generation.
- **Advanced (up to 10.0 points):** Image processing, captioning model (CNN + RNN) to generate textual descriptions and enhance alert generation.

## 🛠️ Tools & Requirements
- **Suggested Datasets:**
  - [CoNLL-2003](https://paperswithcode.com/dataset/conll-2003) (NER)
  - [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (SA)
- **Frameworks & Tools:**
  - [spaCy](https://spacy.io/) for NER
  - [TensorBoard](https://www.tensorflow.org/tensorboard) for training visualization
  - [Hugging Face](https://huggingface.co/) for pre-trained models

## 📂 Installation & Execution
```bash
# Clone the repository
git clone https://github.com/JVISERASS/Automatic-alert-generation-with-NER-and-SA.git
cd Automatic-alert-generation-with-NER-and-SA

# Create a virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# Train models
python train_ner.py
python train_sa.py

# Generate alerts
python generate_alerts.py --input "path/to/file.txt"
```

## 📌 Deliverables
1. **Project Proposal:** Initial research, plan, and objectives.
2. **Progress Report:** Problem definition, technical approach, initial experiments.
3. **Final Submission:** Documented and reproducible code, a scientific paper in LaTeX, and a declaration of team roles.

## 👥 Contributors
- Miguel Angel Vallejo de Bergia (@user1)
- Bernardo Ordas Cernadas (@user2)
- Javier Viseras Comin (@JVISERASS)


