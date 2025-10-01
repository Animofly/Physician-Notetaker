# Physician-Notetaker
# Medical NLP: Clinical Documentation & Sentiment Analysis

A comprehensive Natural Language Processing system for medical applications, featuring clinical note generation, sentiment analysis, and SOAP note generation from medical conversations.

## 📋 Table of Contents

- [Overview]
- [Features]
- [Installation]
- [Project Structure]
- [Usage Guide]
- [Models & Datasets]
- [Results]
- [Dependencies]
- [Contributing]
- [License]

## 🎯 Overview

This project implements three main NLP tasks for healthcare applications:

1. **Medical NLP Summarization**: Converts physician-patient dialogues into structured clinical notes
2. **Sentiment & Intent Analysis**: Classifies mental health statements (Anxiety vs Normal) with intent detection
3. **SOAP Note Generation**: Generates structured SOAP (Subjective, Objective, Assessment, Plan) notes from medical transcripts

## ✨ Features

### 1. Clinical Note Generation
- Uses pre-trained `HealthScribe-Clinical_Note_Generator` model
- Extracts key medical information:
  - Patient symptoms
  - Diagnosis
  - Patient history
  - Treatment plan
- Processes conversational medical dialogues

### 2. Sentiment & Intent Analysis
- **Multi-task classification** using DistilBERT
- Classifies mental health statements into:
  - **Sentiments**: Anxiety, Normal
  - **Intents**: Seeking reassurance, Expressing difficulty, etc.
- Outputs structured JSON format:
  ```json
  {
    "Sentiment": "Anxious",
    "Intent": "Seeking Reassurance"
  }
  ```

### 3. SOAP Note Generation
- **Four implementation approaches**:
  - Rule-based keyword extraction
  - Transformer fine-tuning (T5/BART)
  - BERT token classification
  - Hybrid system
- Generates comprehensive clinical documentation

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd medical-nlp

# Create virtual environment
conda create -n mednlp python=3.10
conda activate mednlp

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=mednlp
```

### Required Packages

```bash
pip install transformers==4.56.2
pip install tensorflow==2.20.0
pip install tf-keras==2.20.1
pip install datasets
pip install pandas
pip install scikit-learn
pip install joblib
```

## 💻 Usage Guide

### 1. Medical Note Generation
**Dataset**: [MTS_Dialogue-Clinical_Note](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note)  
**Model**: [HealthScribe-Clinical_Note_Generator](https://huggingface.co/har1/HealthScribe-Clinical_Note_Generator)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("har1/HealthScribe-Clinical_Note_Generator")
model = AutoModelForSeq2SeqLM.from_pretrained("har1/HealthScribe-Clinical_Note_Generator")

# Prepare conversation
input_text = """
Physician: How are you feeling today?
Patient: I'm doing better, but I still have some discomfort...
"""

# Generate clinical note
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
output_ids = model.generate(**inputs, max_new_tokens=200, num_beams=4)
clinical_note = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(clinical_note)
```

**Expected Output:**
```
Symptoms: occasional backaches
Diagnosis: whiplash injury
History of Patient: Involved in motor vehicle accident...
Plan of Action: N/A
```

### 2. Sentiment & Intent Analysis

#### Data Preparation

```python
import pandas as pd

# Load and clean data
df = pd.read_csv("Combined Data.csv")

# Filter unwanted categories
df = df[~df['status'].isin(['Suicidal', 'Bipolar', 'Personality disorder'])]

# Consolidate categories
df['status'] = df['status'].replace({
    'Depression': 'Anxiety',
    'Stress': 'Anxiety'
})

# Balance dataset
anxiety_df = df[df["status"] == "Anxiety"].sample(n=8000, random_state=42)
normal_df = df[df["status"] == "Normal"].sample(n=8000, random_state=42)
balanced_df = pd.concat([anxiety_df, normal_df]).reset_index(drop=True)

# Save cleaned dataset
balanced_df.to_csv("cleaned_dataset2.csv", index=False)
```

#### Model Training

```python
from transformers import DistilBertTokenizerFast, TFAutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    from_pt=True,
    num_labels=2  # Anxiety, Normal
)

# Train model (see notebook for full code)
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

# Save model
model.save_pretrained("./anxiety_normal_tf_model")
```

#### Inference

```python
import json
import tensorflow as tf

def predict_sentiment_intent(text, model, tokenizer, sentiment_le, intent_le):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="tf", max_length=128, truncation=True)
    
    # Predict
    outputs = model(inputs, training=False)
    sentiment_pred = tf.argmax(outputs['sentiment_output'], axis=1).numpy()[0]
    intent_pred = tf.argmax(outputs['intent_output'], axis=1).numpy()[0]
    
    # Decode labels
    result = {
        "Sentiment": sentiment_le.inverse_transform([sentiment_pred])[0].capitalize(),
        "Intent": intent_le.inverse_transform([intent_pred])[0].replace('_', ' ').title()
    }
    
    return json.dumps(result, indent=2)

# Example usage
text = "I'm really worried about the upcoming exam"
prediction = predict_sentiment_intent(text, model, tokenizer, sentiment_le, intent_le)
print(prediction)
```

**Output:**
```json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking Reassurance"
}
```

### 3. SOAP Note Generation

```python
from soap_generator import RuleBasedSOAPGenerator
import json

# Initialize generator
generator = RuleBasedSOAPGenerator()

# Medical transcript
transcript = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions...
"""

# Generate SOAP note
soap_note = generator.generate_soap_note(transcript)
print(json.dumps(soap_note, indent=2))
```

**Expected Output:**
```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks..."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
    "Observations": "Patient appears in normal health, normal gait."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury and lower back strain",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
    "Follow_Up": "Patient to return if pain worsens or persists beyond six months."
  }
}
```

## 📊 Models & Datasets

### Pre-trained Models Used

1. **HealthScribe-Clinical_Note_Generator**
   - Model: `har1/HealthScribe-Clinical_Note_Generator`
   - Task: Medical dialogue summarization
   - Base: Seq2Seq architecture

2. **DistilBERT**
   - Model: `distilbert-base-uncased`
   - Task: Sentiment & intent classification
   - Fine-tuned on mental health dataset

3. **T5/BART** (Optional)
   - For advanced SOAP note generation
   - Requires fine-tuning on medical data

### Datasets

- **MTS_Dialogue-Clinical_Note**: Medical dialogue dataset from HuggingFace
- **Combined Data.csv**: Mental health statements with labels
  - Original size: 38,175 samples
  - Cleaned size: 16,000 balanced samples
  - Classes: Anxiety (8,000), Normal (8,000)

## 📈 Results

### Dataset Statistics (from Kaggle)

| Category | Original Count | Cleaned Count |
|----------|---------------|---------------|
| Anxiety | 21,832 | 8,000 |
| Normal | 16,343 | 8,000 |
| **Total** | **38,175** | **16,000** |

### Model Performance

- **Clinical Note Generation**: Successfully extracts symptoms, diagnosis, and treatment plans
- **Sentiment Analysis**: Binary classification (Anxiety vs Normal)
- **Intent Detection**: Multi-class classification (6+ intent categories)
- **SOAP Notes**: Structured medical documentation with 4 main sections

## 🔧 Dependencies

```
transformers==4.56.2
tensorflow==2.20.0
tf-keras==2.20.1
datasets
pandas
numpy>=2.2.6
scikit-learn
joblib
jupyter
ipykernel
```

## 🛠️ Troubleshooting

### Common Issues

**1. TensorFlow/Keras Version Conflicts**
```bash
pip install --upgrade tf-keras
pip install tensorflow==2.20.0
```

**2. Memory Errors**
- Reduce batch size to 4 or 2
- Use gradient accumulation
- Process data in smaller chunks

**3. Model Loading Errors**
- Ensure `from_pt=True` when loading PyTorch models in TensorFlow
- Check model compatibility with transformers version

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for pre-trained models and datasets
- Anthropic for medical NLP research
- Open-source community for transformers library


