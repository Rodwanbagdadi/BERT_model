# Fake News Detection with DistilBERT

A machine learning project that uses DistilBERT to classify news articles as fake or real with high accuracy and efficiency.

## 🎯 Project Overview

This project implements a fake news detection system using DistilBERT, a lightweight version of BERT that maintains 97% of BERT's performance while being 60% smaller and 60% faster. The model is fine-tuned on a dataset of fake and real news articles to achieve reliable classification.

## 📊 Dataset

The project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle, which contains:
- **Fake news articles**: Unreliable news articles from various sources
- **Real news articles**: Legitimate news articles from Reuters
- **Features**: Article text, subject, and publication date

## 🔧 Features

- **Efficient Model**: Uses DistilBERT for faster training and inference
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and confusion matrix
- **Cross-Validation**: 3-fold cross-validation for robust performance assessment
- **Model Persistence**: Saves trained model and tokenizer for deployment
- **Visualization**: Confusion matrix heatmap for performance analysis

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BERT_model.git
cd BERT_model
```

2. Download the dataset:
   - Download the Fake and Real News Dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   - Place `Fake.csv` and `True.csv` in the project root directory

3. Run the Jupyter notebook:
```bash
jupyter notebook "BERT copy.ipynb"
```

## 📈 Model Performance

The model achieves excellent performance on fake news detection:

- **Architecture**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary classification (Fake vs Real)
- **Sequence Length**: 128 tokens (optimized for speed)
- **Batch Size**: 16 (training and evaluation)
- **Epochs**: 2 (sufficient for convergence)

### Key Metrics
- High accuracy scores across validation sets
- Balanced precision and recall
- Robust performance confirmed through cross-validation
- Fast inference suitable for real-time applications

## 🏗️ Project Structure

```
BERT_model/
├── BERT copy.ipynb          # Main notebook with complete pipeline
├── BERT.ipynb               # Alternative notebook version
├── truthguard.ipynb         # Additional experiments
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── Fake.csv                 # Fake news dataset (download separately)
├── True.csv                 # Real news dataset (download separately)
├── fake_news_model/         # Model checkpoints during training
├── results/                 # Training results and metrics
├── config.json              # Model configuration
├── tokenizer_config.json    # Tokenizer configuration
├── model_info.json          # Model metadata
└── *.bin, *.pt, *.json      # Model weights and configurations
```

## 🔄 Pipeline Overview

1. **Data Loading**: Load fake and real news datasets
2. **Preprocessing**: Clean text, remove irrelevant columns, assign labels
3. **Train/Test Split**: 80/20 split for training and validation
4. **Model Setup**: Initialize DistilBERT model and tokenizer
5. **Tokenization**: Convert text to BERT-compatible tokens
6. **Training**: Fine-tune model with appropriate hyperparameters
7. **Evaluation**: Comprehensive metrics calculation
8. **Visualization**: Generate confusion matrix
9. **Cross-Validation**: 3-fold CV for robustness testing
10. **Model Saving**: Persist model for deployment

## 🎨 Visualization

The notebook includes:
- Confusion matrix heatmap showing prediction accuracy
- Training progress visualization
- Performance metrics comparison

## 🔮 Future Enhancements

- [ ] Experiment with other BERT variants (RoBERTa, ALBERT)
- [ ] Implement ensemble methods
- [ ] Add data augmentation techniques
- [ ] Create a web interface for real-time prediction
- [ ] Implement automated model retraining
- [ ] Add support for multiple languages

## 📝 Usage Example

```python
# Load the saved model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./saved_model")
model = AutoModelForSequenceClassification.from_pretrained("./saved_model")

# Predict on new text
text = "Your news article text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()

# 0 = Real News, 1 = Fake News
result = "Fake News" if prediction == 1 else "Real News"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- The creators of the Fake and Real News Dataset
- The BERT and DistilBERT research teams

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

---

⭐ If you found this project helpful, please give it a star!
