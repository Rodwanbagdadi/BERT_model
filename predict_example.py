"""
Example script for using the trained fake news detection model.
Run this after training the model with the Jupyter notebook.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model(model_path="./saved_model"):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


def predict_fake_news(text, tokenizer, model, max_length=128):
    """
    Predict if a news article is fake or real.

    Args:
        text (str): The news article text
        tokenizer: The trained tokenizer
        model: The trained model
        max_length (int): Maximum sequence length

    Returns:
        dict: Prediction results with label and confidence
    """
    # Tokenize the input
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Get confidence scores
    confidence_scores = probabilities[0].tolist()

    # Map prediction to label
    label = "Fake News" if predicted_class == 1 else "Real News"
    confidence = confidence_scores[predicted_class]

    return {
        "label": label,
        "confidence": confidence,
        "real_news_probability": confidence_scores[0],
        "fake_news_probability": confidence_scores[1],
    }


def main():
    """Main function to demonstrate usage."""
    print("Loading trained model...")

    try:
        tokenizer, model = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Please make sure you've trained the model first by running the Jupyter notebook."
        )
        return

    # Example news articles for testing
    test_articles = [
        "Scientists have discovered a new species of dinosaur in Argentina. The fossil remains suggest it was one of the largest predators ever found.",
        "Breaking: Local man discovers cure for all diseases using this one weird trick that doctors hate!",
        "The Federal Reserve announced today that interest rates will remain unchanged following their monthly meeting.",
        "Aliens have been confirmed to be living among us, according to a secret government document leaked yesterday.",
    ]

    print("\n" + "=" * 50)
    print("FAKE NEWS DETECTION RESULTS")
    print("=" * 50)

    for i, article in enumerate(test_articles, 1):
        print(f"\nArticle {i}:")
        print(f"Text: {article[:100]}...")

        result = predict_fake_news(article, tokenizer, model)

        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Real News Probability: {result['real_news_probability']:.2%}")
        print(f"Fake News Probability: {result['fake_news_probability']:.2%}")
        print("-" * 50)


if __name__ == "__main__":
    main()
