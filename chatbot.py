import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download("punkt", quiet=True)

# Load FAQ dataset
data = pd.read_csv("faqs.csv")

questions = data["question"].tolist()
answers = data["answer"].tolist()


# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


processed_questions = [
    preprocess_text(question)
    for question in questions
]


# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)


# Find the most similar FAQ
def get_response(user_question):

    processed_user_question = preprocess_text(user_question)

    user_vector = vectorizer.transform([processed_user_question])

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    # Minimum similarity threshold
    if best_score < 0.25:
        return (
            "Sorry, I could not find a suitable answer "
            "to your question."
        )

    return answers[best_match_index]


# Chatbot interface
print("=" * 60)
print("          CodeAlpha FAQ Chatbot")
print("=" * 60)

print("Ask me a question.")
print("Type 'exit' or 'quit' to stop the chatbot.")
print()

while True:

    user_input = input("You: ")

    if user_input.lower().strip() in ["exit", "quit"]:
        print("Bot: Thank you for using the FAQ Chatbot!")
        break

    response = get_response(user_input)

    print("Bot:", response)
    print()