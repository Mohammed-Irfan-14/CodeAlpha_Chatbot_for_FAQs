import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load FAQ dataset
# -----------------------------

data = pd.read_csv("faqs.csv")

questions = data["question"].tolist()
answers = data["answer"].tolist()


# -----------------------------
# Text preprocessing
# -----------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


processed_questions = [
    preprocess_text(question)
    for question in questions
]


# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    processed_questions
)


# -----------------------------
# Generate chatbot response
# -----------------------------

def get_response(user_question):

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.25:
        return (
            "Sorry, I couldn't find a suitable answer "
            "to your question."
        )

    return answers[best_match_index]


# -----------------------------
# Send message
# -----------------------------

def send_message(event=None):

    user_question = entry_box.get().strip()

    if not user_question:
        return

    chat_area.config(state=tk.NORMAL)

    chat_area.insert(
        tk.END,
        "You: " + user_question + "\n",
        "user"
    )

    response = get_response(user_question)

    chat_area.insert(
        tk.END,
        "Bot: " + response + "\n\n",
        "bot"
    )

    chat_area.config(state=tk.DISABLED)

    chat_area.see(tk.END)

    entry_box.delete(0, tk.END)


# -----------------------------
# Main window
# -----------------------------

window = tk.Tk()

window.title("CodeAlpha FAQ Chatbot")

window.geometry("700x550")

window.resizable(False, False)


# -----------------------------
# Title
# -----------------------------

title_label = tk.Label(
    window,
    text="CodeAlpha FAQ Chatbot",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=15)


subtitle_label = tk.Label(
    window,
    text="Ask a question about Artificial Intelligence or CodeAlpha",
    font=("Arial", 10)
)

subtitle_label.pack()


# -----------------------------
# Chat area
# -----------------------------

chat_area = scrolledtext.ScrolledText(
    window,
    width=75,
    height=22,
    font=("Arial", 11),
    wrap=tk.WORD
)

chat_area.pack(padx=15, pady=15)

chat_area.tag_config(
    "user",
    font=("Arial", 11, "bold")
)

chat_area.tag_config(
    "bot",
    font=("Arial", 11)
)

chat_area.insert(
    tk.END,
    "Bot: Hello! Welcome to the CodeAlpha FAQ Chatbot.\n"
)

chat_area.insert(
    tk.END,
    "Bot: Ask me a question to get started.\n\n"
)

chat_area.config(state=tk.DISABLED)


# -----------------------------
# Input area
# -----------------------------

input_frame = tk.Frame(window)

input_frame.pack(
    fill=tk.X,
    padx=15,
    pady=10
)


entry_box = tk.Entry(
    input_frame,
    font=("Arial", 12)
)

entry_box.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=8
)


send_button = tk.Button(
    input_frame,
    text="Send",
    font=("Arial", 11, "bold"),
    command=send_message,
    padx=20,
    pady=5
)

send_button.pack(
    side=tk.RIGHT,
    padx=(10, 0)
)


# Press Enter to send

entry_box.bind(
    "<Return>",
    send_message
)


# -----------------------------
# Start application
# -----------------------------

window.mainloop()