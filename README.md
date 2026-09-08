# CodeAlpha Chatbot for FAQs

## Project Overview

This project is an NLP-based FAQ chatbot developed as part of the CodeAlpha Artificial Intelligence Internship.

The chatbot accepts questions from users and finds the most relevant FAQ using TF-IDF vectorization and cosine similarity.

## Features

- FAQ-based chatbot
- Natural Language Processing
- Text preprocessing
- TF-IDF vectorization
- Cosine similarity
- Similar question matching
- Interactive command-line interface
- Unknown-question handling

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn

## How It Works

1. FAQ questions and answers are stored in a CSV file.
2. User questions are cleaned and preprocessed.
3. FAQ questions are converted into TF-IDF vectors.
4. The user's question is converted into a TF-IDF vector.
5. Cosine similarity is calculated between the user question and FAQ questions.
6. The most similar FAQ is selected.
7. The corresponding answer is displayed.

## Project Structure

```text
CodeAlpha_Chatbot_for_FAQs/
│
├── chatbot.py
├── faqs.csv
├── requirements.txt
└── README.md