<h1 align="center">🧠 NLP-Based Chatbot with Intent Recognition</h1>

<p align="center">
  An intelligent Python chatbot that understands user intent using NLP and machine learning.
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=900&color=0E75B6&center=true&vCenter=true&width=750&lines=NLP-Powered+Intent+Recognition;TF-IDF+%7C+Naive+Bayes+%7C+NLTK;Sentiment+Analysis+%7C+Conversation+Logging;Smarter+Than+Simple+Keyword+Matching" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-85C1E9?style=for-the-badge" alt="NLTK" />
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/Level-Intermediate-F59E0B?style=for-the-badge" alt="Intermediate Project" />
</p>

<br/>

## ✨ About the Project

This chatbot goes beyond simple keyword matching. It uses **Natural Language Processing (NLP)** and a machine-learning model to identify what the user means, classify the message into an intent, and generate a suitable response.

It also tracks conversation context, detects user sentiment, and saves every interaction to a log file for review.

<br/>

## 🚀 Features

- 🔤 Text preprocessing using NLTK
- ✂️ Tokenization, stopword removal, and lemmatization
- 🧠 Intent classification with **TF-IDF + Multinomial Naive Bayes**
- 🏷️ Supports 6 intents:
  - `greeting`
  - `farewell`
  - `question`
  - `complaint`
  - `request`
  - `thanks`
- 📈 Confidence-aware fallback for unclear messages
- 💬 Conversation context tracking between messages
- 😊 Sentiment analysis using NLTK VADER
- 📝 JSON-line conversation logging
- 📊 Session summary when the chatbot closes

<br/>

## 🛠️ Tech Stack

<p align="left">
  <img src="https://skillicons.dev/icons?i=python" alt="Python" />
  <img src="https://img.shields.io/badge/NLTK-02569B?style=flat-square&logo=python&logoColor=white" height="48" alt="NLTK" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" height="48" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/TF--IDF-Text%20Vectorization-8B5CF6?style=flat-square" height="48" alt="TF-IDF" />
  <img src="https://img.shields.io/badge/Naive%20Bayes-Classification-22C55E?style=flat-square" height="48" alt="Naive Bayes" />
</p>

<br/>

## 📁 Project Structure

```text
nlp-based-chatbot/
│
├── chatbot_nlp.py
├── requirements.txt
├── WRITEUP.md
└── README.md
```

> After running the chatbot, `conversation_log.txt` is created automatically.

<br/>

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd nlp-based-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot

```bash
python chatbot_nlp.py
```

> On the first run, the chatbot downloads the required small NLTK datasets. This only happens once.

<br/>

## 💬 Example Conversation

```text
You: hello there
Bot: Hello! How can I help you today?
     [intent: greeting | confidence: 0.74 | sentiment: neutral]

You: this app keeps crashing
Bot: I'm sorry to hear that. Can you tell me more about the issue?
     [intent: complaint | confidence: 0.82 | sentiment: neutral]

You: please cancel my subscription
Bot: Sure, I can help with that request. I'll link this to the issue you just mentioned.
     [intent: request | confidence: 0.85 | sentiment: neutral]

You: bye
Bot: Goodbye! Have a great day.

(Session summary: 3 message(s) exchanged. Full log saved to 'conversation_log.txt'.)
```

<br/>

## ⚙️ How It Works

```text
User Input
     ↓
Text Preprocessing
     ↓
Tokenization + Stopword Removal + Lemmatization
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes Classifier
     ↓
Intent + Confidence Score
     ↓
Sentiment Analysis + Context-Aware Response
     ↓
Save Exchange to Conversation Log
```

### 1. Text Preprocessing

The chatbot cleans each user message by:

- Converting text to lowercase
- Breaking text into tokens
- Removing punctuation and stopwords
- Reducing words to their base form through lemmatization

### 2. Intent Classification

The cleaned text is converted into numerical features using **TF-IDF vectorization**. A **Multinomial Naive Bayes** model then predicts the most likely intent.

### 3. Confidence Check

If the highest prediction probability is below `0.45`, the chatbot does not guess. Instead, it asks the user to rephrase their message.

### 4. Context-Aware Replies

The chatbot remembers the previous intent. This allows connected replies, such as recognizing that a request may be related to a complaint from the previous message.

### 5. Sentiment and Logging

NLTK's VADER sentiment analyzer identifies whether a message appears positive, negative, or neutral. Each exchange is saved with:

- Timestamp
- User message
- Detected intent
- Confidence score
- Sentiment
- Bot response

<br/>

## 📚 Learning Concepts

- Natural Language Processing
- Text tokenization
- Stopword removal
- Lemmatization
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Intent recognition
- Sentiment analysis
- Confidence thresholds
- JSON logging
- Conversation context management

<br/>

## 📖 Documentation

For a complete explanation of the model, preprocessing pipeline, and project decisions, see:

[📄 View the Full Project Write-Up](./WRITEUP.md)

