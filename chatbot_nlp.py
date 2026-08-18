import random
import json
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# STEP 1: Make sure NLTK data is available.
# This runs once - if the data is already downloaded, it's skipped instantly.
# We do this here (instead of assuming it's pre-installed) so the script
# works out-of-the-box on a fresh machine.

def ensure_nltk_data():
    required = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "sentiment/vader_lexicon": "vader_lexicon",
    }
    for path, pkg in required.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()

# STEP 2: Preprocessing tools
# - Tokenizer: splits a sentence into individual words
# - Stopwords: common filler words ("the", "is", "a") that carry little
#   meaning for intent detection, so we remove them
# - Lemmatizer: reduces words to their base/dictionary form
#   (e.g. "running" -> "run", "complaints" -> "complaint")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """
    Cleans and normalizes raw text so the ML model sees consistent input.
    Steps: lowercase -> tokenize -> remove stopwords/punctuation -> lemmatize
    Returns a single cleaned string (words joined by spaces).
    """
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in STOP_WORDS
    ]
    return " ".join(cleaned_tokens)

# STEP 3: Training data
# Each entry is (example_sentence, intent_label).
# In a real product this would come from a much bigger labeled dataset,
# but for a learning project, a small hand-written set works well.

TRAINING_DATA = [
    # greeting
    ("hello there", "greeting"),
    ("hi", "greeting"),
    ("hey how are you", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("what's up", "greeting"),
    ("yo", "greeting"),
    # farewell
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("i have to go now", "farewell"),
    ("talk to you later", "farewell"),
    ("catch you later", "farewell"),
    # question
    ("what is your name", "question"),
    ("how does this work", "question"),
    ("what time is it", "question"),
    ("can you explain that", "question"),
    ("why is this happening", "question"),
    ("how do i reset my password", "question"),
    ("what services do you offer", "question"),
    # complaint
    ("this is not working", "complaint"),
    ("i am unhappy with the service", "complaint"),
    ("my order was late again", "complaint"),
    ("this app keeps crashing", "complaint"),
    ("i am frustrated with the support team", "complaint"),
    ("nothing works properly here", "complaint"),
    ("this is really disappointing", "complaint"),
    # request
    ("can you help me book a ticket", "request"),
    ("please cancel my subscription", "request"),
    ("i need a refund", "request"),
    ("could you send me the invoice", "request"),
    ("please update my address", "request"),
    ("i want to change my password", "request"),
    ("can you schedule a call for me", "request"),
    # thanks
    ("thank you so much", "thanks"),
    ("thanks a lot", "thanks"),
    ("i really appreciate your help", "thanks"),
    ("thanks for the quick response", "thanks"),
    ("much appreciated", "thanks"),
]


# STEP 4: Responses for each intent.
# Lists are used so we can randomly vary the reply (keeps chat less robotic).

RESPONSES = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
    ],
    "farewell": [
        "Goodbye! Have a great day.",
        "See you later, take care!",
    ],
    "question": [
        "That's a good question — let me try to help with that.",
        "Here's what I can tell you about that.",
    ],
    "complaint": [
        "I'm sorry to hear that. Can you tell me more about the issue?",
        "That sounds frustrating — let's see how I can help fix it.",
    ],
    "request": [
        "Sure, I can help with that request.",
        "Got it — I'll note that request down for you.",
    ],
    "thanks": [
        "You're very welcome!",
        "Anytime, happy to help!",
    ],
    "unclear": [
        "I'm not fully sure what you mean — could you rephrase that?",
        "Sorry, I didn't quite catch your intent there. Can you say it differently?",
    ],
}

# Minimum confidence the model must have before we trust its prediction.
# Below this, we treat the intent as "unclear" instead of guessing wildly.
CONFIDENCE_THRESHOLD = 0.45

EXIT_WORDS = {"bye", "goodbye", "exit", "quit"}

LOG_FILE = "conversation_log.txt"


# STEP 5: Build and train the intent classification model.
# Pipeline = TF-IDF vectorizer (turns text into numeric features that
# reflect word importance) + Multinomial Naive Bayes (a simple, fast,
# well-known classifier for text data).

def train_intent_model():
    texts = [preprocess(text) for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    # alpha=0.1 lowers Laplace smoothing. With a small training set like ours,
    # the default alpha=1.0 over-smooths and pushes all class probabilities
    # toward "uniform", making the model seem unsure even when it isn't.
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB(alpha=0.1)),
    ])
    model.fit(texts, labels)
    return model


def predict_intent(model, user_text: str):
    """
    Returns (intent_label, confidence_score).
    confidence_score is the model's own probability estimate (0 to 1)
    for the predicted class - lets us detect "I'm not sure" situations.
    """
    cleaned = preprocess(user_text)
    if cleaned.strip() == "":
        return "unclear", 0.0

    probabilities = model.predict_proba([cleaned])[0]
    best_index = probabilities.argmax()
    intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return "unclear", confidence
    return intent, confidence


# STEP 6 (Bonus): Sentiment analysis using NLTK's VADER.
# VADER gives a "compound" score from -1 (very negative) to +1 (very positive).
# We bucket it into simple human-readable labels.

SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


def analyze_sentiment(user_text: str) -> str:
    score = SENTIMENT_ANALYZER.polarity_scores(user_text)["compound"]
    if score >= 0.3:
        return "happy"
    elif score <= -0.3:
        return "frustrated"
    else:
        return "neutral"


# STEP 7 (Bonus): Log every exchange to a file for later review.

def log_conversation(user_text, intent, confidence, sentiment, bot_reply):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "user": user_text,
        "detected_intent": intent,
        "confidence": round(float(confidence), 2),
        "sentiment": sentiment,
        "bot_reply": bot_reply,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# STEP 8: Context tracking.
# We remember the last detected intent and the last user message.
# This lets the bot react a little differently if, say, the user is
# following up on a previous complaint or question.

def build_response(intent: str, user_text: str, context: dict) -> str:
    base_reply = random.choice(RESPONSES[intent])

    # Example of context-awareness:
    # If the user complains again right after an earlier complaint,
    # acknowledge that this is a continuation of the same issue.
    if intent == "complaint" and context.get("last_intent") == "complaint":
        base_reply += " It sounds like this issue is still ongoing — I'll flag it as a priority."

    # If the user asks a follow-up question right after another question,
    # acknowledge the continuity.
    if intent == "question" and context.get("last_intent") == "question":
        base_reply += " (Following up on your previous question, I'll do my best to clarify further.)"

    # If a request comes right after a complaint, connect the two.
    if intent == "request" and context.get("last_intent") == "complaint":
        base_reply += " I'll link this to the issue you just mentioned."

    return base_reply


# STEP 9: Main chat loop

def main():
    print("=" * 60)
    print(" IntentBot - NLP/ML Based Chatbot")
    print(" (Understands intent, not just keywords. Type 'bye' to exit.)")
    print("=" * 60)

    model = train_intent_model()

    # context stores what happened in the previous turn
    context = {"last_intent": None, "last_message": None}
    message_count = 0

    while True:
        user_text = input("You: ").strip()

        if user_text == "":
            print("Bot: Please type something!")
            continue

        message_count += 1

        # Exit check (kept simple/explicit, like Task 1)
        if user_text.lower() in EXIT_WORDS:
            print("Bot: Goodbye! Have a great day.")
            print(f"\n(Session summary: {message_count} message(s) exchanged. "
                  f"Full log saved to '{LOG_FILE}'.)")
            log_conversation(user_text, "farewell", 1.0, analyze_sentiment(user_text), "Goodbye!")
            break

        intent, confidence = predict_intent(model, user_text)
        sentiment = analyze_sentiment(user_text)
        reply = build_response(intent, user_text, context)

        # Show intent + confidence + sentiment transparently (great for
        # understanding/debugging how the model is deciding things).
        print(f"Bot: {reply}")
        print(f"     [intent: {intent} | confidence: {confidence:.2f} | sentiment: {sentiment}]")

        log_conversation(user_text, intent, confidence, sentiment, reply)

        # Update context for the next turn
        context["last_intent"] = intent
        context["last_message"] = user_text


if __name__ == "__main__":
    main()