# IntentBot 

## Objective
Move beyond simple keyword matching (Task 1) to a chatbot that classifies
the user's **intent** using basic NLP + ML techniques, and responds
accordingly while keeping light conversational context.

## Approach

**1. Text preprocessing (NLTK)**
Every user message is lowercased, tokenized with `word_tokenize`, stripped
of stopwords (e.g. "the", "is", "a") and non-alphabetic tokens, and
lemmatized with `WordNetLemmatizer` (e.g. "crashing" → "crash",
"complaints" → "complaint"). This normalization reduces vocabulary noise
so the classifier can generalize from a small training set.

**2. Intent classification (scikit-learn)**
A small hand-labeled dataset (~40 example sentences across 6 intents) is
vectorized with `TfidfVectorizer` (term-frequency–inverse-document-frequency,
which weights distinctive words more heavily than common ones) and fed into
a `MultinomialNB` (Naive Bayes) classifier via an sklearn `Pipeline`. Naive
Bayes was chosen because it's fast, simple to understand, and performs
well on small text-classification datasets — a good fit for a learning
project. Smoothing (`alpha=0.1`) was tuned down from the default, since the
default over-smooths probabilities on a small vocabulary and makes the
model appear falsely "unsure."

The model outputs a probability for each intent; if the top probability is
below a confidence threshold (0.45), the bot treats the intent as
**"unclear"** rather than guessing, and asks the user to rephrase.

**3. Intents supported (6 total, exceeds the required 5)**
`greeting`, `farewell`, `question`, `complaint`, `request`, `thanks`.

**4. Context tracking**
The bot keeps a small `context` dictionary holding the last detected
intent and the last message. This is used to make responses feel more
connected — e.g., a `request` right after a `complaint` gets linked back
("I'll link this to the issue you just mentioned"), and two consecutive
`complaint`s trigger an escalation-style reply.

**5. Bonus — Sentiment analysis**
NLTK's built-in VADER (`SentimentIntensityAnalyzer`) scores each message's
sentiment (`compound` score), bucketed into `happy`, `frustrated`, or
`neutral`. This is displayed alongside the detected intent for
transparency.

**6. Bonus — Conversation logging**
Every exchange (timestamp, user text, detected intent, confidence,
sentiment, bot reply) is appended as a JSON line to `conversation_log.txt`,
making it easy to review sessions later or feed them back into future
model training.

## Limitations
The training set is intentionally small for a learning exercise, so the
model's real-world accuracy on unseen phrasing is limited. A production
version would use a larger, more diverse labeled dataset and possibly a
stronger model (e.g. logistic regression or a transformer-based classifier).