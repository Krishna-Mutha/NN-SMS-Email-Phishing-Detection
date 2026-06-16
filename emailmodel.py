import pandas as pd
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


FILE_PATH = r"C:\Users\krish\Desktop\Python\NN\CEAS_08.csv"
MAX_WORDS = 20000
MAX_LEN = 250
EPOCHS = 5

~
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"\d+", "NUM", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


df = pd.read_csv(FILE_PATH, encoding="latin-1")

print("Columns:", df.columns)


df['subject'] = df['subject'].fillna('').astype(str)
df['body'] = df['body'].fillna('').astype(str)
df['urls'] = df['urls'].fillna('').astype(str)


df['text'] = df['subject'] + " " + df['body'] + " " + df['urls']


df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)


df['text'] = df['text'].apply(clean_text)


print("\nLabel distribution:")
print(df['label'].value_counts(normalize=True))

df['length'] = df['text'].apply(lambda x: len(x.split()))
print("\nText length stats:")
print(df['length'].describe())

print("Before removing duplicates:", len(df))
df = df.drop_duplicates(subset=['text'])
print("After removing duplicates:", len(df))

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)


X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

print("\nTrain distribution:")
print(pd.Series(y_train).value_counts(normalize=True))


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("\nClass weights:", class_weights)


model = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights
)


probs = model.predict(X_test)
y_pred = (probs > 0.5).astype(int)

print("\nEvaluation:")
print(classification_report(y_test, y_pred))

print("\nPrediction stats:")
print("Min:", probs.min())
print("Max:", probs.max())
print("Mean:", probs.mean())


def test_email(subject, body, urls=""):
    text = subject + " " + body + " " + urls
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded)[0][0]
    label = "PHISHING" if prob > 0.5 else "SAFE"
    print("\nTest Email")
    print("Subject:", subject)
    print("Probability:", prob)
    print("Prediction:", label)

print("\n=== Manual Tests ===")

test_email(
    "Urgent account verification",
    "Your bank account has been suspended. Click the link to verify immediately.",
    "http://secure-update-login.com"
)

test_email(
    "Meeting reminder",
    "Please attend the team meeting tomorrow at 10 AM."
)

test_email(
    "WIN FREE MONEY NOW",
    "Claim your prize immediately by clicking this link",
    "http://free-money-now.com"
)


model.export("email_model")

tokenizer_json = tokenizer.to_json()
with open("email_tokenizer.json", "w") as f:
    f.write(tokenizer_json)

print("\nEmail model and tokenizer saved!")