import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.metrics import classification_report
import json
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"\d+", "NUM", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text
def predict_message(text):

    text = clean_text(text)


    seq = tokenizer.texts_to_sequences([text])


    padded = pad_sequences(seq, maxlen=MAX_LEN)


    prob = model.predict(padded)[0][0]


    if prob > 0.5:
        label = "SPAM / PHISHING"
    else:
        label = "HAM / SAFE"

    print(f"\nMessage: {text}")
    print(f"Probability: {prob:.4f}")
    print(f"Prediction: {label}")
MAX_WORDS = 10000
MAX_LEN = 100
df=pd.read_csv(r"C:\Users\krish\Desktop\Python\NN\spam.csv",encoding="latin-1")
df=df[['v1','v2']]
df.columns=['label','message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].apply(clean_text)
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['message'])
sequences = tokenizer.texts_to_sequences(df['message'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
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

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    f.write(tokenizer_json)
y_pred = (model.predict(X_test) > 0.5).astype(int)
model.export("sms_model")
model.save("sms_model.keras")
print(classification_report(y_test, y_pred))
# Test examples
# while True:
#     user_input = input("\nEnter SMS (or type exit): ")
#     if user_input.lower() == "exit":
#         break
#     predict_message(user_input)
