# NN-SMS-Email-Phishing-Detection

A deep learning-based spam and phishing detection system built using TensorFlow and Convolutional Neural Networks (CNNs). This project provides models for detecting malicious SMS messages and phishing emails, with support for TensorFlow Lite deployment on mobile and edge devices.

---

## Features

* SMS spam detection
* Email phishing detection
* Text preprocessing and normalization
* CNN-based text classification
* TensorFlow/Keras implementation
* SavedModel and Keras model export
* Tokenizer serialization
* TensorFlow Lite conversion for deployment
* Manual testing interface for inference

---

## Project Structure

```
SpamShield-AI/
│
├── sms_detector.py
├── email_detector.py
├── convert_to_tflite.py
│
├── sms_model/
├── sms_model.keras
├── tokenizer.json
│
├── email_model/
├── email_tokenizer.json
│
├── email_model_select.tflite
│
├── datasets/
│   ├── spam.csv
│   └── CEAS_08.csv
│
└── README.md
```

---

## Datasets

### SMS Spam Dataset

The SMS classifier is trained on the Enron Spam Data dataset.

Dataset columns:

| Column | Description      |
| ------ | ---------------- |
| v1     | Label (ham/spam) |
| v2     | Message text     |

Labels:

* ham → 0
* spam → 1

---

### CEAS_08 Email Dataset

The phishing detector is trained on the CEAS 2008 Email Dataset.

Features used:

* Subject
* Email Body
* URLs

Labels:

* 0 → Legitimate Email
* 1 → Phishing Email

---

## Text Preprocessing

Before training, all text undergoes cleaning:

* Convert text to lowercase
* Replace URLs with `URL`
* Replace numbers with `NUM`
* Remove special characters
* Normalize whitespace

Example:

```
Visit https://example.com now and win $1000!
```

becomes

```
visit URL now and win NUM
```

---

## Model Architecture

Both SMS and Email models use a Convolutional Neural Network (CNN).

```
Input Text
    │
Tokenizer
    │
Padding
    │
Embedding Layer
    │
Conv1D (128 filters)
    │
Global Max Pooling
    │
Dense (64 ReLU)
    │
Dropout (0.5)
    │
Dense (Sigmoid)
    │
Prediction
```

### Architecture

```python
Embedding(MAX_WORDS, 64)
Conv1D(128, 5, activation='relu')
GlobalMaxPooling1D()
Dense(64, activation='relu')
Dropout(0.5)
Dense(1, activation='sigmoid')
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/SpamShield-AI.git
cd SpamShield-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install scikit-learn
```

or

```bash
pip install -r requirements.txt
```

---

## Training the SMS Model

Place:

```
spam.csv
```

inside your dataset folder.

Run:

```bash
python sms_detector.py
```

Outputs:

```
sms_model/
sms_model.keras
tokenizer.json
```

---

## Training the Email Phishing Model

Place:

```
CEAS_08.csv
```

inside your dataset folder.

Run:

```bash
python email_detector.py
```

Outputs:

```
email_model/
email_tokenizer.json
```

---

## Testing SMS Messages

Example:

```python
predict_message(
    "Congratulations! You have won a free prize."
)
```

Output:

```
Probability: 0.98
Prediction: SPAM / PHISHING
```

---

## Testing Emails

Example:

```python
test_email(
    "Urgent account verification",
    "Your account has been suspended.",
    "http://fake-bank.com"
)
```

Output:

```
Probability: 0.97
Prediction: PHISHING
```

---

## TensorFlow Lite Conversion

Convert the email model to TensorFlow Lite:

```bash
python convert_to_tflite.py
```

Output:

```
email_model_select.tflite
```

This model can be integrated into:

* Android applications
* Embedded systems
* Edge AI devices
* Offline security tools

---

## Performance Evaluation

The project uses:

```python
classification_report()
```

Metrics reported:

* Accuracy
* Precision
* Recall
* F1-Score

The email model additionally handles dataset imbalance through:

```python
compute_class_weight()
```

which improves phishing detection performance.

---

## Future Improvements

* Bidirectional LSTM architecture
* Transformer-based classifiers (BERT)
* URL feature engineering
* Real-time email scanning
* Mobile application integration
* Multi-language support
* Explainable AI predictions
* Ensemble learning approaches

---

## Applications

* SMS spam filtering
* Email phishing detection
* Enterprise email security
* Mobile cybersecurity
* Fraud detection systems
* Secure communication platforms

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-Learn
* TensorFlow Lite

---

## License

This project is released under the MIT License.

Feel free to use, modify, and distribute it for educational and research purposes.

---

## Author

Developed as a Deep Learning and Cybersecurity project focused on spam and phishing detection using Convolutional Neural Networks.
