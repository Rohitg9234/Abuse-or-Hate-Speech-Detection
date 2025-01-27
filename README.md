# Abuse-or-Hate-Speech-Detection Detection

This project aims to build a machine learning model capable of classifying tweets into three categories: **Hate Speech**, **Offensive Language**, and **Neither**. The approach uses natural language processing (NLP) techniques to clean the data, extract meaningful features, and train a neural network model. The model leverages **LSTM (Long Short-Term Memory)** networks, which are effective for text classification tasks.

---

### Table of Contents

1. [Dataset Collection](#1-dataset-collection)
2. [Data Cleaning](#2-data-cleaning)
   - [Remove Special Characters, Punctuation, and Emojis](#21-remove-special-characters-punctuation-and-emojis)
   - [Remove Stop Words](#22-remove-stop-words)
   - [Normalization: Lowercasing](#23-normalization-lowercasing)
   - [Stemming/Lemmatization](#24-stemminglemmatization)
   - [Handle URLs](#25-handle-urls)
   - [Check for Empty/Null Tweets](#26-check-for-emptynull-tweets)
3. [Exploratory Data Analysis (EDA)](#3-eda)
   - [Class Distribution](#31-class-distribution)
   - [Most Common Words per Class](#32-most-common-wordsper-class)
   - [Word Cloud Visualization](#33-word-cloud-visualization)
4. [Modeling](#4-modeling)
   - [Tokenization](#41-tokenization)
   - [Padding](#42-padding)
   - [Evaluation Metrics](#43-evaluation-metrics)
   - [Model Architecture](#44-model-architecture)
   - [Training the Model](#45-training-the-model)
5. [Loss Curve Visualization](#5-loss-curve-visualization)

---

### 1. Dataset Collection

The dataset used in this project is a CSV file containing tweets. Each tweet is labeled with one of three classes:

- **0:** Hate Speech
- **1:** Offensive Language
- **2:** Neither

```python
df = pd.read_csv('/content/drive/MyDrive/Data/train.csv')
df.head(10)
```

### 2. Data Cleaning

Before training the model, the dataset is cleaned to ensure that the input data is in a suitable format for analysis.

#### 2.1 Remove Special Characters, Punctuation, and Emojis

We remove unnecessary special characters, URLs, and emojis that don't contribute to the sentiment or meaning of the tweet. This ensures that the model doesn't learn from irrelevant information.

```python
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text
```

#### 2.2 Remove Stop Words

Stop words like "the", "a", "and", etc., are common in language but don't contribute much meaning to the context of a tweet. These are removed to improve the performance of the model.

```python
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)
```

#### 2.3 Normalization: Lowercasing

Converting all text to lowercase ensures that words like "Apple" and "apple" are treated as the same.

```python
df['tweet'] = df['tweet'].str.lower()
```

#### 2.4 Stemming/Lemmatization

Lemmatization reduces words to their base form (e.g., "better" becomes "good"), making the dataset more consistent and improving model performance.

```python
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return " ".join(lemmatized_text)
```

#### 2.5 Handle URLs

URLs are removed as they don’t provide useful information for the classification task.

```python
df['tweet'] = df['tweet'].str.replace(r'http\S+', '', regex=True)
```

#### 2.6 Check for Empty/Null Tweets

After cleaning, some tweets may become empty or too short to be useful. These are removed from the dataset.

```python
df = df[df['tweet'].str.strip().notna()]
```

---

### 3. Exploratory Data Analysis (EDA)

#### 3.1 Class Distribution

A count plot is generated to understand the distribution of the dataset across the three classes.

```python
sns.countplot(x='class', data=df)
```

#### 3.2 Most Common Words per Class

We filter tweets by class and tokenize them. Then, we extract the most frequent words from each category using a **CountVectorizer**.

```python
hate_speech_tweets = df[df['class'] == 0]['tweet']
offensive_language_tweets = df[df['class'] == 1]['tweet']
neither_tweets = df[df['class'] == 2]['tweet']

# Get the top 10 most common words for each class
hate_speech_top_words = get_most_common_words(hate_speech_tweets)
```

#### 3.3 Word Cloud Visualization

We create word clouds to visualize the most frequent words in each class. This helps identify key terms associated with hate speech, offensive language, and neutral content.

```python
plot_word_cloud(hate_speech_tokens, 'Word Cloud for Hate Speech')
```

---

### 4. Modeling

#### 4.1 Tokenization

The tweets are tokenized into sequences of integers. This allows the model to handle the data more effectively. A **Tokenizer** is used to convert the words into integer sequences.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
```

#### 4.2 Padding

Padding ensures that all input sequences are of the same length. This is crucial because neural networks require consistent input dimensions.

```python
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)
```

#### 4.3 Evaluation Metrics

We define custom evaluation metrics such as **precision**, **recall**, and **F1-score**. These metrics are particularly useful for imbalanced datasets.

```python
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_positives / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
```

#### 4.4 Model Architecture

The model is built using an **Embedding** layer to represent the words in a dense vector space, followed by an **LSTM** layer for sequence modeling, and **Dense** layers for classification.

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=200, input_length=max_length))
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
```

#### 4.5 Training the Model

The model is trained using the **Adam** optimizer and the **categorical cross-entropy** loss function. The model is validated on a test set during training.

```python
model_history = model.fit(
    X_train, y_train, 
    batch_size=64, 
    epochs=10, 
    validation_data=(X_test, y_test)
)
```

---

### 5. Loss Curve Visualization

To track the performance of the model during training, we visualize the loss curve, which shows how the model's error decreases over time.

```python
plt.plot(hist['loss'], 'r', linewidth=2, label='Training loss')
plt.plot(hist['val_loss'], 'g', linewidth=2, label='Validation loss')
```

---

### Conclusion

This project demonstrates the full pipeline for building a machine learning model to classify tweets into hate speech, offensive language, or neither. By using techniques such as text cleaning, tokenization, and LSTM-based classification, we can effectively preprocess the data and train a model for real-world applications. The final model can be evaluated using precision, recall, and F1-score, providing a comprehensive assessment of its performance.

---

### Dependencies

- **Python 3.x**
- **TensorFlow** for deep learning models
- **Keras** for model construction
- **NLTK** for natural language processing tasks (tokenization, stopwords removal, etc.)
- **Seaborn** and **Matplotlib** for data visualization
- **scikit-learn** for machine learning utilities like train-test split and vectorization
- **WordCloud** for visualizing word frequency

Install the required libraries by running:

```bash
pip install tensorflow nltk scikit-learn seaborn matplotlib wordcloud
```
