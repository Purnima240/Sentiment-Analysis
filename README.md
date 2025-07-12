# Sentiment-Analysis
# Sentiment Analysis using LSTM and Word2Vec

This project implements a binary sentiment classification model using a stacked LSTM architecture trained on preprocessed review data. The model classifies reviews into **Positive** or **Negative** categories.

---

##  Preprocessing Pipeline

Each review was cleaned and processed using the following steps:

- Lowercased all text  
- Removed punctuation  
- Tokenized text (split into individual words)  
- Removed non-vocabulary words (Word2Vec trained words only)  
- Converted tokens into Word2Vec vectors (300-dimensional)
- Applied padding to sequences (max length = 150)  

---

##  Model Architecture

The model is a **stacked LSTM** using Keras:

```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(150, 300)))
model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
The LSTM model gave a accuracy of  86%.
