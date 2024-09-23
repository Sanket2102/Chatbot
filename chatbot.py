import numpy as np
import pickle

#Loading the pickle files
# tokenizer for one hot encoding
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# deep learning model
with open("model.pkl","rb") as file:
    model = pickle.load(file)

# prompt from user
prompt = input("Enter your text: ")
text = [prompt]

# text preprossesing
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
corpus = []

review = text[0].lower()
review = review.split()

review = [lemmatizer.lemmatize(word) for word in review]
review = ' '.join(review)
corpus.append(review)

# one hot encoding and padding
one_hot_repr = tokenizer.texts_to_sequences(text)

from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

sent_length = 15
padded_doc = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)

# model prediction
predictions = model.predict(padded_doc)

predicted_class = np.argmax(predictions, axis=1)
print(predicted_class)