# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Reading input of data
df = pd.read_csv('spam.csv', delimiter=',', encoding='latin-1')
df.head()

# Pre-processing data by filtering essential information resulting from dropping irrelevant data
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.info()

# Visualization of data distribution
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title("Number of ham versus spam messages using Random Sampling")

# Identifying feature predictors and target variables
x = df.v2
y = df.v1

# Label encoding of the data
le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1,1)

# Split the data into training and test data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# Tokenization of data
max_words = 1000
max_len = 150
token = Tokenizer(num_words=max_words)
token.fit_on_texts(x_train)
sequences = token.texts_to_sequences(x_train)
sequenceMatrix = sequence.pad_sequences(sequences, maxlen=max_len)

# Define RNN Structure
def RNN():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model

# Define the model to be used
model = RNN()
print("Summary of the RNN model used:")
model.summary()

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Fit model using training data set
model.fit(sequenceMatrix, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

# Evaluate the model's performance using test data set
testSequences = token.texts_to_sequences(x_test)
test_sequenceMatrix = sequence.pad_sequences(testSequences, maxlen=max_len)

# Compute the accuracy of the model
accuracy = model.evaluate(test_sequenceMatrix, y_test)
print('Test set\n Loss: {:0.3}\n Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))

