# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

# Reading input of data
df = pd.read_csv('spam.csv', delimiter=',', encoding='latin-1')
df.head()

# Pre-processing data by filtering essential information resulting from dropping irrelevant data
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.info()

# Frequency count of data across various columns
count = df.groupby(['v1']).count()
print('The frequency of data belonging to each category:', count)

# Visualization of data distribution
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title("Number of ham versus spam messages using General Random Sampling")

# Identifying the specific indices associated with each class
hamData = df[df.v1 == 'ham'].index
print("Length of data associated to ham: ",len(hamData))
print(hamData)
spamData = df[df.v1 == 'spam'].index
print("Length of data assocaited to spam: ",len(spamData))
print(spamData)

# Accessing data based on the class it belongs to
dataHam = df[df['v1'] == 'ham']
print(dataHam)
dataSpam = df[df['v1'] == 'spam']
print(dataSpam)

# Random over-sampling of data
spamOver = dataSpam.sample(len(df[df['v1'] == hamData]))
dfOver = pd.concat([spamOver, dataHam], axis=0)

# Visualization of data distribution using under sampling
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title("Number of ham versus spam messages using Random Under Sampling")
print("No. of spam messages resulting from under sampling: ", len(dfOver['v1'] == spamData))
print("No. of ham messages resulting from under sampling: ", len(dfOver['v1'] == hamData))

# Identifying feature predictors and target variables
x = dfOver.v2
y = dfOver.v1

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
