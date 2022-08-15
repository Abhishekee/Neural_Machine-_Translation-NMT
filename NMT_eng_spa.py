#!/usr/bin/env python
# coding: utf-8

# In[126]:


get_ipython().system('pip install graphviz')


# In[158]:


# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# In[159]:


# Values of Different parametes
BATCH_SIZE = 64
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 30000
MAX_SENTENCE_LENGTH = 100
MAX_NUM_WORDS = 10000
EMBEDDING_SIZE = 200


# # Data Preprocessing 

# In[248]:


input_sentences = []
output_sentences = []
output_sentences_input = []

counter = 0
for line in open(r"C:\Users\aplus\NMT\spa.txt", encoding= "utf-8"):
    counter += 1
    if counter > NUM_SENTENCES:
        break
    if "\t" not in line:
        continue
    input_sentence = line.rstrip().split('\t')[0]    
    output_part = line.rstrip().split('\t')[1]

#adding eos and sos in the output part
    output_sentence = output_part + '<eos>'
    output_sentence_input = '<sos>' + output_part

#appending into respective lists
    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    output_sentences_input.append(output_sentence_input)


# In[249]:


#printing the lists
print("No. of input are:", len(input_sentences))
print("No. of ouput are:", len(output_sentences))
print("No. of output-input are:", len(output_sentences_input))


# ### Printing Sentences in English and their Spanish Translations

# In[250]:


print("English Sentence is:", input_sentences[250])
print("Spanish Translation is:", output_sentences[250])

print("English Sentence is:", input_sentences[2654])
print("Spanish Translation is:", output_sentences[2654])

print("English Sentence is:", input_sentences[10000])
print("Spanish Translation is:", output_sentences[10000])


# ## Data Visualization 

# In[251]:


# creating lists for storing lengths of english and spanish
len_eng = []
len_spa = []

# appending sentences lengths
for i in input_sentences:
    len_eng.append(len(i.split()))
    
for i in output_sentences:
    len_spa.append(len(i.split()))
    
len_df = pd.DataFrame({'English':len_eng, 'Spanish':len_spa})
len_df.hist(bins = 15, color = 'green')
plt.show


# ## Tokenization
# 

# In[252]:


# tokenize the english(input) sentences
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
max_len_input = max(len(sent) for sent in input_integer_seq)

#printing length of max sentence and number of unique words
print(input_integer_seq)
print('No. of Unique Words are:', len(word2idx_inputs))
print('Max of length of sentence is:', max_len_input)


# In[253]:


word2idx_inputs.items()


# In[300]:


# tokenizing the spanish(Output) sentnces
output_tokenizer = Tokenizer(num_words = MAX_NUM_WORDS, filters = '')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_input)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_input)

word2idx_outputs = output_tokenizer.word_index
max_len_out = max(len(sent) for sent in output_integer_seq)
max_num_words_output = len(word2idx_outputs) + 1

#printing length of max sentence and number of unique words
print(output_integer_seq)
print('No. of Unique Words are:', len(word2idx_outputs))
print('Max of length of sentence is:', max_len_out)
print(max_num_words_output)


# In[302]:


print(output_tokenizer.word_index)


# ## Word Embeddings and Padding

# In[167]:


# padding input integer sequence
encoder_input_seq = pad_sequences(input_integer_seq, maxlen = max_len_input, padding = 'pre')
print('Shape of encoder_input_seq:', encoder_input_seq.shape)
print('encoder_input_Seq[10000]:', encoder_input_seq[10000])

print(word2idx_inputs["want"])
print(word2idx_inputs["to"])
print(word2idx_inputs["leave"])


# In[168]:


# padding output-input integer sequence
decoder_input_seq = pad_sequences(output_input_integer_seq, maxlen = max_len_out, padding = 'post')
print("decoder_input_seq_shape:", decoder_input_seq.shape)
print("decoder_input_seq[10000]:", decoder_input_seq[10000])

print(word2idx_outputs['quiero'])


# In[169]:


# padding output integer sequence
decoder_output_seq = pad_sequences(output_integer_seq, maxlen = max_len_out, padding = 'post')
print("decoder_output_seq_shape:", decoder_output_seq.shape)


# In[170]:


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove = open(r"E:\GloVe\glove.twitter.27B.200d.txt", encoding="utf-8")

for line in glove:
    split = line.split()
    word = split[0]
    vector_dimensions = asarray(split[1:], dtype = 'float32')
    embeddings_dictionary[word] = vector_dimensions
glove.close()


# In[179]:


embeddings_dictionary['want']


# In[180]:


num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
print(embedding_matrix)
print('Shape of embedding Matrix is:',embedding_matrix.shape)


# In[181]:


for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
embedding_vector


# In[182]:


# creating embedding layer
embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights = [embedding_matrix], input_length = max_len_input)

# onehot representation
decoder_one_hot = np.zeros((len(input_sentences), max_len_out, max_num_words_output), dtype = 'uint8')
decoder_one_hot.shape


# In[198]:


for i, d in enumerate(decoder_output_seq):
    for t, word in enumerate(d):
        decoder_one_hot[i, t, word] = 1
decoder_one_hot


# ## CREATING THE MODEL

# In[184]:


# Encoder LSTM Model
encoder_inputs = Input(shape = (max_len_input,))
a = embedding_layer(encoder_inputs)
encoder = LSTM(LSTM_NODES, return_state = True)
o_e,h_e,c_e = encoder(a)
encoder_states = [h_e,c_e]


# In[319]:


a


# In[185]:


# Decoder LSTM Model
decoder_inputs = Input(shape = (max_len_out,))
decoder_embedding = Embedding(max_num_words_output, LSTM_NODES)
b = decoder_embedding(decoder_inputs)
decoder = LSTM(LSTM_NODES, return_state = True, return_sequences = True)
o_d,h_d,c_d =decoder(b, initial_state = encoder_states) 


# In[178]:


# Dense Model 


# In[189]:


decoder_dense = Dense(25413, activation = 'sigmoid')
decoder_outputs = decoder_dense(o_d)


# ## Compiling The Model

# In[199]:


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()


# In[200]:


#from keras.callbacks import EarlyStopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#fitting the model
history = model.fit([encoder_input_seq, decoder_input_seq], decoder_one_hot,batch_size = BATCH_SIZE, epochs = 5,validation_split = 0.2)


# In[201]:


model.save('eng_spa.h5')


# In[202]:



import matplotlib.pyplot as plt
# %matplotlib inline
plt.title('Model Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[203]:


plt.title('model accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[205]:


encoder_model = Model(encoder_inputs, encoder_states)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights('eng_spa.h5')


# In[206]:


decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)


# In[280]:


decoder_outputs, h, c = decoder(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
decoder_states

