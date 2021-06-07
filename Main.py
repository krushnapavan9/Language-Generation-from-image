import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint




text = (open("dataset/dataset2.txt").read())
text=text.lower()

text=text.split('\n')
lenght=len(text)
sentences=[]
for  i  in range(lenght):
    temp=text[i].split("\t")
    if len(temp)>1:

        sentences.append(temp[1])
sentences=sentences[0:3000]
input_data_set=[]
words=[]

for i in sentences:
    temp=[]
    i=i[:-1]
    temp=i.split(" ")
    temp[-1]="."
    input_data_set.append(temp)
lengt2=len(input_data_set)
for i in range(lengt2):
    for j in range(len(input_data_set[i])):
        words.append(input_data_set[i][j])

words_set= list(set(words))
words_set.sort()
words_set.insert(0,' ')


num2word = {n:char for n, char in enumerate(words_set)}

word2num = {char:n for n, char in enumerate(words_set)}



X=[]
Y=[]
length=len(temp)
sequence_length=5

for sentence in input_data_set:
    length4=len(sentence)
    
    for i in range(length4):
        
        displacement=abs(sequence_length-(1))

        j=0
        for k in range(sequence_length):
            input_train=[]

            
            if not (abs(displacement)<0):

                for j in range(abs(displacement)):
                    input_train.append(word2num[' '])
            j=0
            


            while (len(input_train))!=sequence_length:
                
                if i+j >=length4:
                    j+=1
                    break
                input_train.append(word2num[sentence[i+j]])
                j+=1
                
            displacement-=1


            if len(input_train)==sequence_length:

                X.append(input_train)
                
                
                if (i+j)>=length4:
                    X.pop()
                else:
                    Y.append(word2num[sentence[i+j]])

for i in range(length-sequence_length):
    sequence=temp[i:i+sequence_length]
    label=temp[i+sequence_length]
    X.append([word2num[word] for word in sequence])
    Y.append(word2num[label])



X_modified = np.reshape(X, (len(X), sequence_length, 1))
X_modified = X_modified / float(len(words_set))
Y_modified = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam')

#filepath="checkpoint_model_{epoch:02d}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='auto',period=5)
#callbacks_list = [checkpoint]


#model.fit(X_modified, Y_modified, validation_split=0.33, epochs=100, batch_size=64, callbacks=callbacks_list, verbose=1)
#model.fit(X_modified, Y_modified, epochs=100, batch_size=32)


#model.save_weights('flicker.h5')

model.load_weights('checkpoints/checkpoint_model_100.hdf5')


string_mapped = X[5]
while True:
    strings=input().split(' ')
    
    while (len(strings)<sequence_length):
        strings.insert(0,' ')
    if len(strings)>sequence_length:
        strings=strings[-4:]
    full_string=strings
    string_mapped=[]

    for i in range(sequence_length):
        string_mapped.append(word2num[strings[i]])

    # generating characters
    for i in range(20):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(words_set))
        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [num2word[value] for value in string_mapped]
        full_string.append(num2word[pred_index])

        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]
    txt=""
    for char in full_string:
        txt = txt+" "+ char
        if char==".":
            break
    print("Output :")
    print(txt)  
