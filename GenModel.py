import numpy as np
from keras import models
from keras import layers
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report
from keras.utils import to_categorical
import pickle 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
# from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import cycle

with open('Train_dataset', 'rb') as f:
    dataset = pickle.load(f)

with open('Train_labels', 'rb') as f:
    labels = pickle.load(f)


Train_samples = np.array(dataset).astype('float32')
train_labels = np.array(labels)
train_labels = train_labels.reshape((train_labels.shape[0],1))

train_labels = to_categorical(train_labels, 26)


X_train, X_val, y_train, y_val = train_test_split(Train_samples, train_labels, test_size = 1/3, random_state = 0)

#Normalizing the dataset
pca = PCA(n_components=30, whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_val_pca = pca.transform(X_val)



def buildFinalModel():
    network = models.Sequential()
    network.add(layers.Dense(units=32,activation='relu',input_shape=(X_train_pca.shape[1], )))
    network.add(layers.BatchNormalization())
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(units=64, activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=128, activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(units=26, activation='softmax'))

    network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return network



network = buildFinalModel()
history = network.fit(X_train_pca, y_train, epochs=200, validation_data=(X_val_pca, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['acc']
val_acc = history.history['val_acc']


epochs = range(1, len(acc) + 1)

# Validation vs Training in accuracy
plt.plot(epochs, acc, label='training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()

plt.show()


with open('Test_dataset', 'rb') as f:
    test = pickle.load(f)

with open('Test_labels', 'rb') as f:
    test_y = pickle.load(f)

# print("test: ",test)
test_x = np.array(test).astype('float32')
test_y = np.array(test_y)

# print("test x: ",test_x)
new_test_x = []
new_test_y = []

for i, j in zip(test_x, test_y):
    if j != 17:
        new_test_x.append(i)
        new_test_y.append(j)
new_test_x = np.array(new_test_x)
new_test_y = np.array(new_test_y)


new_test_y = new_test_y.reshape((new_test_y.shape[0],1))
cat_test_y = to_categorical(new_test_y, 26)

# pca = PCA(n_components=100, whiten=True).fit(X_train)
pca_test_x = pca.transform(new_test_x)

network.save("models/model9.h5")
loaded_model = load_model("models/model9.h5")
result = loaded_model.evaluate(pca_test_x, cat_test_y)

print("Test Accuracy: ", result[1]*100)

