import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset1= pd.read_csv("creditcardfraud/creditcard.csv")
# dataset1.shape = (284807, 31)
# check null values in dataset using dataset1.isnull().sum()
# no null values in this dataset
# observations in each class
# dataset1['Class'].value_counts() = Class 0: 284315, 1: 492
fraud= dataset1[dataset1['Class']==1]
non_fraud= dataset1[dataset1['Class']==0]
# fraud.shape= (492, 31), non_fraud.shape=(284315, 31)
# unbalanced dataset, so adjusting
non_fraud_t=non_fraud.sample(n=492)
# non_fraud_t.shape=(492, 31)
dataset=fraud._append(non_fraud_t, ignore_index=True)

x=dataset.drop(labels=['Class'], axis=1)
y=dataset['Class']
# x.shape=(984, 30), y.shape=(984,)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,
                                                   random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
# x_train.shape= (787, 30), x_test.shape= (197, 30)
x_train=x_train.reshape(787, 30, 1)
x_test=x_test.reshape(197, 30, 1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same',
                                 activation='relu', input_shape=(30,1)))
model.add(tf.keras.layers.BatchNormalization())
# batch normalisation is used to find internal covariance shift problem  to
# normalise it to increase its speed, performance, efficiency and stability
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
# maxpool layer selects the max value of a batch
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=25,validation_data=(x_test, y_test))
y_pred=(model.predict(x_test)>0.5).astype('int32')
print(y_pred[0], y_test[0])
print(y_pred[10], y_test[10])
print(y_pred[-3], y_test[-3])

cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)

epoch_range=range(1,26)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()
