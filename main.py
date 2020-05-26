#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
import random

model = Sequential()

model.add(Convolution2D(filters=32,
                        kernel_size=(2,2),
                        activation='relu',
                        input_shape=(64,64,3)
                       ))
#tweak1

model.add(MaxPooling2D(pool_size=(2, 2)))

#tweak2

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

#tweak3

model.add(Dense(units=1,activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/fold1/cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/fold1/cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

with open('/fold1/epoch.txt') as f:
  epochs=int(f.readline())
  
out=model.fit(
        training_set,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=800)

accuracy=out.history['accuracy'][-1] *100
print("Accuracy for the model is : " , accuracy ,"%")

f= open("/fold1/accuracy.txt","w+")
f.write(str(accuracy))
f.close()

f= open("epoch.txt","w+")
f.write(str(epochs))
f.close()

'''import matplotlib.pyplot as plt
loss = out.history['loss']
loss_val = out.history['val_loss']
epo= epochs
epo = range(1,epochs+1)
plt.plot(epo, loss, 'g', label='Training loss',marker='o')
plt.plot(epo, loss_val, 'b', label='validation loss',marker='o')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()'''

# In[ ]:




