#!/usr/bin/env python
# coding: utf-8

# In[56]:


from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam ,RMSprop ,SGD ,Nadam ,Adamax
from keras.models import Sequential
import random


# In[57]:


model = Sequential()


# In[58]:


model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu',
                        input_shape=(64,64,3)
                       ))


# In[59]:


model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))


# In[60]:


print(model.summary())


# In[61]:


def architecture(option):
    if option == 1:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
    elif option == 2:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
    elif option == 3:
        #two convolutional and 2 max pooling layers
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
        
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))
    elif option == 4:
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(Convolution2D(filters=random.randint(30,60),
                        kernel_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6))),
                        activation='relu'
                       ))
        model.add(MaxPool2D(pool_size=random.choice(((2,2),(3,3),(4,4),(5,5),(6,6)))))


# In[62]:


architecture(random.randint(1,4))


# In[63]:


print(model.summary())


# In[64]:


model.add(Flatten())


# In[65]:


def fullyconnected(option):
    if option == 1:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 2:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 3:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
    elif option == 4:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        
    else:
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))
        model.add(Dense(units=random.randint(80,200),activation=random.choice(('relu','sigmoid','softmax'))))


# In[66]:


fullyconnected(random.randint(1,5))


# In[67]:


print(model.summary())


# In[68]:


model.add(Dense(units=1,activation='sigmoid'))


# In[69]:


print(model.summary())


# In[70]:


model.compile(optimizer=random.choice((RMSprop(lr=.0001),Adam(lr=.0001),SGD(lr=.001),Nadam(lr=.001),Adamax(lr=.001))),loss='binary_crossentropy',metrics=['accuracy'])


# In[71]:


from keras.preprocessing.image import ImageDataGenerator


# In[76]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
out=model.fit(
        training_set,
        steps_per_epoch=50,
        epochs=1,
        validation_data=test_set,
        validation_steps=28)


# In[78]:


out.history
print(out.history['accuracy'][0])


# In[79]:


mod =str(model.layers)
accuracy = str(out.history['accuracy'][0])







