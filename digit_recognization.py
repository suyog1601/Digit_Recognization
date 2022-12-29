import keras
from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
from keras.models import load_model
import keras.layers as layers
from keras.layers import Dropout
from keras import regularizers
from keras.optimizers import rmsprop,Adam
import cv2
import tensorflow as tf
import numpy as np
import numpy.random as nr

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

print(train_images.shape)
print(train_images.dtype)
print(test_images.shape)
print(test_images.dtype)

train_images = train_images.reshape((60000,28,28,1)).astype('float')/255
print(train_images.shape)
print(train_images.dtype)
test_images = test_images.reshape((10000,28,28,1)).astype('float')/255
print(test_images.shape)
print(test_images.dtype)

train_labels = ku.to_categorical(train_labels)
print(train_labels[5:,])

test_labels = ku.to_categorical(test_labels)
print(test_labels[5:,])

filepath = 'digit.h5'
callbacks_list = [
    keras.callbacks.EarlyStopping
    (
        monitor = 'val_loss',
        patience = 1
    ),
    keras.callbacks.ModelCheckpoint
    (
        filepath = filepath,
        monitor = 'val_loss',
        save_best_only = True
    )
]

nn = models.Sequential()

nn.add(layers.Conv2D(32, (3,3) ,padding = 'same', activation = 'relu',input_shape = (28,28,1)))
nn.add(layers.MaxPooling2D((2,2)))
nn.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
nn.add(layers.MaxPooling2D((2,2)))
nn.add(layers.Conv2D(64,(3,3), padding = 'same', activation = 'relu'))

nn.add(layers.Flatten())

nn.add(layers.Dense(64, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
nn.add(Dropout(0.5))
nn.add(layers.Dense(10, activation = 'softmax'))

nn.summary()

nr.seed(2323)
tf.random.set_seed(4466)

nn.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

nr.seed(88776)
tf.random.set_seed(5666)

nn.fit(train_images, train_labels, epochs = 40, batch_size = 128, validation_data = (test_images, test_labels), callbacks = callbacks_list, verbose =1 )
score = nn.evaluate(test_images, test_labels, verbose = 0)

print('test_loss:', score[0])
print('test_accuracy:', score[1])

    
img = cv2.imread('image.jfif')


gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray.copy(), (5,5), 0)
retir,thresh = cv2.threshold(blur.copy(), 75, 255, cv2.THRESH_BINARY_INV)
dilate = cv2.dilate(thresh.copy(), None, iterations = 2)
_,contrs,_ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contr in contrs:
    x,y,w,h = cv2.boundingRect(contr)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    digit = thresh[y:y+h,x:x+w]
    resize_digit = cv2.resize(digit, (28,28))
    padded_digit = np.pad(resize_digit, ((5,5),(5,5)), "constant", constant_values = 0)
    result =  padded_digit.reshape((1,28,28,1))
    result = result/255
    model = load_model("digit.h5")

    # predict digit
    prediction = model.predict(result)
    prob = np.amax(prediction)
    probv = prob * 100
    probval = "%.2f"%probv
    final = np.argmax(prediction)
    print(str(final)+"  "+str(probval))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,str(probval), (x,y), font, 1, (0,0,255))
    
cv2.imshow("frame", img)

if cv2.waitKey(0) & 0xFF == ord('s'):
    cv2.destroyAllWindows()

