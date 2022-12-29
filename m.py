import cv2
import numpy as np 
#from skimage.util import img_as_ubyte
#from skimage.color import rgb2gray
from keras.models import load_model

img = cv2.imread('image.jfif')

gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray.copy(), (5,5), 0)
crop = cv2.equalizeHist(blur)
crop = crop/255
cv2.imshow('gray', crop)
retir,thresh = cv2.threshold(blur.copy(),75 , 255, cv2.THRESH_BINARY_INV)
dilate = cv2.dilate(thresh.copy(), None, iterations = 2)
cv2.imshow('thresh',dilate)
_,contrs,_ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contr in contrs:
    x,y,w,h = cv2.boundingRect(contr)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    digit = crop[y:y+h,x:x+w]
    resize_digit = cv2.resize(digit, (28,28))
    #padded_digit = np.pad(resize_digit, ((5,5),(5,5)), "constant", constant_values = 0)
    result =  resize_digit.reshape((1,28,28,1))
    #result = result/255
    model = load_model("digit.h5")

    # predict digit
    prediction = model.predict(result)
    prob = np.amax(prediction)
    probv = prob * 100
    probval = "%.2f"%probv
    final = np.argmax(prediction)
    print(str(final)+"  "+str(probval))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(probval), (x,y), font, 1, (0,0,255))
    
cv2.imshow("frame", img)

if cv2.waitKey(0) & 0xFF == ord('s'):
    cv2.destroyAllWindows()
