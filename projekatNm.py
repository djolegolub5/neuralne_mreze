
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)


import numpy as np
import matplotlib.pyplot as plt
import cv2

main_path = './dataset/'
img_size = (64, 64)
batch_size = 64




from keras.utils import image_dataset_from_directory
Xtrain = image_dataset_from_directory(main_path,
 subset='training',
 validation_split=0.2,
 image_size=img_size,
 batch_size=batch_size,
 seed=123)
Xval = image_dataset_from_directory(main_path,
 subset='validation',
 validation_split=0.2,
 image_size=img_size,
 batch_size=batch_size,
 seed=123)
classes = Xtrain.class_names
print(classes)

x=['rabbit','cat']
y=[1012,1017]

plt.bar(x,y)

max_ylim = max(y) + 100
min_ylim = 0
plt.ylim(min_ylim, max_ylim)
plt.grid()
plt.show()

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')



    plt.show() #OVO TREBA





from keras import layers
from keras import Sequential
data_augmentation = Sequential(
 [
 layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
 layers.RandomRotation(0.25),
 layers.RandomZoom(0.1),
 ]
)



N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
    plt.show()

from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
num_classes = len(classes)
model = Sequential([
 data_augmentation,
 layers.Rescaling(1./255, input_shape=(64, 64, 3)),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Dropout(0.2),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(num_classes, activation='softmax')
])


model.summary()
model.compile(Adam(learning_rate=0.001),
 loss=SparseCategoricalCrossentropy(),
 metrics='accuracy')




history = model.fit(Xtrain,
 epochs=50,
 validation_data=Xval,
 verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()



labelsVal = np.array([])
predVal = np.array([])
listaPogresnihSlika=[]
listaTacnihSlika=[]
listaPogresnihLabela=[]
listaTacnihLabela=[]


flagTrue=False
flagFalse=False

for img, lab in Xval:
    labelsVal = np.append(labelsVal, lab)


    prediction=np.argmax(model.predict(img, verbose=0), axis=1)
    predVal = np.append(predVal, prediction)

    for j in range(len(lab)):
        if (flagFalse!=True or flagTrue!=True):
            if (prediction[j]!=lab[j] and flagFalse==False):
                imgFalse=img
                jFalseLabela=prediction[j]
                jFalseSlika=j
                flagFalse=True



            elif (flagTrue==False and prediction[j]==lab[j]):
                imgTrue=img
                jTrueSlika=j
                jTrueLabela=prediction[j]
                flagTrue=True





from sklearn.metrics import accuracy_score
print('Tačnost modela na test skupu je: ' + str(100*accuracy_score(labelsVal, predVal)) + '%')



labelsTrain = np.array([])
predTrain = np.array([])
for img, lab in Xtrain:
    labelsTrain = np.append(labelsTrain, lab)
    predTrain = np.append(predTrain, np.argmax(model.predict(img, verbose=0), axis=1))


print('Tačnost modela na trening skupu je: ' + str(100*accuracy_score(labelsTrain, predTrain)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labelsVal, predVal, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()



cm = confusion_matrix(labelsTrain, predTrain, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()







plt.figure()
plt.imshow(imgFalse[jFalseSlika].numpy().astype('uint8'))
plt.title(classes[jFalseLabela])
plt.show()
plt.figure()
plt.imshow(imgTrue[jTrueSlika].numpy().astype('uint8'))
plt.title(classes[jTrueLabela])
plt.show()


