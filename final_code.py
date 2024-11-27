import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,MaxPooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, r2_score
import seaborn as sns
from tensorflow.keras import regularizers

path = "archive/UTKFace"
images = []
age = []
gender = []
for img in os.listdir(path):
  ages = img.split("_")[0]
  genders = img.split("_")[1]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  images.append(np.array(img))
  age.append(np.array(ages))
  gender.append(np.array(genders))
  
age = np.array(age,dtype=np.int64)
images = np.array(images)
gender = np.array(gender,np.uint64)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)


#Define age model and train. 
def create_age_model():
  age_model = Sequential()
  age_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200,200,3)))
  age_model.add(BatchNormalization())
  age_model.add(MaxPooling2D((2,2)))
  age_model.add(Conv2D(64, (3, 3), activation='relu'))
  age_model.add(BatchNormalization())
  age_model.add(MaxPooling2D((2, 2)))
  age_model.add(Flatten())
  age_model.add(Dropout(0.2))
  age_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  age_model.add(Dense(1, activation='linear', name='age'))

  age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  print(age_model.summary())

  age_model.fit(x_train_age, y_train_age, validation_data=(x_test_age, y_test_age), epochs=30)

  age_model.save('age_model_30epochs.keras')


#Define gender model and train
def create_gender_model():
  gender_model = Sequential()
  gender_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200,200,3)))
  gender_model.add(BatchNormalization())
  gender_model.add(MaxPooling2D((2,2)))
  gender_model.add(Conv2D(64, (3, 3), activation='relu'))
  gender_model.add(BatchNormalization())
  gender_model.add(MaxPooling2D((2, 2)))
  gender_model.add(Flatten())
  gender_model.add(Dropout(0.2))
  gender_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  gender_model.add(Dense(1, activation='sigmoid', name='gender'))

  gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  print(gender_model.summary())

  gender_model.fit(x_train_gender, y_train_gender, validation_data=(x_test_gender, y_test_gender), epochs=30)

  gender_model.save('gender_model_30epochs.keras')


def test_age_model(age_model):
  # test age model
  age_predictions = age_model.predict(x_test_age)
  print("Age Model Mean Absolute Error = ", metrics.mean_absolute_error(y_test_age, age_predictions))
  print("Age Model R-Squared = ", r2_score(y_test_age, age_predictions))


def test_gender_model(gender_model):
  # test gender model
  gender_predictions = gender_model.predict(x_test_gender)
  y_pred_gender = (gender_predictions>= 0.5).astype(int)[:,0]

  print("Gender Model Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred_gender))


def prediction_with_test_data(test_img, age_model, gender_model):
  # match input shape of model
  test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
  test_img_resized = cv2.resize(test_img, (200, 200))
  test_imgs = []
  test_imgs.append(np.array(test_img_resized))
  test_imgs = np.array(test_imgs)

  # predict
  print("\nGround Truth: ( 1 , 33)")
  age_prediction = round(age_model.predict(test_imgs)[0][0])
  gender_prediction = round(gender_model.predict(test_imgs)[0][0])
  print("Prediction: (" , gender_prediction, " ,", age_prediction, " )")

  plt.imshow(test_img_resized)
  plt.show()


if __name__ == '__main__':
  # create_age_model()
  # create_gender_model()

  age_model = load_model('age_model_30epochs.keras', compile=False)
  test_age_model(age_model)
  gender_model = load_model('gender_model_30epochs.keras', compile=False)
  test_gender_model(gender_model)

  test_img = cv2.imread('custom_dataset/woman3.jpg')
  prediction_with_test_data(test_img, age_model, gender_model)