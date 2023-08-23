from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import cv2;

image_list = []
label_list = []

for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/train/" + dir 
    label = 0

    if dir == "chiikawa":
        label = 0
    elif dir == "hachiware":
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = image.transpose(2, 0, 1)
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            image_list.append(image / 255.)

image_list = np.array(image_list)

Y = to_categorical(label_list)

model = Sequential()
model.add(Dense(200, input_dim=1875))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

opt = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(image_list, Y, epochs=300, batch_size=100, validation_split=0.1)

total = 0.
ok_count = 0.

print("-+-+-+-+-+-+-+-+-+-+-+ testing +-+-+-+-+-+-+-+-+-+-")
for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/test/" + dir 
    label = 0

    if dir == "chiikawa":
        label = 0
    elif dir == "hachiware":
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            image = image.transpose(2, 0, 1)
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            predict = model.predict(np.array([image / 255.]))
            result = np.argmax(predict, axis=-1)
            print("expectations:", label, "result:", result[0])
            per_chiikawa = round(predict[0][0] * 100, 3)
            per_hachiware = round(predict[0][1] * 100, 3) 
            print("percentage: chiikawa:", per_chiikawa,'%', ' hachiware:', per_hachiware,'%')

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("-+-+-+-+-+-+-+-+-+-+-+ result +-+-+-+-+-+-+-+-+-+-")
print("count ok:", ok_count ," total:", total)
print("Correct answer rate: ", ok_count / total * 100, "%")