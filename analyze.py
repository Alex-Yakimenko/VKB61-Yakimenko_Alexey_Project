# Импорт необходимых библиотек/модулей
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import random
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.utils import shuffle

# Загрузка данных для обучения и пустые массивы
img_path = 'C:/Users/Demon777/Desktop/подборка/Cyrillic/'
X_train = []
y_train = []
X_test = []
y_test = []

os.chdir(img_path)
dirs = os.listdir()
class_dot = 0
for d in dirs:
    print(d, end=' ')
    os.chdir(os.path.join(img_path, d))
    index = 0
    os.chdir(os.path.join(img_path, d))

    # Дополнительная аугментация изображений с помощью: поворотов, смещения, уменьшения изображений
    files = os.listdir()
    for f in files:
        for way in ['decrease', 'rotate+', 'rotate-', 'shift']:
            # Уменьшение
            if way == 'decrease':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))  # (255, 255, 255) (145, 145, 145)
                res_img.paste(img, mask=img.split()[3])
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)

            # Поворот по часовой стрелке
            elif way == 'rotate+':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                angle = random.randint(0, 30)
                res_img = res_img.rotate(angle, fillcolor='white')
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)

            # Поворот против часовой стрелки
            elif way == 'rotate-':
                img = Image.open(f)
                res_img = res_img.resize((32, 32))
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                angle = random.randint(-30, 0)
                res_img = res_img.rotate(angle, fillcolor='white')
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)

            # Смещение
            elif way == 'shift':
                img = Image.open(f)
                res_img = Image.new("RGB", img.size, (255, 255, 255))
                res_img.paste(img, mask=img.split()[3])
                res_img = res_img.rotate(0, fillcolor='white')
                horizontal, vertical = random.randint(-5, 5), random.randint(-5, 5)
                res_img = res_img.resize((32, 32))
                img_arr = np.array(res_img)

            # Разделение изображений на тестовые и тренировочные наборы
            if index >= round(len(files) / 100 * 85):
                X_test.append(img_arr)
                y_test.append([class_dot])

            else:
                X_train.append(img_arr)
                y_train.append([class_dot])

        index += 1

    class_dot += 1

# Разделение данных на тренировочные и тестовые выборки
X_train = np.array(X_train, dtype='float32')
y_train = np.array(y_train, dtype='uint8')
X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test, dtype='uint8')

Y_train = to_categorical(y_train, 33)
Y_test = to_categorical(y_test, 33)

# Нормализация данных путем деления
X_train = X_train / 255
# Перемешивание тестовой выборки
X_test, Y_test = shuffle(X_test / 255, Y_test)

print('\nНаборы данных успешно аугментированы и разделены')
print('Всего тренировочных данных:')
print(len(X_train))
print('Всего тестовых данных:')
print(len(X_test))

# Архитектура модели
model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(33, activation='softmax'))

# Настройка обучения методом compile
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Обучение модели методом fit
model.fit(X_train, Y_train,batch_size=80, epochs=5, validation_data=(X_test, Y_test),shuffle=True)

# Сохранение модели нейронной сети для дальнейшего использования
os.chdir(r'C:/Users/Demon777/Desktop/подборка/')
model.save(f'model_CoMNIST.keras')
print('Модель успешно обучена и сохранена')
