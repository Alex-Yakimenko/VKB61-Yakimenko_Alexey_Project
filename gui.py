# Импорт необходимых библиотек/модулей

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from PIL import Image
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPen, QFont
from PyQt5.QtWidgets import QPushButton, QMainWindow, QLabel, QLineEdit, QApplication, QFileDialog
import keras
import tensorflow as tf
# Установка переменной окружения QT_QPA_PLATFORM_PLUGIN_PATH для работы с плагином PyQt5
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'venv/Lib/site-packages/PyQt5/Qt5/plugins/'


class Window(QMainWindow):
    # Графический интерфейс
    def __init__(self):
        super().__init__()

        # Наименование и размерность окна
        title = "Распознавание буквы кириллицы"
        top = 200
        left = 200
        width = 540
        height = 340

        # Шрифты
        font = QFont('Arial', 10)
        fontBig = QFont('Arial', 12)

        # Кисть для рисования
        self.drawing = False
        self.brushSize = 8
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        # Холст
        self.image = QImage(278, 278, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        # Лейбл "результат"
        label = QLabel('РЕЗУЛЬТАТ:', self)
        label.setFont(font)
        label.move(290, 140)

        # Размерность и расположение поля "результат"
        self.line = QLineEdit(self)
        self.line.setFont(fontBig)
        self.line.move(370, 134)
        self.line.resize(99, 42)

        # Кнопка "распознать"
        recognize_button = QPushButton('РАСПОЗНАТЬ', self)
        recognize_button.setFont(font)
        recognize_button.move(290, 30)
        recognize_button.resize(230, 33)
        recognize_button.clicked.connect(self.save)
        recognize_button.clicked.connect(self.predicting)

        # Кнопка "очистить"
        clean_button = QPushButton('ОЧИСТИТЬ', self)
        clean_button.setFont(font)
        clean_button.move(290, 80)
        clean_button.resize(230, 33)
        clean_button.clicked.connect(self.clear)
        clean_button.clicked.connect(self.line.clear)

        # Кнопка "загрузить изображение"
        load_button = QPushButton('ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ', self)
        load_button.setFont(font)
        load_button.move(290, 246)
        load_button.resize(230, 33)
        load_button.clicked.connect(self.load_image)
        load_button.clicked.connect(self.save)
        load_button.clicked.connect(self.predicting)

        # Лейбл "drag n drop"
        drag_n_drop = QLabel('Вы также можете\nперетащить изображение на окно', self)
        drag_n_drop.setFont(font)
        drag_n_drop.move(290,190)
        drag_n_drop.resize(230,30)

        # Разрешаем принимать файлы на окно GUI
        self.setAcceptDrops(True)

        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)

    # Метод принимающий результат распознавания буквы и вывод ее на поле "результат"
    def print_letter(self, result):
        letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        self.line.setText(str(letters[result]))
        return letters[result]

    # Метод загрузки модели нейронной сети и изображения буквы
    def predicting(self):
        image = keras.preprocessing.image
        model = keras.models.load_model('C:/Users/Demon777/Desktop/подборка/model_CoMNIST')
        img = image.load_img('res.jpg', target_size=(32,32))
        #Image.fromqimage(img)

        # Преобразование изображения в массив и передача в модель
        #img_pil = Image.fromqimage(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=1)

        # Получение результата
        result = int(np.argmax(classes))
        self.print_letter(result)
    # Методы mousePressEvent, mouseMoveEvent, mouseReleaseEvent, paintEvent
    # отображают реагирование на движение и нажатие кнопок мыши и отображение изменения на экране
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)

    # Метод сохранения изображения буквы
    def save(self):
        self.image.save('res.jpg')

    # Метод очищения холста
    def clear(self):
        self.image = QImage(278, 278, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.update()

    # Метод загрузки изображения из папки
    def load_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Выбрать изображение', '', 'Изображения (*.jpg *.jpeg *.png)')[0]
        if fname:
            img = Image.open(fname)
            img = img.resize((278, 278), Image.BILINEAR)
            img.save(fname)
            self.image = QImage(fname)
            self.update()

    # Методы dragEnterEvent и dropEvent обрабатывают события перетаскивания файлов на окно приложения
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            fname = url.toLocalFile()
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(fname)
                img = img.resize((278, 278), Image.BILINEAR)
                img.save(fname)
                self.image = QImage(fname)
                self.predicting()
                self.update()
                break

# Конструкция для запуска приложения с GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()


