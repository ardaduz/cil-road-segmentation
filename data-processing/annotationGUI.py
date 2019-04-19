import sys
from PyQt5 import QtCore
from PyQt5 import QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog,QGraphicsView,QGraphicsScene,QVBoxLayout, QGridLayout

from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtWidgets import QMainWindow, QApplication

import numpy as np
import cv2
from os import listdir
from os.path import isfile
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setFixedSize(400, 400)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        self.fitInView(self._photo)
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(pixmap)

        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())

    def modifyCurrentPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(pixmap)

        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView(self._photo)
            else:
                self._zoom = 0

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(QtCore.QPoint(event.pos()))
        super(PhotoViewer, self).mousePressEvent(event)


class MyWidget(QWidget):
    def __init__(self, parent=None, read_directory_X = None, read_directory_y = None,
                 save_directory_X = None, save_directory_y = None, index_start = None,
                 index_end = None, sampling_rate=None, crop_size=None):
        QWidget.__init__(self, parent)

        self.index = index_start
        self.index_end = index_end
        self.mode = 'M'
        self.sampling_rate = sampling_rate
        self.crop_size = crop_size

        self.read_directory_X = read_directory_X
        self.read_directory_y = read_directory_y
        self.save_directory_X = save_directory_X
        self.save_directory_y = save_directory_y
        self.filenames_X = sorted([listdir(read_directory_X)][0])
        self.filenames_y = sorted([listdir(read_directory_y)][0])

        self.original_sil = None
        self.original_overlay = None
        self.original_img = None

        self.overlay = None
        self.sil = None
        self.img = None

        self.grid = QGridLayout()
        self.overlay_label = PhotoViewer(self)
        self.sil_label = QLabel(self)

        self.backstack = []
        self.brushsize = 3
        self.setLayout(self.grid)
        self.saveButton = QtWidgets.QPushButton('Save', self)
        self.skipButton = QtWidgets.QPushButton('Skip', self)
        self.increaseButton = QtWidgets.QPushButton('Increase Brush', self)
        self.decreaseButton = QtWidgets.QPushButton('Decrease Brush', self)
        self.resetButton = QtWidgets.QPushButton('Reset', self)
        self.undoButton = QtWidgets.QPushButton('Undo', self)
        self.modeButton = QtWidgets.QPushButton('Mode:'+ self.mode, self)
        self.brushSizeText = QtWidgets.QTextEdit()
        self.brushSizeText.setText(str(self.brushsize))
        self.brushSizeText.setFixedSize(50, 30)
        self.indexText = QtWidgets.QTextEdit()
        self.indexText.setFixedSize(200, 30)
        self.indexText.setText(str(self.index) + ' / ' + str(self.index_end))

        self.grid.addWidget(self.overlay_label, 0, 0)
        self.grid.addWidget(self.sil_label, 0, 1)
        self.grid.addWidget(self.indexText, 1, 0)
        self.grid.addWidget(self.saveButton, 0, 2)
        self.grid.addWidget(self.skipButton, 0, 3)
        self.grid.addWidget(self.increaseButton, 1, 4)
        self.grid.addWidget(self.decreaseButton, 1, 5)
        self.grid.addWidget(self.brushSizeText, 1, 6)
        self.grid.addWidget(self.resetButton, 0, 4)
        self.grid.addWidget(self.undoButton, 0, 5)
        self.grid.addWidget(self.modeButton, 0, 6)
        self.saveButton.clicked.connect(self.handleSave)
        self.skipButton.clicked.connect(self.handleSkip)
        self.increaseButton.clicked.connect(self.handleIncrease)
        self.decreaseButton.clicked.connect(self.handleDecrease)
        self.resetButton.clicked.connect(self.handleReset)
        self.undoButton.clicked.connect(self.handleUndo)
        self.modeButton.clicked.connect(self.handleMode)

        self.preprocess()

    def keyPressEvent(self, a0: QtGui.QKeyEvent):
        if a0.key() == QtCore.Qt.Key_M:
            self.handleMode()
        elif a0.key() == QtCore.Qt.Key_Period:
            self.handleIncrease()
        elif a0.key() == QtCore.Qt.Key_Comma:
            self.handleDecrease()


    def preprocess(self):
        self.filename_X = self.filenames_X[self.index]
        self.filename_y = self.filenames_y[self.index]

        img_path = self.read_directory_X + self.filename_X
        sil_path = self.read_directory_y + self.filename_y

        img = cv2.imread(img_path)
        sil = cv2.imread(sil_path)

        self.loadImage(img, sil)

        self.indexText.setText(str(self.index) + ' / ' + str(self.index_end))


    def loadImage(self, img, sil):

        self.original_img = np.copy(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sil = cv2.cvtColor(sil, cv2.COLOR_BGR2RGB)

        sil[:, :, 1:3] = 0

        overlay = cv2.addWeighted(img, 1, sil, 0.6, 0)

        self.original_sil = np.copy(sil)
        self.original_overlay = np.copy(overlay)

        self.sil = np.copy(sil)
        self.overlay = np.copy(overlay)
        self.img = np.copy(img)

        height, width, channel = self.overlay.shape
        bytesPerLine = 3 * width
        qimage = QtGui.QImage(self.overlay, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.overlay_label.setPhoto(qpixmap)
        # self.overlay_label.setPixmap(qpixmap)
        self.overlay_label.mousePressEvent = self.getPos

        resized_sil_gray = cv2.resize(self.sil[:,:,0], (400, 400), interpolation=cv2.INTER_NEAREST)
        height, width = resized_sil_gray.shape
        bytesPerLine = width
        qimage = QtGui.QImage(resized_sil_gray, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.sil_label.setPixmap(qpixmap)


    def handleMode(self):
        if self.mode == 'M':
            self.mode = 'U'

        else:
            self.mode = 'M'

        self.modeButton.setText('Mode:' + self.mode)

    def handleSave(self):
        print('Save')

        cv2.imwrite(self.save_directory_X + self.filename_X, self.original_img)
        cv2.imwrite(self.save_directory_y + self.filename_y, self.sil[:,:,0])

        np.save('index_checkpoint.npy', np.array([self.index, sampling_rate, index_end]))

        self.index = self.index + sampling_rate

        if self.index >= self.index_end:
            sys.exit(app.exec_())

        self.preprocess()

    def handleSkip(self):
        print('Skip')
        self.backstack = []

        np.save('index_checkpoint.npy', np.array([self.index, sampling_rate, index_end]))

        self.index = self.index + sampling_rate

        if self.index >= self.index_end:
            sys.exit(app.exec_())

        self.preprocess()

    def handleReset(self):
        self.overlay = np.copy(self.original_overlay)
        self.img = np.copy(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        self.sil = np.copy(self.original_sil)
        self.backstack = []

        self.myUpdate()

    def handleIncrease(self):
        self.brushsize = np.clip(self.brushsize + 1, 0, None)
        self.brushSizeText.setText(str(self.brushsize))

    def handleDecrease(self):
        self.brushsize = np.clip(self.brushsize - 1, 0, None)
        self.brushSizeText.setText(str(self.brushsize))

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        sz = self.brushsize

        point = self.overlay_label.mapToScene(x, y)

        x = np.int(point.x())
        y = np.int(point.y())

        left = np.clip(x - sz, 0, self.crop_size)
        right = np.clip(x + sz, 0, self.crop_size)
        top = np.clip(y - sz, 0, self.crop_size)
        bottom = np.clip(y + sz, 0, self.crop_size)

        if self.mode == 'M':
            self.mark(top, bottom, left, right)
        else:
            self.unmark(top, bottom, left, right)

        self.myUpdate()

        print(y, x)

    def mark(self, top, bottom, left, right):
        if left==right:
            self.backstack.append((top, bottom, left, right, np.copy(self.sil[top, left, 0])))
            self.sil[top, left, 0] = 255
        else:
            self.backstack.append((top, bottom, left, right, np.copy(self.sil[top:bottom, left:right, 0])))
            self.sil[top:bottom, left:right, 0] = 255

        self.overlay = cv2.addWeighted(self.img, 1, self.sil, 0.6, 0)

    def unmark(self, top, bottom, left, right):
        if left==right:
            self.backstack.append((top, bottom, left, right, np.copy(self.sil[top, left, 0])))
            self.sil[top, left, 0] = 0
        else:
            self.backstack.append((top, bottom, left, right, np.copy(self.sil[top:bottom, left:right, 0])))
            self.sil[top:bottom, left:right, 0] = 0

        self.overlay = cv2.addWeighted(self.img, 1, self.sil, 0.6, 0)

    def handleUndo(self):
        if self.backstack.__len__() != 0:
            top, bottom, left, right, saved = self.backstack.pop()

            if left == right:
                self.sil[top, left, 0] = saved
            else:
                self.sil[top:bottom, left:right, 0] = saved

            self.overlay = cv2.addWeighted(self.img, 1, self.sil, 0.6, 0)

            self.myUpdate()

    def myUpdate(self):
        height, width, channel = self.overlay.shape
        bytesPerLine = 3 * width
        qimage = QtGui.QImage(self.overlay, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.overlay_label.modifyCurrentPhoto(qpixmap)

        resized_sil_gray = cv2.resize(self.sil[:, :, 0], (400, 400), interpolation=cv2.INTER_NEAREST)
        height, width = resized_sil_gray.shape
        bytesPerLine = width
        qimage = QtGui.QImage(resized_sil_gray, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.sil_label.setPixmap(qpixmap)

        self.layout().update()



if __name__ == '__main__':


    checkpoint_exists = isfile('index_checkpoint.npy')

    if checkpoint_exists:
        checkpoint_array = np.load('index_checkpoint.npy')
        index_start = checkpoint_array[0]
    else:
        index_start = 0

    index_end = 10000
    sampling_rate = 1

    read_directory_X = '../google-maps-data/images/'
    read_directory_y = '../google-maps-data/groundtruth/'

    save_directory_X = '../google-maps-data-annotated/images/'
    save_directory_y = '../google-maps-data-annotated/groundtruth/'

    app = QApplication(sys.argv)
    w = MyWidget(None, read_directory_X, read_directory_y, save_directory_X, save_directory_y, index_start, index_end, sampling_rate, crop_size=1024)
    w.show()
    sys.exit(app.exec_())