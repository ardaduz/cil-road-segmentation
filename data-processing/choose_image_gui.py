import sys
from PyQt5 import QtCore
from PyQt5 import QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QGraphicsView, QGraphicsScene, \
    QVBoxLayout, QGridLayout

from PyQt5.QtWidgets import QApplication, QLabel, QWidget
from PyQt5.QtWidgets import QMainWindow, QApplication

import numpy as np
import cv2
from os import listdir
from os.path import isfile
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

image_size = 400

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
        self.setFixedSize(image_size, image_size)
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
    def __init__(self, parent=None, read_directory_X=None, read_directory_y=None,
                 save_directory_X=None, save_directory_y=None, index_start=None,
                 index_end=None, sampling_rate=None, crop_size=None):
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

        self.overlaid_status = True

        self.grid = QGridLayout()
        self.overlay_label = PhotoViewer(self)
        self.sil_label = QLabel(self)

        self.setLayout(self.grid)
        self.keepButton = QtWidgets.QPushButton('Keep', self)
        self.discardButton = QtWidgets.QPushButton('Discard', self)
        self.toggleOverlayButton = QtWidgets.QPushButton('Toggle Overlay', self)
        self.indexText = QtWidgets.QTextEdit()
        self.indexText.setFixedSize(200, 30)
        self.indexText.setText(str(self.index) + ' / ' + str(self.index_end))

        self.grid.addWidget(self.overlay_label, 0, 0)
        self.grid.addWidget(self.sil_label, 0, 1)
        self.grid.addWidget(self.indexText, 1, 0)
        self.grid.addWidget(self.keepButton, 0, 2)
        self.grid.addWidget(self.discardButton, 0, 3)
        self.grid.addWidget(self.toggleOverlayButton, 0, 4)
        self.keepButton.clicked.connect(self.handleKeep)
        self.discardButton.clicked.connect(self.handleDiscard)
        self.toggleOverlayButton.clicked.connect(self.handleToggle)

        self.preprocess()

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

        resized_sil_gray = cv2.resize(self.sil[:, :, 0], (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        height, width = resized_sil_gray.shape
        bytesPerLine = width
        qimage = QtGui.QImage(resized_sil_gray, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.sil_label.setPixmap(qpixmap)

    def handleKeep(self):
        cv2.imwrite(self.save_directory_X + self.filename_X, self.original_img)
        cv2.imwrite(self.save_directory_y + self.filename_y, self.sil[:, :, 0])

        np.save('index_checkpoint.npy', np.array([self.index, sampling_rate, index_end]))

        self.index = self.index + sampling_rate

        if self.index >= self.index_end:
            sys.exit(app.exec_())

        self.preprocess()

    def handleDiscard(self):
        np.save('index_checkpoint.npy', np.array([self.index, sampling_rate, index_end]))

        self.index = self.index + sampling_rate

        if self.index >= self.index_end:
            sys.exit(app.exec_())

        self.preprocess()

    def handleToggle(self):
        if self.overlaid_status:
            self.overlay = np.copy(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        else:
            self.overlay = np.copy(self.original_overlay)

        self.overlaid_status = not self.overlaid_status

        self.img = np.copy(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        self.sil = np.copy(self.original_sil)

        self.myUpdate()

    def myUpdate(self):
        height, width, channel = self.overlay.shape
        bytesPerLine = 3 * width
        qimage = QtGui.QImage(self.overlay, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.overlay_label.modifyCurrentPhoto(qpixmap)

        resized_sil_gray = cv2.resize(self.sil[:, :, 0], (image_size, image_size), interpolation=cv2.INTER_NEAREST)
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

    save_directory_X = '../google-maps-data-chosen/images/'
    save_directory_y = '../google-maps-data-chosen/groundtruth/'

    app = QApplication(sys.argv)
    w = MyWidget(None, read_directory_X, read_directory_y, save_directory_X, save_directory_y, index_start, index_end,
                 sampling_rate, crop_size=1024)
    w.show()
    sys.exit(app.exec_())
