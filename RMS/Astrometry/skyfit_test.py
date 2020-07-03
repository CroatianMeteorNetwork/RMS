from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap, QPicture, QPainter, QBrush, QPen
from PyQt5.QtCore import QPoint, QRectF, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsView, QLabel, QVBoxLayout, QMainWindow, QGridLayout, \
    QLabel, QStackedWidget
import pyqtgraph as pg

import time
import sys
import numpy as np
import dill
import os

from RMS.Astrometry.CustomPyqtgraphClasses import *


class GUI(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()

        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.setMinimumSize(640, 480)
        self.show()

        layout = QGridLayout()
        self.central.setLayout(layout)

        self.mouse = (0, 0)

        # timing
        self.i = 0
        self.n = 100
        self.frames = np.zeros(self.n)
        self.time = time.time()

        # Image on left
        self.v = pg.GraphicsView()
        self.vb = pg.ViewBox()
        self.vb.invertY()
        self.vb.setAspectLocked()
        self.vb.setMouseEnabled(False, False)
        self.image_scroll = True  # optional
        self.vb.setMenuEnabled(False)
        self.v.setCentralItem(self.vb)
        layout.addWidget(self.v, 0, 1)

        ### REMOVE THIS ###
        self.data = pg.gaussianFilter(np.random.normal(size=(256, 256)), (20, 20))
        for i in range(32):
            for j in range(32):
                self.data[i*8, j*8] += .1

        self.img = None
        self.img_copy = None

        # zoom window
        self.v_zoom = pg.GraphicsView(self.v)
        self.zoom_window = pg.ViewBox()
        self.zoom_window.setAspectLocked()
        self.zoom_window.setMouseEnabled(False, False)
        self.zoom_window.setMenuEnabled(False)
        self.zoom_window_width = 200
        self.v_zoom.setFixedWidth(self.zoom_window_width)
        self.v_zoom.setFixedHeight(self.zoom_window_width)
        self.zoom_visible = False
        self.zoom()
        self.v_zoom.hide()
        self.v_zoom.setCentralItem(self.zoom_window)
        self.v_zoom.move(QPoint(0, 0))

        # histogram
        self.auto = False
        self.hist = HistogramLUTWidget2()
        layout.addWidget(self.hist, 0, 2)
        self.hist_visible = False
        self.hist.hide()

        # label
        # self.label = QLabel(self.v)
        # self.label_visible = 0
        # self.label.setFixedWidth(150)
        # self.label.setStyleSheet("background-color: rgba(255,255,255,100)")
        # self.label.setMargin(10)

        # markers
        self.markers = pg.ScatterPlotItem()
        self.vb.addItem(self.markers)
        self.markers.setPen('b')
        self.markers.setSize(10)
        self.markers.setSymbol(CircleLine())
        self.markers.setBrush(QColor(0, 0, 0, 0))
        self.markers.setZValue(2)

        # markers
        self.markers2 = pg.ScatterPlotItem()
        self.zoom_window.addItem(self.markers2)
        self.markers2.setPen('b')
        self.markers2.setSize(20)
        self.markers2.setSymbol('+')
        self.markers2.setZValue(2)

        # cursor
        self.cursor = CursorItem(1, pxmode=True)
        self.radius = 1
        self.vb.addItem(self.cursor, ignoreBounds=True)
        self.cursor_visible = False
        self.cursor.hide()
        self.cursor_scroll = False  # optional
        self.cursor.setZValue(100)

        self.x = []

        self.cursor2 = CursorItem(1)
        self.zoom_window.addItem(self.cursor2, ignoreBounds=True)
        self.cursor2.hide()
        self.cursor2.setZValue(200)

        # text
        self.text2 = TextItem(html='hello how are you doing\nhello',
                              fill=QColor(255, 255, 255, 100), anchor=(0.5, -0.5))
        self.text2.setColor(QColor(255, 255, 0))
        self.text2.setPos(100, 100)
        self.text2.setZValue(1000)
        self.vb.addItem(self.text2)

        self.text3 = TextItem(html='hello how are you doing\nhello',
                      fill=QColor(255, 255, 255, 100), anchor=(0.5, 1.5))
        self.text3.setColor(QColor(255, 255, 0))
        self.text3.setPos(100, 100)
        self.text3.setZValue(1000)
        self.vb.addItem(self.text3)

        self.text = TextItem('', fill=QColor(255, 255, 255, 100))
        self.text.setTextWidth(150)
        self.text.setPos(0,0)
        self.text.setParentItem(self.vb)
        self.text.setZValue(1000)

        # key binding
        self.vb.scene().sigMouseMoved.connect(self.mouseMove)
        self.vb.scene().sigMouseClicked.connect(self.mouseClick)  # NOTE: clicking event doesnt trigger if moving

        self.updateImage()
        self.updateLabel()

    def updateLabel(self):
        """ Update text """
        self.text.setText("Position: ({:.2f},{:.2f})\n"
                                         "Gamma: {:.2f}\n"
                                         "MORE TEXT".format(*self.mouse, self.img.gamma))

    def zoom(self):
        """ Update the zoom window to zoom on the correct position """
        # self.zoom_window.autoRange()
        # zoom_scale = 0.1
        # self.zoom_window.scaleBy(zoom_scale, QPoint(*self.mouse))
        self.zoom_window.setXRange(self.mouse[0] - 20, self.mouse[0] + 20)
        self.zoom_window.setYRange(self.mouse[1] - 20, self.mouse[1] + 20)

    def mouseClick(self, event):
        if self.cursor_visible:
            if event.button() == 1:  # left click
                self.markers.addPoints(pos=[self.mouse])
                self.markers2.addPoints(pos=[self.mouse])
            elif event.button() == 2:  # right click
                pass

    def mouseMove(self, event):
        pos = event
        if self.vb.sceneBoundingRect().contains(pos):
            mp = self.vb.mapSceneToView(pos)
            self.cursor.setCenter(mp)
            self.cursor2.setCenter(mp)
            self.text2.setPos(mp)
            self.text3.setPos(mp)
            self.mouse = (mp.x(), mp.y())

            self.zoom()
            # moving zoom window to the right position
            range_ = self.vb.getState()['viewRange'][0]
            if mp.x() > (range_[1] - range_[0])/2 + range_[0]:
                self.v_zoom.move(QPoint(0, 0))
            else:
                self.v_zoom.move(QPoint(self.vb.size().width() - self.zoom_window_width, 0))

            self.updateLabel()

        # self.printFrameRate()
        # self.update()

    def wheelEvent(self, event):
        """ Change star selector aperature on scroll. """
        delta = event.angleDelta().y()
        modifier = QApplication.keyboardModifiers()

        if self.vb.sceneBoundingRect().contains(event.pos()):
            if modifier == Qt.ControlModifier:
                if delta < 0:
                    self.radius += 1
                    self.cursor.setRadius(self.radius)
                    self.cursor2.setRadius(self.radius)
                elif delta > 0 and self.radius > 1:
                    self.radius -= 1
                    self.cursor.setRadius(self.radius)
                    self.cursor2.setRadius(self.radius)
            else:
                if delta < 0:
                    self.vb.autoRange(padding=0)
                elif delta > 0:
                    self.vb.scaleBy([0.95, 0.95], QPoint(*self.mouse))

    def updateImage(self):
        """ Adjust image to new self. """
        # remove previous items
        if self.img is not None:
            self.vb.removeItem(self.img)
            self.zoom_window.removeItem(self.img_copy)
            gamma = self.img.gamma
        else:
            gamma = 1

        # CHANGES ON DATA DUE TO VARIABLES HERE
        self.img = ImageItem2(self.data, gamma=gamma)
        self.img_copy = ImageItem2(self.data, gamma=gamma)

        # add new images to viewboxes
        self.vb.addItem(self.img)
        self.vb.autoRange(padding=0)
        self.zoom_window.addItem(self.img_copy)

        # initialize histogram levels
        if not self.auto:
            self.hist_levels = self.hist.getLevels()
        self.auto_levels = (np.percentile(self.data, 0.1), np.percentile(self.data, 99.95))

        # connect images to the histogram
        self.hist.setImageItem(self.img)
        self.hist.setImages(self.img_copy)

        if self.auto:
            self.hist.setLevels(*self.auto_levels)
        else:
            self.hist.setLevels(*self.hist_levels)

    def saveState(self):
        dir_path = 'C:/users/jonat/documents'
        file_name = 'file.state'
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            dill.dump(self, f, protocol=2)

    def nextImg(self):
        """ get new data """
        self.data = pg.gaussianFilter(np.random.normal(size=(256, 256)), (20, 20))
        for i in range(32):
            for j in range(32):
                self.data[i*8, j*8] += .1

        self.markers.clear()
        self.updateImage()

    def keyPressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()

        if event.key() == Qt.Key_H and modifiers == Qt.ControlModifier:
            if self.hist_visible:
                self.hist.hide()
            else:
                self.hist.show()
            self.hist_visible = not self.hist_visible

        elif event.key() == Qt.Key_R and modifiers == Qt.ControlModifier:
            if self.cursor_visible:
                self.cursor.hide()
                self.cursor2.hide()
            else:
                self.cursor.show()
                self.cursor2.show()
            self.cursor_visible = not self.cursor_visible

        elif event.key() == Qt.Key_A and modifiers == Qt.ControlModifier:
            self.auto = not self.auto
            if self.auto:
                self.hist_levels = self.hist.getLevels()
                self.hist.setLevels(*self.auto_levels)
                self.hist.setMovable(False)
            else:
                self.hist.setLevels(*self.hist_levels)
                self.hist.setMovable(True)

        elif event.key() == Qt.Key_F1:
            if self.label_visible == 0:
                self.label.hide()
                self.label_visible = 1
            elif self.label_visible == 1:
                self.label.show()
                self.label_visible = 0

        elif event.key() == Qt.Key_S and modifiers == Qt.ControlModifier:
            self.saveState()

        elif not self.image_scroll and event.key() == Qt.Key_Z and modifiers == Qt.ShiftModifier:
            if self.zoom_visible:
                self.v_zoom.hide()
            else:
                self.v_zoom.show()
            self.zoom_visible = not self.zoom_visible

        elif event.key() == Qt.Key_Left:
            self.nextImg()

        elif event.key() == Qt.Key_U:
            self.img.updateGamma(1/0.9)
            self.img_copy.updateGamma(1/0.9)
            self.updateLabel()

        elif event.key() == Qt.Key_J:
            self.img.updateGamma(0.9)
            self.img_copy.updateGamma(0.9)
            self.updateLabel()

    def printFrameRate(self):
        try:
            print('FPS: {}'.format(np.average(self.frames)))
            self.frames[self.i] = 1/(time.time() - self.time)
            self.i = (self.i + 1)%self.n
        except ZeroDivisionError:
            pass
        self.time = time.time()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    sys.exit(app.exec_())
