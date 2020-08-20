from __future__ import division, absolute_import, unicode_literals

import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from RMS.Routines import Image
from RMS.Routines.DebruijnSequence import findAllInDeBruijnSequence, generateDeBruijnSequence

import time
import re
import sys


class Plus(QtGui.QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(Plus())

    Consists of two lines with no fill making a plus sign
    """

    def __init__(self):
        QtGui.QPainterPath.__init__(self)
        points = np.asarray([
            (-0.5, 0),
            (0.5, 0),
            (0, 0.5),
            (0, -0.5),
        ])

        for i in range(0, len(points), 2):
            self.moveTo(*points[i])
            self.lineTo(*points[i + 1])
        self.closeSubpath()


class Cross(QtGui.QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(Cross())

    Consists of two lines with no fill making a cross
    """

    def __init__(self):
        QtGui.QPainterPath.__init__(self)
        points = np.asarray([
            (-0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5),
            (0.5, -0.5),
        ])

        for i in range(0, len(points), 2):
            self.moveTo(*points[i])
            self.lineTo(*points[i + 1])
        self.closeSubpath()


class CircleLine(QtGui.QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(CircleLine())

    Consists of a circle with fill that can be removed (with setBrush(QColor(0,0,0,0))),
    with a line going from the top to the center
    """

    def __init__(self):
        QtGui.QPainterPath.__init__(self)
        points = np.asarray([(0, -0.5), (0, 0)])
        self.moveTo(*points[0])
        self.lineTo(*points[1])
        self.closeSubpath()

        self.addEllipse(QtCore.QPoint(0, 0), 0.5, 0.5)


class Crosshair(QtGui.QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(Crosshair())

    Consists of a circle with fill that can be removed (with setBrush(QColor(0,0,0,0))),
    with four lines going from the top, bottom, left and right to near the center
    """

    def __init__(self):
        QtGui.QPainterPath.__init__(self)
        points = np.asarray([(0, -0.5), (0, -0.2),
                             (0, 0.5), (0, 0.2),
                             (0.5, 0), (0.2, 0),
                             (-0.5, 0), (-0.2, 0)])

        for i in range(0, len(points), 2):
            self.moveTo(*points[i])
            self.lineTo(*points[i + 1])
        self.closeSubpath()

        self.addEllipse(QtCore.QPoint(0, 0), 0.5, 0.5)


class CustomMessageBox(QtWidgets.QMessageBox):
    """  Identical to QMessageBox except doesn't use setInformativeText and an icon.
        instead allows to add widgets to the top section that can be changed externally.
     """

    def __init__(self, *args, **kwargs):
        QtWidgets.QMessageBox.__init__(self, *args, **kwargs)
        content = QtWidgets.QWidget()
        self.vbox = QtWidgets.QVBoxLayout(content)
        self.layout().addWidget(content, 0, 0)

        self._label = QtWidgets.QLabel()
        self._label.hide()
        self.vbox.addWidget(self._label)

    def addWidget(self, widget):
        self.vbox.addWidget(widget)

    def setText(self, text):
        self._label.setText(text)
        if self._label.text() != '':
            self._label.show()
        else:
            self._label.hide()

    def setInformativeText(self, text):
        pass

    def setIcon(self, icon):
        pass


class TextItemList(pg.GraphicsObject):
    """
    Allows for a list of TextItems without having to constantly add items to a widget
    ex.
    text_list = TextItemList()
    text_list.addNewTextItem('hello')
    widget.addItem(text_list)
    """

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.text_list = []
        self.z = 0

    def getTextItem(self, i):
        """
        Return the TextItem at index i. Can only be used for getting information, changing
        values will not change any values in the list

        Identical to:
        text = TextItemList()
        ...
        text[i]

        Arguments:
            i [int]: index
        """
        return self.text_list[i]

    def __getitem__(self, key):
        return self.text_list[key]

    def addTextItem(self, text):
        """
        Add TextItem object to list. It will be displayed automatically without
        any management of the TextItem

        Arguments:
            text [TextItem]: TextItem to add to list
        """
        text.setParentItem(self.parentItem())
        text.setZValue(self.z)
        self.text_list.append(text)

    def addNewTextItem(self, *args, **kwargs):
        """
        Has the same arguments as __init__ in TextItem
        """
        new = TextItem(*args, **kwargs)
        new.setParentItem(self.parentItem())
        new.setZValue(self.z)
        self.text_list.append(new)

    def setZValue(self, z):
        """
        Sets all TextItem's in list to have Z value (affects when it is drawn) and
        new TextItem's will have this Z value

        Arguments:
            z [float]: z value to set all TextItem's to
        """
        self.z = z
        for text in self.text_list:
            text.setZValue(z)

    def clear(self):
        """
        Remove all TextItem's in list
        """
        while self.text_list:
            self.removeTextItem(0)

    def removeTextItem(self, i):
        """
        Remove TextItem at index i

        Arguments:
            i [int]: index
        """
        item = self.text_list.pop(i)
        try:
            self.parentItem().scene().removeItem(item)
        except:
            pass
        item.setParentItem(None)

    def setParentItem(self, parent):
        super().setParentItem(parent)
        for text in self.text_list:
            text.setParentItem(parent)

    def paint(self, painter, option, widget=None):
        for text in self.text_list:
            text.update()

    def boundingRect(self):
        return QtCore.QRectF()


class TextItem(pg.TextItem):
    def __init__(self, text='', color=(200, 200, 200), html=None, anchor=(0, 0),
                 border=None, fill=None, angle=0, rotateAxis=None):
        pg.TextItem.__init__(self, text, color, html, anchor, border, fill, angle, rotateAxis)

    def setAlign(self, align):
        """
        Set alignment of the TextItem

        Arguments:
            align [Qt.AlignLeft, Qt.AlignCenter, Qt.AlignRight]
        """
        option = self.textItem.document().defaultTextOption()
        option.setAlignment(align)
        self.textItem.document().setDefaultTextOption(option)
        self.textItem.setTextWidth(self.textItem.boundingRect().width())
        self.updateTextPos()


class ViewBox(pg.ViewBox):
    # new signals are made since they give more information that mouseClickEvent
    sigMousePressed = QtCore.pyqtSignal(object)
    sigMouseReleased = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        pg.ViewBox.__init__(self, *args, **kwargs)

    def keyPressEvent(self, ev):
        """
        The original ViewBox accepts key mappings to + - and =, which prevents their usage.
        This passes the keypressevent on
        """
        ev.ignore()

    def mouseReleaseEvent(self, event):
        self.sigMouseReleased.emit(event)

    def mousePressEvent(self, event):
        if self.mouseEnabled()[0]:
            super().mousePressEvent(event)
        else:
            self.sigMousePressed.emit(event)
            event.accept()


class ImageItem(pg.ImageItem):
    # ImageItem that provides an interface around img_handle
    def __init__(self, img_handle=None, **kwargs):
        """
        Makes an image item with img_handle, with the default image of avepixel

        Arguments:
            image_handle: [InputType]
            invert: [boolean] whether to invert image when displaying
            gamma: [float]
            dark:
            flat_struct:
            kwargs: other __init__ arguments of pg.ImageItem
        """
        self.img_handle = img_handle
        pg.ImageItem.__init__(self, image=None, **kwargs)

        if 'gamma' in kwargs.keys():
            self._gamma = kwargs['gamma']
        else:
            self._gamma = 1

        if 'invert' in kwargs.keys():
            self.invert_img = kwargs['invert']
        else:
            self.invert_img = False

        if 'dark' in kwargs.keys():
            self.dark = kwargs['dark']
        else:
            self.dark = None

        if 'flat_struct' in kwargs.keys():
            self.flat_struct = kwargs['flat_struct']
        else:
            self.flat_struct = None

        self.avepixel()
        self.img_showing = 'avepixel'

    def maxpixel(self):
        maxpixel = self.img_handle.loadChunk().maxpixel

        # adding background to FR files
        if self.img_handle.name()[:2] == 'FR':
            original_index = self.img_handle.current_ff_index
            original_time = self.img_handle.currentTime()
            for index in range(len(self.img_handle.ff_list)):
                if index == original_index:
                    continue

                self.img_handle.current_ff_index = index
                if original_time == self.img_handle.currentTime() and self.img_handle.name()[:2] == 'FF':
                    maxpixel = maxpixel + self.img_handle.loadChunk().maxpixel*(maxpixel == 0)
                    break

            self.img_handle.current_ff_index = original_index

        if self.dark is not None:
            maxpixel = Image.applyDark(maxpixel, self.dark)
        if self.flat_struct is not None:
            maxpixel = Image.applyFlat(maxpixel, self.flat_struct)
        self.setImage(np.swapaxes(maxpixel, 0, 1))
        self.img_showing = 'maxpixel'

    def avepixel(self):
        avepixel = self.img_handle.loadChunk().avepixel

        # adding background to FR files
        if self.img_handle.name()[:2] == 'FR':
            original_index = self.img_handle.current_ff_index
            original_time = self.img_handle.currentTime()
            for index in range(len(self.img_handle.ff_list)):
                if index == original_index:
                    continue

                self.img_handle.current_ff_index = index
                if original_time == self.img_handle.currentTime() and self.img_handle.name()[:2] == 'FF':
                    avepixel = avepixel + self.img_handle.loadChunk().avepixel*(avepixel == 0)
                    break

            self.img_handle.current_ff_index = original_index

        if self.dark is not None:
            avepixel = Image.applyDark(avepixel, self.dark)
        if self.flat_struct is not None:
            avepixel = Image.applyFlat(avepixel, self.flat_struct)
        self.setImage(np.swapaxes(avepixel, 0, 1))
        self.img_showing = 'avepixel'

    def reloadImage(self):
        """ If img_handle or the flats and darks was changed, reload the current image """
        if self.img_showing == 'maxpixel':
            self.maxpixel()
        elif self.img_showing == 'avepixel':
            self.avepixel()
        elif self.img_showing == 'frame':
            self.loadFrame()

    def changeHandle(self, img_handle):
        """
        Sets the img_handle to a new one and updates the image accordingly

        Arguments:
            img_handle: [InputType]

        """
        self.img_handle = img_handle
        self.reloadImage()

    def loadFrame(self):
        frame = self.img_handle.loadFrame(avepixel=True)

        if frame is not None:
            # adding background to FR files
            if self.img_handle.name()[:2] == 'FR':
                original_index = self.img_handle.current_ff_index
                original_time = self.img_handle.currentTime()
                for index in range(len(self.img_handle.ff_list)):
                    if index == original_index:
                        continue

                    self.img_handle.current_ff_index = index
                    if original_time == self.img_handle.currentTime() and self.img_handle.name()[:2] == 'FF':
                        frame = frame + self.img_handle.loadChunk().avepixel*(frame == 0)
                        break

                self.img_handle.current_ff_index = original_index

            if self.dark is not None:
                frame = Image.applyDark(frame, self.dark)
            if self.flat_struct is not None:
                frame = Image.applyFlat(frame, self.flat_struct)
            self.setImage(np.swapaxes(frame, 0, 1))
            self.img_showing = 'frame'

    def nextChunk(self):
        self.img_handle.nextChunk()

    def prevChunk(self):
        self.img_handle.prevChunk()

    def nextFrame(self):
        self.img_handle.nextFrame()

    def prevFrame(self):
        self.img_handle.prevFrame()

    def setFrame(self, n):
        self.img_handle.setFrame(n)

    def getAutolevels(self, lower=0.1, upper=99.95):
        return np.percentile(self.image, lower), np.percentile(self.image, upper)

    def loadImage(self, mode, flag='avepixel'):
        """
        Loads an image for the given flag in the given mode. To change the image,
        use nextChunk, prevChunk, or nextFrame, prevFrame, setFrame, or nextLine,
        prevLine (depending on the flag and mode), before calling this

        Args:
            mode: [str] 'skyfit' or ''manualreduction'
            flag: [str] 'avepixel' or 'maxpixel' or 'frame'

        """
        if flag == 'maxpixel':
            self.maxpixel()
        elif mode == 'skyfit':
            self.avepixel()
        else:
            self.loadFrame()

    def getFrame(self):
        return self.img_handle.current_frame

    def nextLine(self):
        if hasattr(self.img_handle, 'current_line'):
            self.img_handle.current_line = (self.img_handle.current_line + 1)% \
                                           self.img_handle.line_number[self.img_handle.current_ff_index]

    def prevLine(self):
        if hasattr(self.img_handle, 'current_line'):
            self.img_handle.current_line = (self.img_handle.current_line - 1)% \
                                           self.img_handle.line_number[self.img_handle.current_ff_index]

    @property
    def line(self):
        return self.img_handle.current_line

    @line.setter
    def line(self, line):
        self.img_handle.current_line = line

    @property
    def gamma(self):
        return self._gamma

    @property
    def data(self):
        return self.image

    def setGamma(self, gamma):
        """
        Sets the image gamma to the given then updates the image

        Arguments:
            gamma: [float]

        """
        old = self._gamma
        self._gamma = gamma

        # require boundaries for gamma
        if self._gamma < 0.1:
            self._gamma = old
        elif self._gamma > 10:
            self._gamma = old

        self.updateImage()

    def updateGamma(self, factor):
        self.setGamma(self.gamma*factor)

    def invert(self):
        self.invert_img = not self.invert_img
        self.updateImage()

    def render(self):
        # THIS WAS COPY PASTED FROM SOURCE CODE AND WAS SLIGHTLY
        # CHANGED TO IMPLEMENT GAMMA AND INVERT

        # Convert data to QImage for display.

        profile = pg.debug.Profiler()
        if self.image is None or self.image.size == 0:
            return
        if callable(self.lut):
            lut = self.lut(self.image)
        else:
            lut = self.lut

        if self.autoDownsample:
            # reduce dimensions of image based on screen resolution
            o = self.mapToDevice(pg.QtCore.QPointF(0, 0))
            x = self.mapToDevice(pg.QtCore.QPointF(1, 0))
            y = self.mapToDevice(pg.QtCore.QPointF(0, 1))
            w = pg.Point(x - o).length()
            h = pg.Point(y - o).length()
            if w == 0 or h == 0:
                self.qimage = None
                return
            xds = max(1, int(1.0/w))
            yds = max(1, int(1.0/h))
            axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
            image = pg.fn.downsample(self.image, xds, axis=axes[0])
            image = pg.fn.downsample(image, yds, axis=axes[1])
            self._lastDownsample = (xds, yds)
        else:
            image = self.image

        # if the image data is a small int, then we can combine levels + lut
        # into a single lut for better performance
        levels = self.levels
        # print('Levels:',levels)
        if levels is not None and levels.ndim == 1 and image.dtype in (np.ubyte, np.uint16):
            if self._effectiveLut is None:
                eflsize = 2**(image.itemsize*8)
                ind = np.arange(eflsize)
                minlev, maxlev = levels
                levdiff = maxlev - minlev
                levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
                if lut is None:
                    efflut = pg.fn.rescaleData(ind, scale=255./levdiff,
                                               offset=minlev, dtype=np.ubyte)
                else:
                    lutdtype = np.min_scalar_type(lut.shape[0] - 1)
                    efflut = pg.fn.rescaleData(ind, scale=(lut.shape[0] - 1)/levdiff,
                                               offset=minlev, dtype=lutdtype, clip=(0, lut.shape[0] - 1))
                    efflut = lut[efflut]

                self._effectiveLut = efflut
            lut = self._effectiveLut
            levels = None

        # Assume images are in column-major order for backward compatibility
        # (most images are in row-major order)

        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        argb, alpha = pg.fn.makeARGB(image, lut=lut, levels=levels)
        # LINE THAT WAS CHANGED
        argb[:, :, :3] = np.clip(np.power(argb[:, :, :3]/255, 1/self._gamma)*255, 0, 255)
        if self.invert_img:
            argb[:, :, :3] = 255 - argb[:, :, :3]
        self.qimage = pg.fn.makeQImage(argb, alpha, transpose=False)


class CursorItem(pg.GraphicsObject):
    # this object could be changed so that it uses scatterplotitems instead, since using
    # their libraries is probably faster than what I've made, but it isn't necessary.
    def __init__(self, r, pxmode=False, thickness=1):
        """
        Adds a CursorItem to the point (0,0).

        Arguments:
            r [float]: radius of cursor inner circle
            pxmode [boolean]: whether or not the width of cursor is invariant
            thickness [float]: width of the circles and center dot
        """
        super().__init__()
        self._center = QtCore.QPoint(0, 0)
        self._r = r
        self.mode = 0
        self.thickness = thickness

        self.pxmode = pxmode
        self.picture = QtGui.QPicture()

    def setMode(self, mode):
        """
        Change the mode of the cursor which changes its appearance

        Arguments:
            mode [int]: 0 is two yellow circles with blue point
                        1 is a single purple circle
                        2 is a single filled in red circle
        """
        self.mode = mode
        self.update()

    @property
    def r(self):
        return self._r

    @property
    def center(self):
        return self._center

    def setRadius(self, r):
        self._r = r
        self.update()

    def setCenter(self, new_center):
        """
        Use QPoint(x,y)

        Arguments:
            new_center [QPoint]: Center of the new circle
        """
        self.setPos(new_center)
        self._center = new_center
        self.update()

    def generatePicture(self):
        if self.pxmode and self.parentItem() is not None:
            origin = self.parentItem().mapToDevice(pg.Point(0, 0))
            pos = self.parentItem().mapToDevice(pg.Point(self.r, self.r))
            r = pos.x() - origin.x()
        else:
            r = self.r

        painter = QtGui.QPainter(self.picture)
        if self.mode == 0:
            pen = QtGui.QPen(QtCore.Qt.yellow, self.thickness, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPoint(0, 0), r, r)

            # pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawEllipse(QtCore.QPoint(0, 0), 2*r, 2*r)
            painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2*self.thickness))
            painter.drawPoint(QtCore.QPoint(0, 0))
        elif self.mode == 1:
            pen = QtGui.QPen(QtGui.QColor(128, 0, 128), self.thickness, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPoint(0, 0), 2*r, 2*r)
        else:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0), self.thickness, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(255, 0, 0, 100))
            painter.drawEllipse(QtCore.QPoint(0, 0), r, r)
        painter.end()

        rect = QtCore.QRect(-3*self.r, -3*self.r, 6*self.r, 6*self.r)
        self.picture.setBoundingRect(rect)

    def paint(self, painter, option, widget=None):
        self.generatePicture()
        if self.pxmode:
            painter.translate(self.center.x(), self.center.y())
            t = painter.transform()
            pts = self.parentItem().mapToDevice(pg.Point(self.center.x(), self.center.y()))
            painter.setTransform(QtGui.QTransform(1, 0, t.m13(),
                                                  t.m21(), 1, t.m23(),
                                                  pts.x(), pts.y(), t.m33()))
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class HistogramLUTWidget(pg.HistogramLUTWidget):
    def __init__(self, parent=None, *args, **kwargs):
        pg.HistogramLUTWidget.__init__(self, parent, *args, **kwargs)
        self.item = HistogramLUTItem(*args, **kwargs)
        self.setCentralItem(self.item)
        self.vb.setMenuEnabled(False)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        modifier = QtWidgets.QApplication.keyboardModifiers()
        pos = self.vb.mapSceneToView(event.pos())
        if self.item.region.movable and modifier == QtCore.Qt.ControlModifier:
            if event.button() == QtCore.Qt.LeftButton:
                self.setLevels(pos.y(), self.getLevels()[1])
            elif event.button() == QtCore.Qt.RightButton:
                self.setLevels(self.getLevels()[0], pos.y())


class HistogramLUTItem(pg.HistogramLUTItem):
    def __init__(self, *args, **kwargs):
        pg.HistogramLUTItem.__init__(self, *args, **kwargs)
        self.level_images = []
        self.auto_levels = False
        self.saved_manual_levels = None
        self.region.setBounds((0, None))

    def setImages(self, img):
        """ Store images to automatically set levels that correspond to
            the initial one

        Arguments:
            img: [ImageItem or list of ImageItem]
        """
        if type(img) == ImageItem:
            self.level_images = [img]
        elif type(img) == list and type(img[0]) == ImageItem:
            self.level_images = img
        else:
            raise TypeError

    def toggleAutoLevels(self):
        """
        Switch between auto levels and manual levels
        """
        if not self.auto_levels:
            self.saved_manual_levels = self.getLevels()
            self.setLevels(*self.imageItem().getAutolevels())
        else:
            self.setLevels(*self.saved_manual_levels)

        self.auto_levels = not self.auto_levels
        self.region.setMovable(not self.auto_levels)

    def paint(self, p, *args):
        # tbh this is an improvement
        pass

    def regionChanging(self):
        pass  # doesnt update when moving it

    def regionChanged(self):
        super().regionChanged()
        for img in self.level_images:
            img.setLevels(self.getLevels())

    def setLevels(self, min=None, max=None, rgba=None):
        super().setLevels(min, max, rgba)
        for img in self.level_images:
            # print('img levels:', img.getLevels(), img.levels)
            img.setLevels(self.getLevels())

    def imageChanged(self, autoLevel=False, autoRange=False):
        if not self.auto_levels:
            self.saved_manual_levels = self.getLevels()
        super().imageChanged(autoLevel, autoRange)
        if self.auto_levels:
            self.setLevels(*self.imageItem().getAutolevels())
            # print('auto levels:', self.imageItem().getAutolevels())
        else:
            self.setLevels(*self.saved_manual_levels)
            # print('saved manual levels:', self.saved_manual_levels)

class RightOptionsTab(QtWidgets.QTabWidget):
    """
    Tab widget which initializes and holds each of the tabs. They can be accessed with
    self.hist, self.param_manager, self.debruijn and self.settings
    """

    def __init__(self, gui):
        super(RightOptionsTab, self).__init__()

        self.gui = gui

        self.hist = HistogramLUTWidget()
        self.param_manager = PlateparParameterManager(gui)
        self.settings = SettingsWidget(gui)
        self.debruijn = DebruijnSequenceManager(gui)

        self.index = 0
        self.maximized = True
        self.setFixedWidth(250)

        self.addTab(self.hist, 'Levels')
        self.addTab(self.param_manager, 'Fit Parameters')
        self.addTab(self.settings, 'Settings')

        self.setCurrentIndex(self.index)  # redundant
        self.setTabPosition(QtWidgets.QTabWidget.East)

        self.tabBarClicked.connect(self.onTabBarClicked)

    def keyPressEvent(self, event):
        """ Pressing escape when you're focused on any widget on the right focuses
            on the main widget
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.gui.view_widget.setFocus()

    def onTabBarClicked(self, index):
        if index != self.index:
            self.index = index
            self.maximized = True
            self.setFixedWidth(250)
        else:
            self.maximized = not self.maximized
            if self.maximized:
                self.setFixedWidth(250)
            else:
                self.setFixedWidth(19)

    def onSkyFit(self):
        self.removeTabText('Debruijn')

        self.insertTab(1, self.param_manager, 'Fit Parameters')
        self.settings.onSkyFit()

        self.setCurrentIndex(self.index)

    def onManualReduction(self):
        self.removeTabText('Fit Parameters')
        self.settings.onManualReduction()

        if self.gui.img.img_handle.input_type == 'dfn':
            self.insertTab(1, self.debruijn, 'Debruijn')

        self.setCurrentIndex(self.index)

    def removeTabText(self, text):
        """
        Removes the tab with text. If it can't be found, does nothing.

        Arguments:
            text: The tab to be removed has text text

        """
        for i in range(self.count()):
            if self.tabText(i) == text:
                self.removeTab(i)
                break


class DebruijnSequenceManager(QtWidgets.QWidget):
    # this whole thing could use some huge lower level changes
    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui
        self.sequence = generateDeBruijnSequence(2, 9)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # table
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setFixedWidth(205)
        self.table.setColumnWidth(0, 45)
        self.table.setColumnWidth(1, 75)
        self.table.setColumnWidth(2, 40)
        self.table.setHorizontalHeaderLabels(['break', 'time', 'value'])
        # self.table.verticalHeader().hide()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.currentCellChanged.connect(self.onCurrentCellChanged)
        self.table.cellChanged.connect(self.onCellChanged)
        self.updateTable()
        layout.addWidget(self.table)

        # check sequence button
        self.button = QtWidgets.QPushButton('Check Sequence')
        self.button.clicked.connect(self.onButtonPressed)
        layout.addWidget(self.button)

        # direction radio buttons
        self.no_direction = QtWidgets.QRadioButton('Either time direction')
        layout.addWidget(self.no_direction)
        self.no_direction.setChecked(True)
        self.correct_direction = QtWidgets.QRadioButton('Picked time direction')
        layout.addWidget(self.correct_direction)
        self.reverse_direction = QtWidgets.QRadioButton('Reverse time direction')
        layout.addWidget(self.reverse_direction)

    def onButtonPressed(self):
        reversed = None
        if self.correct_direction.isChecked():
            reversed = False
        elif self.reverse_direction.isChecked():
            reversed = True

        test, paired_first_bit = self.getSequence(get_paired_first_bit=True)
        if test is None:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('DFN Manual Reduction Error')
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText('Sequence is incorrect')
            msg.setInformativeText('Inputted sequence must be a sequence of 11 or 10. '
                                   'The inputted sequence does not.')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        forward, backward = findAllInDeBruijnSequence(test, self.sequence, unknowns=True, reverse=reversed)

        # if multiple solutions exist, show popup window that allows you to select between them
        if len(forward) + len(backward) > 1:
            print('Multiple solutions exist')
            msg = CustomMessageBox()
            msg.setWindowTitle('DFN Manual Reduction Solution Selection')
            msg.setText('There are multiple possible matches of the given sequence to the De Bruijn Sequence.\n'
                        'Select one.')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            msg.buttons()[0].setDisabled(True)
            table = QtWidgets.QTableWidget(len(forward) + len(backward), 4)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            table.setHorizontalHeaderLabels(['break', 'start time', 'direction', 'pattern'])
            table.setColumnWidth(0, 60)
            table.setColumnWidth(3, 170)
            table.verticalHeader().hide()
            table.currentCellChanged.connect(lambda: msg.buttons()[0].setDisabled(False))
            table.setFixedWidth(450)
            table.setFixedHeight(300)
            msg.addWidget(table)
            for row, frame in enumerate(forward):
                break_ = 2*frame + (not paired_first_bit)

                item1 = QtWidgets.QTableWidgetItem(str(break_))
                item1.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 0, item1)

                item2 = QtWidgets.QTableWidgetItem(
                    self.gui.img.img_handle.currentFrameTime(dt_obj=True, frame_no=break_).strftime("%H:%M:%S.%f")[:-3])
                item2.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 1, item2)

                item3 = QtWidgets.QTableWidgetItem('forward')
                item3.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 2, item3)

                item4 = QtWidgets.QTableWidgetItem(
                    ''.join(str(x) for x in self.sequence[frame - 4:frame + len(test) + 4]))
                item4.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 3, item4)

            for row, frame in enumerate(backward[::-1]):
                row += len(forward)
                break_ = 1024 - 2*frame - (not paired_first_bit) - self.table.rowCount() + 1

                item1 = QtWidgets.QTableWidgetItem(str(break_))
                item1.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 0, item1)

                item2 = QtWidgets.QTableWidgetItem(
                    self.gui.img.img_handle.currentFrameTime(dt_obj=True, frame_no=break_).strftime("%H:%M:%S.%f")[:-3])
                item2.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 1, item2)

                item3 = QtWidgets.QTableWidgetItem('backward')
                item3.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 2, item3)

                item4 = QtWidgets.QTableWidgetItem(
                    ''.join(str(x) for x in self.sequence[::-1][frame - 4:frame + len(test) + 4]))
                item4.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 3, item4)

            result = msg.exec_()
            if result == msg.standardButton(msg.buttons()[0]):
                index = table.currentIndex().row()
                if index < len(forward):
                    forward = [forward[index]]
                    backward = []
                else:
                    backward = [(forward + backward[::-1])[index]]
                    forward = []
            else:
                return
        # if there is exactly one solution (possibly after selecting one), update pick frames
        if len(forward) == 1:
            f = self.gui.resetPickFrames(2*forward[0] + (not paired_first_bit), reverse=False)

        elif len(backward) == 1:
            f = self.gui.resetPickFrames(2*backward[0] + (not paired_first_bit), reverse=True)

        else:
            print('Neither were found')
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('DFN Manual Reduction Error')
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText('Sequence could not be found')
            msg.setInformativeText('The sequence given is incorrect.')
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        self.gui.img.setFrame(f(self.gui.img.getFrame()))
        self.gui.updateLeftLabels()
        self.updateTable()
        self.correct_direction.setChecked(True)

    def updateTable(self):
        self.table.setRowCount(0)

        for frame, pick in self.gui.pick_list.items():
            self.modifyRow(frame, pick['mode'])

    def getSequence(self, get_paired_first_bit=False):
        dic = {(1, 1): 1, (1, 0): 0}  # for reference, not used

        sequence = [None]*1024
        start = 1024
        end = 0
        for frame, pick in self.gui.pick_list.items():
            if pick['x_centroid'] is not None:
                if frame < start:
                    start = frame
                if frame > end:
                    end = frame
                sequence[frame] = pick['mode']

        # sequence with pairs as bits
        parsed_sequence = []
        sequence = sequence[start:end + 1]
        paired_first_bit = True

        # convert sequence with pairs in to single sequence
        worked = True
        for i in range(int(np.ceil(len(sequence)/2))):
            if sequence[2*i] == 1 or sequence[2*i] is None:
                try:
                    parsed_sequence.append(sequence[2*i + 1])
                except IndexError:
                    parsed_sequence.append(None)

            elif sequence[2*i] == 0:
                worked = False
                break

        if not worked:
            paired_first_bit = False
            worked = True
            parsed_sequence = [sequence[0]]
            for i in range(len(sequence)//2):
                if sequence[2*i + 1] == 1 or sequence[2*i + 1] is None:
                    try:
                        parsed_sequence.append(sequence[2*i + 2])
                    except IndexError:
                        parsed_sequence.append(None)

                elif sequence[2*i + 1] == 0:
                    worked = False
                    break
        if not worked:
            parsed_sequence = None
            paired_first_bit = None

        if get_paired_first_bit:
            return parsed_sequence, paired_first_bit
        else:
            return parsed_sequence

    def removeRow(self, frame):
        for row in range(self.table.rowCount()):
            if int(self.table.item(row, 0).text()) == frame:
                self.table.removeRow(row)
                break
            elif int(self.table.item(row, 0).text()) > frame:
                break

    def modifyRow(self, frame, value):
        """
        Edit or append row to table with given information.

        Args:
            frame: [int] If frame isn't in table, append new row with this value. Otherwise change
                        the value of the row with this value.
            value: [0 or 1]
        """

        if value is None:
            return

        for row in range(self.table.rowCount()):
            if int(self.table.item(row, 0).text()) == frame:
                self.table.item(row, 2).setText(str(value))
                return
            elif int(self.table.item(row, 0).text()) > frame:
                self.table.insertRow(row)

                item1 = QtWidgets.QTableWidgetItem(str(frame))
                item1.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 0, item1)

                item2 = QtWidgets.QTableWidgetItem(
                    self.gui.img.img_handle.currentFrameTime(dt_obj=True, frame_no=frame).strftime("%H:%M:%S.%f")[:-3])
                item2.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 1, item2)

                item3 = QtWidgets.QTableWidgetItem(str(value))
                self.table.setItem(row, 2, item3)
                return

        row = self.table.rowCount()
        self.table.insertRow(row)

        item1 = QtWidgets.QTableWidgetItem(str(frame))
        item1.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.table.setItem(row, 0, item1)

        item2 = QtWidgets.QTableWidgetItem(
            self.gui.img.img_handle.currentFrameTime(dt_obj=True, frame_no=frame).strftime("%H:%M:%S.%f")[:-3])
        item2.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.table.setItem(row, 1, item2)

        item3 = QtWidgets.QTableWidgetItem(str(value))
        self.table.setItem(row, 2, item3)

    @QtCore.pyqtSlot(int, int, int, int)
    def onCurrentCellChanged(self, row, column, prev_row, prev_col):
        if self.table.item(row, 0) is not None:
            self.gui.img.img_handle.setFrame(int(self.table.item(row, 0).text()))
            self.gui.updateLeftLabels()
            self.gui.updatePicks()

    @QtCore.pyqtSlot(int, int)
    def onCellChanged(self, row, column):
        if column == 2:
            frame = int(self.table.item(row, 0).text())
            pick = self.gui.pick_list[frame]

            if '1' in self.table.item(row, column).text():
                self.table.item(row, column).setText('1')
                pick['mode'] = 1
            elif '0' in self.table.item(row, column).text():
                self.table.item(row, column).setText('0')
                pick['mode'] = 0
            else:
                self.table.item(row, column).setText('1')
                pick['mode'] = 1

            self.gui.updatePicks()


class PlateparParameterManager(QtWidgets.QWidget):
    """
    QWidget that contains various QDoubleSpinBox's that can be changed to
    manage platepar parameters
    """
    sigAzAltChanged = QtCore.pyqtSignal()
    sigRotChanged = QtCore.pyqtSignal()
    sigScaleChanged = QtCore.pyqtSignal()
    sigFitParametersChanged = QtCore.pyqtSignal()
    sigLocationChanged = QtCore.pyqtSignal()
    sigElevChanged = QtCore.pyqtSignal()
    sigExtinctionChanged = QtCore.pyqtSignal()

    sigFitPressed = QtCore.pyqtSignal()
    sigAstrometryPressed = QtCore.pyqtSignal()
    sigPhotometryPressed = QtCore.pyqtSignal()

    sigRefractionToggled = QtCore.pyqtSignal()
    sigEqAspectToggled = QtCore.pyqtSignal()
    sigForceDistortionToggled = QtCore.pyqtSignal()

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui

        full_layout = QtWidgets.QVBoxLayout()
        self.setLayout(full_layout)

        # buttons
        box = QtWidgets.QVBoxLayout()

        self.fit_astrometry_button = QtWidgets.QPushButton("Fit")
        self.fit_astrometry_button.clicked.connect(self.sigFitPressed.emit)
        box.addWidget(self.fit_astrometry_button)

        hbox = QtWidgets.QHBoxLayout()
        self.astrometry_button = QtWidgets.QPushButton('Astrometry')
        self.astrometry_button.clicked.connect(self.sigAstrometryPressed.emit)
        hbox.addWidget(self.astrometry_button)

        self.photometry_button = QtWidgets.QPushButton('Photometry')
        self.photometry_button.clicked.connect(self.sigPhotometryPressed.emit)
        hbox.addWidget(self.photometry_button)
        box.addLayout(hbox)

        self.updatePairedStars()
        group = QtWidgets.QGroupBox('Photometry and Astrometry')
        group.setLayout(box)
        full_layout.addWidget(group)

        # check boxes
        self.refraction = QtWidgets.QCheckBox('Refraction')
        self.refraction.released.connect(self.onRefractionToggled)
        full_layout.addWidget(self.refraction)

        self.eqAspect = QtWidgets.QCheckBox('Equal Aspect')
        self.eqAspect.released.connect(self.onEqualAspectToggled)
        full_layout.addWidget(self.eqAspect)
        if not self.gui.platepar.distortion_type.startswith('radial'):
            self.eqAspect.hide()

        self.fdistortion = QtWidgets.QCheckBox('Force Distortion Centre')
        self.fdistortion.released.connect(self.onForceDistortionToggled)
        full_layout.addWidget(self.fdistortion)
        if not self.gui.platepar.distortion_type.startswith('radial'):
            self.fdistortion.hide()

        # spin boxes
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        full_layout.addLayout(form)

        hbox = QtWidgets.QHBoxLayout()
        self.az_centre = DoubleSpinBox()
        self.az_centre.setMinimum(-360)
        self.az_centre.setMaximum(360)
        self.az_centre.setDecimals(8)
        self.az_centre.setSingleStep(1)
        self.az_centre.setFixedWidth(100)
        self.az_centre.valueModified.connect(self.onAzChanged)
        hbox.addWidget(self.az_centre)
        hbox.addWidget(QtWidgets.QLabel('°', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Azim'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.alt_centre = DoubleSpinBox()
        self.alt_centre.setMinimum(-90)
        self.alt_centre.setMaximum(90)
        self.alt_centre.setDecimals(8)
        self.alt_centre.setSingleStep(1)
        self.alt_centre.setFixedWidth(100)
        self.alt_centre.valueModified.connect(self.onAltChanged)
        hbox.addWidget(self.alt_centre)
        hbox.addWidget(QtWidgets.QLabel('°', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Alt'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.rotation_from_horiz = DoubleSpinBox()
        self.rotation_from_horiz.setMinimum(-360)
        self.rotation_from_horiz.setMaximum(360)
        self.rotation_from_horiz.setDecimals(8)
        self.rotation_from_horiz.setSingleStep(1)
        self.rotation_from_horiz.setFixedWidth(100)
        self.rotation_from_horiz.valueModified.connect(self.onRotChanged)
        hbox.addWidget(self.rotation_from_horiz)
        hbox.addWidget(QtWidgets.QLabel('°', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Horz rot'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.F_scale = DoubleSpinBox()
        self.F_scale.setMinimum(0)
        self.F_scale.setMaximum(1)
        self.F_scale.setDecimals(8)
        self.F_scale.setSingleStep(0.01)
        self.F_scale.setFixedWidth(100)
        self.F_scale.valueModified.connect(self.onScaleChanged)
        hbox.addWidget(self.F_scale)
        hbox.addWidget(QtWidgets.QLabel('\'/px', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Scale'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.extinction_scale = DoubleSpinBox()
        self.extinction_scale.setMinimum(0)
        self.extinction_scale.setMaximum(100)
        self.extinction_scale.setDecimals(8)
        self.extinction_scale.setSingleStep(0.1)
        self.extinction_scale.setFixedWidth(100)
        self.extinction_scale.valueModified.connect(self.onExtinctionChanged)
        hbox.addWidget(self.extinction_scale)
        hbox.addWidget(QtWidgets.QLabel('', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Extinction'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.lat = DoubleSpinBox()
        self.lat.setMinimum(-360)
        self.lat.setMaximum(360)
        self.lat.setDecimals(8)
        self.lat.setSingleStep(1)
        self.lat.setFixedWidth(100)
        self.lat.valueModified.connect(self.onLatChanged)
        hbox.addWidget(self.lat)
        hbox.addWidget(QtWidgets.QLabel('°', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Lat'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.lon = DoubleSpinBox()
        self.lon.setMinimum(-360)
        self.lon.setMaximum(360)
        self.lon.setDecimals(8)
        self.lon.setSingleStep(1)
        self.lon.setFixedWidth(100)
        self.lon.valueModified.connect(self.onLonChanged)
        hbox.addWidget(self.lon)
        hbox.addWidget(QtWidgets.QLabel('°', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Lon'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.elev = DoubleSpinBox()
        self.elev.setMinimum(0)
        self.elev.setMaximum(1000000)
        self.elev.setDecimals(8)
        self.elev.setSingleStep(100)
        self.elev.setFixedWidth(100)
        self.elev.valueModified.connect(self.onElevChanged)
        hbox.addWidget(self.elev)
        hbox.addWidget(QtWidgets.QLabel('m', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Elev'), hbox)

        self.distortion_type = QtWidgets.QComboBox(self)
        self.distortion_type.addItems(self.gui.platepar.distortion_type_list)
        self.distortion_type.currentIndexChanged.connect(self.onIndexChanged)
        form.addRow(QtWidgets.QLabel('Distortion'), self.distortion_type)

        self.fit_parameters = ArrayTabWidget(platepar=self.gui.platepar)
        self.fit_parameters.valueModified.connect(self.onFitParametersChanged)
        form.addRow(self.fit_parameters)

        self.updatePlatepar()

    def onRefractionToggled(self):
        self.gui.platepar.refraction = self.refraction.isChecked()
        self.sigRefractionToggled.emit()

    def onEqualAspectToggled(self):
        self.gui.platepar.equal_aspect = self.eqAspect.isChecked()
        self.sigEqAspectToggled.emit()

    def onForceDistortionToggled(self):
        self.gui.platepar.force_distortion_centre = self.fdistortion.isChecked()
        self.sigForceDistortionToggled.emit()

    def onLatChanged(self):
        self.gui.platepar.lat = self.lat.value()
        self.sigLocationChanged.emit()

    def onLonChanged(self):
        self.gui.platepar.lon = self.lon.value()
        self.sigLocationChanged.emit()

    def onElevChanged(self):
        self.gui.platepar.elev = self.elev.value()
        self.sigElevChanged.emit()

    def onAzChanged(self):
        self.gui.platepar.az_centre = self.az_centre.value()
        self.sigAzAltChanged.emit()

    def onAltChanged(self):
        self.gui.platepar.alt_centre = self.alt_centre.value()
        self.sigAzAltChanged.emit()

    def onRotChanged(self):
        self.gui.platepar.rotation_from_horiz = self.rotation_from_horiz.value()
        self.sigRotChanged.emit()

    def onScaleChanged(self):
        self.gui.platepar.F_scale = self.F_scale.value()*60
        self.sigScaleChanged.emit()

    def onExtinctionChanged(self):
        self.sigExtinctionChanged.emit()

    def onFitParametersChanged(self):
        # fit parameter object updates platepar by itself
        self.sigFitParametersChanged.emit()

    def onIndexChanged(self):
        text = self.distortion_type.currentText()
        self.gui.platepar.setDistortionType(text, reset_params=False)

        if text == 'poly3+radial':
            self.fit_parameters.changeNumberShown(12)
        elif text == 'radial3':
            self.fit_parameters.changeNumberShown(5)
        elif text == 'radial4':
            self.fit_parameters.changeNumberShown(6)
        elif text == 'radial5':
            self.fit_parameters.changeNumberShown(7)

        if self.gui.platepar.distortion_type.startswith('radial'):
            self.eqAspect.show()
            self.fdistortion.show()
        else:
            self.eqAspect.hide()
            self.fdistortion.hide()

        self.sigFitParametersChanged.emit()

    def updatePlatepar(self):
        """
        Updates QDoubleSpinBox values to the values of the platepar.
        Call this whenever the platepar values are changed
        """
        self.az_centre.setValue(self.gui.platepar.az_centre)
        self.alt_centre.setValue(self.gui.platepar.alt_centre)
        self.rotation_from_horiz.setValue(self.gui.platepar.rotation_from_horiz)
        self.F_scale.setValue(self.gui.platepar.F_scale/60)
        self.fit_parameters.updateValues()
        self.distortion_type.setCurrentIndex(
            self.gui.platepar.distortion_type_list.index(self.gui.platepar.distortion_type))
        self.lat.setValue(self.gui.platepar.lat)
        self.lon.setValue(self.gui.platepar.lon)
        self.elev.setValue(self.gui.platepar.elev)
        self.extinction_scale.setValue(self.gui.platepar.extinction_scale)
        self.refraction.setChecked(self.gui.platepar.refraction)
        self.eqAspect.setChecked(self.gui.platepar.equal_aspect)
        self.fdistortion.setChecked(self.gui.platepar.force_distortion_centre)

        if self.gui.platepar.distortion_type.startswith('radial'):
            self.eqAspect.show()
            self.fdistortion.show()
        else:
            self.eqAspect.hide()
            self.fdistortion.hide()

    def updatePairedStars(self):
        """
        Updates QPushButtons to be enabled/disabled based on the number of paired stars
        Call whenever paired_stars is changed
        """
        self.astrometry_button.setEnabled(len(self.gui.paired_stars) > 0)
        self.photometry_button.setEnabled(len(self.gui.paired_stars) > 3)
        self.fit_astrometry_button.setEnabled(len(self.gui.paired_stars) > 3)


class ArrayTabWidget(QtWidgets.QTabWidget):
    """
    Widget to the right which holds the histogram as well as the parameter manager
    This class does not manipulate their values itself, that is done by accessing
    the variables themselves
    """
    valueModified = QtCore.pyqtSignal()

    def __init__(self, platepar):
        super(ArrayTabWidget, self).__init__()
        self.platepar = platepar

        self.vars = ['x_poly_rev', 'y_poly_rev', 'x_poly_fwd', 'y_poly_fwd']

        self.tabs = [QtWidgets.QWidget() for x in range(4)]
        self.layouts = []
        self.boxes = [[], [], [], []]
        self.labels = [[], [], [], []]

        for i in range(len(self.vars)):
            self.addTab(self.tabs[i], self.vars[i])
            self.setupTab(i)

        self.n_shown = 12

    def changeNumberShown(self, n):
        """
        Change the number of QDoubleSpinBoxes visible

        Arguments:
            n [int]: Number of QDoubleSpinBoxes to be visible
        """
        assert 0 <= n <= 12
        if n == self.n_shown:
            return

        elif n > self.n_shown:
            for i in range(4):
                for j in range(self.n_shown, n):
                    self.layouts[i].insertRow(j, self.labels[i][j], self.boxes[i][j])
                    self.labels[i][j].show()
                    self.boxes[i][j].show()

        elif n < self.n_shown:
            for i in range(4):
                for j in range(n, self.n_shown):
                    self.labels[i][j].hide()
                    self.boxes[i][j].hide()
                    self.layouts[i].removeWidget(self.labels[i][j])
                    self.layouts[i].removeWidget(self.boxes[i][j])

        self.n_shown = n

    def setupTab(self, i):
        layout = QtWidgets.QFormLayout()

        for j in range(12):
            box = ScientificDoubleSpinBox()
            box.setSingleStep(0.5)
            box.setFixedWidth(100)
            box.setValue(getattr(self.platepar, self.vars[i])[j])
            box.valueModified.connect(self.onFitParameterChanged(i, j))
            label = QtWidgets.QLabel("{}[{}]".format(self.vars[i], j))
            layout.addRow(label, box)
            self.boxes[i].append(box)
            self.labels[i].append(label)

        self.setTabText(i, self.vars[i])
        self.tabs[i].setLayout(layout)
        self.layouts.append(layout)

    def onFitParameterChanged(self, i, j):
        def f():
            getattr(self.platepar, self.vars[i])[j] = self.boxes[i][j].value()
            self.valueModified.emit()

        return f

    def updateValues(self):
        for i in range(4):
            for j in range(12):
                self.boxes[i][j].setValue(getattr(self.platepar, self.vars[i])[j])


class SettingsWidget(QtWidgets.QWidget):
    """
    QWidget which displays all of the visual values of the gui. Changing any parameters
    here will not affect the functionality of the gui and will not be saved with savestate.
    """
    sigMaxAveToggled = QtCore.pyqtSignal()
    sigCatStarsToggled = QtCore.pyqtSignal()
    sigCalStarsToggled = QtCore.pyqtSignal()
    sigDistortionToggled = QtCore.pyqtSignal()
    sigInvertToggled = QtCore.pyqtSignal()
    sigGridToggled = QtCore.pyqtSignal()
    sigSelStarsToggled = QtCore.pyqtSignal()
    sigPicksToggled = QtCore.pyqtSignal()
    sigRegionToggled = QtCore.pyqtSignal()

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui

        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(vbox)

        hbox = QtWidgets.QHBoxLayout()
        pixel_group = QtWidgets.QButtonGroup(self)
        self.ave_pixel = QtWidgets.QRadioButton('avepixel')
        self.max_pixel = QtWidgets.QRadioButton('maxpixel')
        self.updateMaxAvePixel()
        self.ave_pixel.released.connect(self.sigMaxAveToggled.emit)
        self.max_pixel.released.connect(self.sigMaxAveToggled.emit)
        pixel_group.addButton(self.ave_pixel)
        pixel_group.addButton(self.max_pixel)
        hbox.addWidget(self.ave_pixel)
        hbox.addWidget(self.max_pixel)
        vbox.addLayout(hbox)

        self.catalog_stars = QtWidgets.QCheckBox('Show Catalog Stars')
        self.catalog_stars.released.connect(self.sigCatStarsToggled.emit)
        self.updateShowCatStars()
        vbox.addWidget(self.catalog_stars)

        self.detected_stars = QtWidgets.QCheckBox('Show Detected Stars')
        self.detected_stars.released.connect(self.sigCalStarsToggled.emit)
        self.updateShowCalStars()
        vbox.addWidget(self.detected_stars)

        self.selected_stars = QtWidgets.QCheckBox('Show Selected Stars')
        self.selected_stars.released.connect(self.sigSelStarsToggled.emit)
        self.updateShowSelStars()
        vbox.addWidget(self.selected_stars)

        self.picks = QtWidgets.QCheckBox('Show Picks')
        self.picks.released.connect(self.sigPicksToggled.emit)
        self.updateShowPicks()
        self.picks.hide()
        vbox.addWidget(self.picks)

        self.region = QtWidgets.QCheckBox('Show Photometry Highlight')
        self.region.released.connect(self.sigRegionToggled.emit)
        self.updateShowRegion()
        self.region.hide()
        vbox.addWidget(self.region)

        self.distortion = QtWidgets.QCheckBox('Show Distortion')
        self.distortion.released.connect(self.sigDistortionToggled.emit)
        self.updateShowDistortion()
        vbox.addWidget(self.distortion)

        self.invert = QtWidgets.QCheckBox('Invert Colors')
        self.invert.released.connect(self.sigInvertToggled.emit)
        try:
            self.updateInvertColours()
        except AttributeError:
            self.invert.setChecked(False)
        vbox.addWidget(self.invert)

        hbox = QtWidgets.QHBoxLayout()
        grid_group = QtWidgets.QButtonGroup()
        self.grid = []
        for i, text in enumerate(['None', 'RaDec Grid', 'AzAlt Grid']):
            button = QtWidgets.QRadioButton(text)
            grid_group.addButton(button)
            button.released.connect(self.onGridChanged)
            hbox.addWidget(button)
            self.grid.append(button)
        self.updateShowGrid()
        vbox.addLayout(hbox)

        form = QtWidgets.QFormLayout()
        vbox.addLayout(form)

        self.img_gamma = DoubleSpinBox()
        self.img_gamma.setSingleStep(0.1)
        self.img_gamma.setDecimals(5)
        try:
            self.updateImageGamma()
        except AttributeError:
            self.img_gamma.setValue(1)
        self.img_gamma.valueModified.connect(self.onGammaChanged)
        form.addRow(QtWidgets.QLabel('Gamma'), self.img_gamma)

        self.lim_mag = DoubleSpinBox()
        self.lim_mag.setSingleStep(0.1)
        self.lim_mag.setMinimum(0)
        self.lim_mag.setDecimals(1)
        self.updateLimMag()
        self.lim_mag.valueModified.connect(self.onLimMagChanged)
        self.lim_mag_label = QtWidgets.QLabel('Lim Mag')
        form.addRow(self.lim_mag_label, self.lim_mag)

        self.std = DoubleSpinBox()
        self.std.setSingleStep(0.1)
        self.std.setMinimum(0)
        self.std.setValue(self.gui.stdev_text_filter)
        self.std.valueModified.connect(self.onStdChanged)
        self.std_label = QtWidgets.QLabel('Filter Res Std')
        form.addRow(self.std_label, self.std)

    def updateMaxAvePixel(self):
        self.ave_pixel.setChecked(self.gui.img_type_flag == 'avepixel')
        self.max_pixel.setChecked(self.gui.img_type_flag == 'maxpixel')

    def updateShowCatStars(self):
        self.catalog_stars.setChecked(self.gui.catalog_stars_visible)

    def updateShowCalStars(self):
        self.detected_stars.setChecked(self.gui.draw_calstars)

    def updateShowSelStars(self):
        self.selected_stars.setChecked(self.gui.selected_stars_visible)

    def updateShowPicks(self):
        self.picks.setChecked(self.gui.pick_marker.isVisible())

    def updateShowRegion(self):
        self.region.setChecked(self.gui.region.isVisible())

    def updateShowDistortion(self):
        self.distortion.setChecked(self.gui.draw_distortion)

    def updateShowGrid(self):
        for i, button in enumerate(self.grid):
            button.setChecked(self.gui.grid_visible == i)

    def updateInvertColours(self):
        self.invert.setChecked(self.gui.img.invert_img)

    def updateImageGamma(self):
        self.img_gamma.setValue(self.gui.img.gamma)

    def updateLimMag(self):
        self.lim_mag.setValue(self.gui.cat_lim_mag)

    def onGammaChanged(self):
        self.gui.img.setGamma(self.img_gamma.value())
        self.gui.img_zoom.setGamma(self.img_gamma.value())
        self.gui.updateLeftLabels()
        self.updateImageGamma()  # gamma may be changed by setGamma

    def onGridChanged(self):
        if self.grid[0].isChecked():
            self.gui.grid_visible = 0
        elif self.grid[1].isChecked():
            self.gui.grid_visible = 1
        else:
            self.gui.grid_visible = 2
        self.sigGridToggled.emit()

    def onLimMagChanged(self):
        self.gui.cat_lim_mag = self.lim_mag.value()
        self.gui.catalog_stars = self.gui.loadCatalogStars(self.gui.cat_lim_mag)
        self.gui.updateLeftLabels()
        self.gui.updateStars()

    def onStdChanged(self):
        self.gui.stdev_text_filter = self.std.value()
        self.gui.photometry()

    def onSkyFit(self):
        self.lim_mag.show()
        self.lim_mag_label.show()
        self.std.show()
        self.std_label.show()
        self.detected_stars.show()
        self.distortion.show()
        self.selected_stars.show()
        self.picks.hide()
        self.region.hide()

        self.gui.selected_stars_visible = False
        self.sigSelStarsToggled.emit()  # toggle makes it true
        self.updateShowSelStars()

        self.gui.draw_calstars = False
        self.sigCalStarsToggled.emit()  # toggle makes it true
        self.updateShowCalStars()

    def onManualReduction(self):
        self.lim_mag.hide()
        self.lim_mag_label.hide()
        self.std.hide()
        self.std_label.hide()
        self.detected_stars.hide()
        self.distortion.hide()
        self.selected_stars.hide()
        self.picks.show()
        if self.gui.img.img_handle.input_type != 'dfn':
            self.region.show()

        self.gui.draw_distortion = True
        self.sigDistortionToggled.emit()  # toggle makes it false
        self.updateShowDistortion()

        self.gui.selected_stars_visible = True
        self.sigSelStarsToggled.emit()  # toggle makes it false
        self.updateShowSelStars()

        self.gui.draw_calstars = True
        self.sigCalStarsToggled.emit()  # toggle makes it false
        self.updateShowCalStars()


# https://jdreaver.com/posts/2014-07-28-scientific-notation-spin-box-pyside.html

_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')


def valid_float_string(string):
    match = _float_re.search(string)
    return match.groups()[0] == string if match else False


class FloatValidator(QtGui.QValidator):
    def validate(self, string, position):
        if valid_float_string(string):
            state = QtGui.QValidator.Acceptable
        elif string == "" or string[position - 1] in 'e.-+':
            state = QtGui.QValidator.Intermediate
        else:
            state = QtGui.QValidator.Invalid
        return state, string, position

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""


class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    buttonPressed = QtCore.pyqtSignal()
    valueModified = QtCore.pyqtSignal()  # press enter or buttonpressed

    def __init__(self, *args, **kwargs):
        """
        Identical to QDoubleSpinBox functionally except has more signals
        so you can tell more of what's happening
        """
        super().__init__(*args, **kwargs)

    def stepBy(self, steps):
        super().stepBy(steps)
        self.buttonPressed.emit()
        self.valueModified.emit()

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        if e.key() == QtCore.Qt.Key_Return:
            self.valueModified.emit()


class ScientificDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    buttonPressed = QtCore.pyqtSignal()
    valueModified = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-1e30)
        self.setMaximum(1e30)
        self.validator = FloatValidator()
        self.setDecimals(1000)
        self.step_size = 1

    def setSingleStep(self, val):
        self.step_size = val

    def singleStep(self):
        return self.step_size

    def validate(self, text, position):
        return self.validator.validate(text, position)

    def fixup(self, text):
        return self.validator.fixup(text)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        return format_float(value)

    def stepBy(self, steps):
        text = self.cleanText()
        groups = _float_re.search(text).groups()
        decimal = float(groups[1])
        decimal += steps*self.singleStep()
        new_string = "{:e}".format(float(str(decimal) + groups[3]))
        self.lineEdit().setText(new_string)

        self.buttonPressed.emit()
        self.valueModified.emit()

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        if e.key() == QtCore.Qt.Key_Return:
            self.valueModified.emit()


def format_float(value):
    """Modified form of the 'g' format specifier."""
    string = "{:e}".format(value)  # .replace("e+", "e")
    # string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string
