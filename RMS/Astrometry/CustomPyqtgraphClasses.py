import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt, QLine, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QPicture, QPainter, QPen, QFont, QTransform, QPainterPath, QBrush, \
    QValidator
from PyQt5.QtWidgets import QApplication, QLineEdit, QWidget, QGridLayout, QDoubleSpinBox, QLabel, \
    QComboBox, QTabWidget, QFormLayout, QHBoxLayout

import time
import re
import sys


class Plus(QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(Plus())

    Consists of two lines with no fill making a plus sign
    """

    def __init__(self):
        QPainterPath.__init__(self)
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


class Cross(QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(Cross())

    Consists of two lines with no fill making a cross
    """

    def __init__(self):
        QPainterPath.__init__(self)
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


class CircleLine(QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(CircleLine())

    Consists of a circle with fill that can be removed (with setBrush(QColor(0,0,0,0))),
    with a line going from the top to the center
    """

    def __init__(self):
        QPainterPath.__init__(self)
        points = np.asarray([(0, -0.5), (0, 0)])
        self.moveTo(*points[0])
        self.lineTo(*points[1])
        self.closeSubpath()

        self.addEllipse(QPoint(0, 0), 0.5, 0.5)


class Crosshair(QPainterPath):
    """
    Used as a symbol for ScatterPlotItem
    ex. item.setSymbol(Crosshair())

    Consists of a circle with fill that can be removed (with setBrush(QColor(0,0,0,0))),
    with four lines going from the top, bottom, left and right to near the center
    """

    def __init__(self):
        QPainterPath.__init__(self)
        points = np.asarray([(0, -0.5), (0, -0.2),
                             (0, 0.5), (0, 0.2),
                             (0.5, 0), (0.2, 0),
                             (-0.5, 0), (-0.2, 0)])

        for i in range(0, len(points), 2):
            self.moveTo(*points[i])
            self.lineTo(*points[i + 1])
        self.closeSubpath()

        self.addEllipse(QPoint(0, 0), 0.5, 0.5)


# custom pyqtgraph items
class PlotLines(pg.GraphicsObject):
    """
    Used to add to a pyqt widget (such as ViewBox), which allows for the plotting of lines
    ex.
    lines = PlotLines()
    widget.addItem(lines)
    """

    def __init__(self, data=None, pxmode=False):
        pg.GraphicsObject.__init__(self)
        self.data = data

        if self.data is None:
            self.data = []
            self.max_x = 0
            self.max_y = 0
        else:
            self.max_x = max([max([x[0], x[2]]) for x in self.data])
            self.max_y = max([max([x[1], x[3]]) for x in self.data])

        self.picture = QPicture()
        self.pxmode = pxmode

        self.generatePicture()

    def setData(self, data):
        """
        Arguments:
            data [list of (float, float, float, float, QPen)]:
                First two floats are x0 and y0, the initial coordinates of the line to draw
                Second two floats are xf and yf, the final coordinates of the line to draw
                QPen argmument is the pen to draw the line with
                A list of these five arguments in a tuple will allow for drawing any number of lines
        """
        self.data = data
        self.max_x = max([max([x[0], x[2]]) for x in self.data])
        self.max_y = max([max([x[1], x[3]]) for x in self.data])
        self.update()

    def generatePicture(self):
        painter = QPainter(self.picture)
        for x0, y0, xnd, ynd, pen in self.data:
            if self.pxmode and self.parentItem():
                pos1 = self.parentItem().mapToDevice(pg.Point(x0, y0))
                pos2 = self.parentItem().mapToDevice(pg.Point(xnd, ynd))
                x0, y0, xnd, ynd = pos1.x(), pos1.y(), pos2.x(), pos2.y()
            painter.setPen(pen)
            painter.drawLine(QLine(x0, y0, xnd, ynd))
        painter.end()

    def paint(self, painter, option, widget=None):
        self.generatePicture()
        t = painter.transform()

        if self.pxmode:  # stays in coordinates according to view without changing size
            painter.setTransform(QTransform(1, 0, t.m13(),
                                            t.m21(), 1, t.m23(),
                                            0, 0, t.m33()))
        painter.drawPicture(QPoint(0, 0), self.picture)

    def boundingRect(self):
        return QRectF(0, 0, self.max_x, self.max_y)


class TextItemList(pg.GraphicsObject):
    """
    Allows for a list of TextItems without having to constantly add items to a widget
    ex.
    text_list = TextItemList()
    text_list.addTextItem(0,0,100,100,'hello')
    text_list.addTextItem(10,10,100,100,'you')
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

        Arguments:
            i [int]: index
        """
        return self.text_list[i]

    def addTextItem(self, *args, **kwargs):
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

    def moveText(self, i, x, y):
        """
        Moves the TextItem at index i to coordinates (x,y) while maintaining all of its properties

        Arguments:
            i [int]: index of TextItem
            x [float]: x coordinate to move TextItem to (depends on pxmode)
            y [float]: y coordinate to move TextItem to (depends on pxmode)
        """
        text = self.getTextItem(i)
        visible = text.isVisible()
        self.setTextItem(i, x, y, *text.wh, text.text,
                         pen=text.pen, font=text.font, align=text.align, pxmode=text.pxmode,
                         background_brush=text.background_brush, background_pen=text.background_pen,
                         margin=text.margin)
        if visible:
            self.getTextItem(i).show()
        else:
            self.getTextItem(i).hide()

    def setTextItem(self, parent, i, *args, **kwargs):
        """
        Replace the TextItem at index i to a TextItem initialized with args and kwargs

        Arguments:
            i [int]: index of TextItem to replace
            args, kwargs: same arguments as __init__ in TextItem
        """
        self.parentItem().scene().removeItem(self.text_list[i])
        self.text_list[i].setParentItem(None)
        self.text_list.insert(i, TextItem(*args, **kwargs))
        self.text_list[i].setParentItem(self.parentItem())
        self.text_list[i].setZValue(self.z)

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
        self.parentItem().scene().removeItem(item)
        item.setParentItem(None)

    def setParentItem(self, parent):
        super().setParentItem(parent)
        for text in self.text_list:
            text.setParentItem(parent)

    def paint(self, painter, option, widget=None):
        for text in self.text_list:
            text.update()

    def boundingRect(self):
        return QRectF()


class TextItem(pg.GraphicsObject):
    def __init__(self, x, y, w, h, text,
                 pen=None, font=None, align=None, pxmode=0,
                 background_brush=None, background_pen=None, margin=None):
        """
        Adds a TextItem

        Arguments:
            x [float]: x coordinate of TextItem (depends on pxmode)
            y [float]: y coordinate of TextItem (depends on pxmode)
            w [float]: width of area to write in (depends on pxmode)
            h [float]: height of area to write in (depends on pxmode)
            text [str]: text to show
            pen [QPen]: pen to write text in
            font [QFont]: font to write text in
            align: Qt.AlignLeft, Qt.AlignRight or Qt.AlignCenter
            pxmode:
                0: x, y, w, and h represent distance and text has a fixed length in distance
                1: x and y respresent distance distance, w and h represent pixels and text has
                    fixed length in pixels
                2: x, y, w and h represent pixels and text has a fixed length in pixels
                3: same as 2 except x any y represent pixels from corner of parent not its axis
            background_brush [QBrush]: Brush used for background (fill)
            background_pen [QPen]: Pen used for background (outline)
            margin [float]: amount of space between sides of background to text
        """

        pg.GraphicsObject.__init__(self)
        self.xy = (x, y)
        self.wh = (w, h)

        self.text = text
        self.font = font
        self.pen = pen
        self.align = align
        self.margin = margin
        self.background_brush = background_brush
        self.background_pen = background_pen

        if pen is None:
            self.pen = QPen(QColor(Qt.white))
        if font is None:
            self.font = QFont()
        if align is None:
            self.align = Qt.AlignLeft
        if margin is None:
            self.margin = 0  # margin does nothing if align is Qt.AlignCenter

        self.pxmode = pxmode
        self.setFlag(self.ItemIgnoresTransformations, self.pxmode != 0)

        self.picture = QPicture()

    def size(self):
        """
        Returns:
             (float, float): width and height of the background
        """
        return self.wh

    def setPos(self, x, y):
        self.xy = x, y
        self.update()

    def setBackgroundBrush(self, brush):
        self.background_brush = brush
        self.update()

    def setBackgroundPen(self, pen):
        self.background_pen = pen
        self.update()

    def setPen(self, pen):
        self.pen = pen
        self.update()

    def setFont(self, font):
        self.font = font
        self.update()

    def setText(self, text):
        self.text = text
        self.update()

    def setAlignment(self, align):
        self.align = align
        self.update()

    def generatePicture(self):
        painter = QPainter(self.picture)
        background = (self.background_pen is not None) or (self.background_brush is not None)
        if background:
            if self.background_pen:
                painter.setPen(self.background_pen)
                painter.setBrush(Qt.NoBrush)
            elif self.background_brush:
                painter.setBrush(self.background_brush)
                painter.setPen(Qt.NoPen)

            if self.background_brush:
                painter.setBrush(self.background_brush)

            painter.drawRect(0, 0, *self.wh)

        painter.setPen(self.pen)
        painter.setFont(self.font)
        if self.align == Qt.AlignLeft or self.align == Qt.AlignRight:
            painter.drawText(self.margin, self.margin,
                             self.wh[0] - 2*self.margin, self.wh[1] - 2*self.margin,
                             self.align, self.text)
        elif self.align == Qt.AlignCenter:
            painter.drawText(0, 0, *self.wh, self.align, self.text)
        else:
            raise KeyError
        painter.end()

    def paint(self, painter, option, widget=None):
        self.generatePicture()

        # transformations
        painter.save()
        t = painter.transform()

        painter.translate(*self.xy)  # transformation is overriden if self.pxmode != 0
        if self.pxmode == 1:  # stays in coordinates according to view without changing size
            pts = self.parentItem().mapToDevice(pg.Point(self.xy[0], self.xy[1]))
            painter.setTransform(QTransform(1, 0, t.m13(),
                                            t.m21(), 1, t.m23(),
                                            pts.x(), pts.y(), t.m33()))
        elif self.pxmode == 2:  # constant amount of pixels from corner of view
            painter.setTransform(QTransform(1, 0, t.m13(),
                                            t.m21(), 1, t.m23(),
                                            t.m31() + self.xy[0], t.m32() + self.xy[1], t.m33()))
        elif self.pxmode == 3:  # constant number of pixels from corner of parent widget
            pts = self.parentItem().mapToDevice(pg.Point(0, 0))
            painter.setTransform(QTransform(1, 0, t.m13(),
                                            t.m21(), 1, t.m23(),
                                            t.m31() + self.xy[0] - pts.x(), t.m32() + self.xy[1] - pts.y(), t.m33()))

        # transform where the center is for convenience
        if self.align == Qt.AlignCenter:
            painter.translate(-self.wh[0]/2, -self.wh[1]/2)
        elif self.align == Qt.AlignRight:
            painter.translate(-self.wh[0], 0)

        painter.drawPicture(0, 0, self.picture)
        painter.restore()

    def boundingRect(self):
        rect = QRectF(0, 0, *self.wh)

        if self.pxmode in [0, 1]:
            origin = self.parentItem().mapToDevice(pg.Point(0, 0))
            pos = self.parentItem().mapToDevice(pg.Point(self.xy[0], self.xy[1]))
            rect.translate(pos.x() - origin.x(), pos.y() - origin.y())
        elif self.pxmode == 2:
            rect.translate(self.xy[0], self.xy[1])
        elif self.pxmode == 3:
            origin = self.parentItem().mapToDevice(pg.Point(0, 0))
            rect.translate(self.xy[0] - origin.x(), self.xy[1] - origin.y())

        if self.align == Qt.AlignCenter:
            rect.translate(-self.wh[0]/2, -self.wh[0]/2)
        elif self.align == Qt.AlignRight:
            rect.translate(-self.wh[0], 0)

        return rect


class ImageItem2(pg.ImageItem):
    # ImageItem that allows for a change in gamma
    def __init__(self, image=None, default_key=None, invert=False, **kwargs):
        """
        ex
        ImageItem2({'maxpixel':data1,'avepixel':data2}, 'avepixel')
        selectImage('maxpixel')

        Arguments:
            image [2D np.array or dict with 2D np.array]:
                data to store and show
            default_key: if image is a dict, use to pick which data to use first
            invert [boolean]: whether to invert image when displaying
            kwargs: other __init__ arguments of pg.ImageItem
        """
        if type(image) == dict:
            self.data_dict = image
            image = self.data_dict[default_key]
        else:
            self.data_dict = None

        pg.ImageItem.__init__(self, image, **kwargs)
        if 'gamma' in kwargs.keys():
            self._gamma = kwargs['gamma']
        else:
            self._gamma = 1

        self.invert_img = invert

    def selectImage(self, key):
        self.image = self.data_dict[key]
        self.updateImage()

    @property
    def gamma(self):
        return self._gamma

    @property
    def data(self):
        return self.image

    def setGamma(self, gamma):
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
    def __init__(self, r, pxmode=False, thickness=1):
        """
        Adds a CursorItem to the point (0,0).

        Arguments:
            r [float]: radius of cursor inner circle
            pxmode [boolean]: whether or not the width of cursor is invariant
            thickness [float]: width of the circles and center dot
        """
        super().__init__()
        self._center = QPoint(0, 0)
        self.last_center = QPoint(0, 0)
        self._r = r
        self.mode = True
        self.thickness = thickness

        self.pxmode = pxmode
        self.picture = QPicture()

    def setMode(self, mode):
        """
        Change the mode of the cursor which changes its appearance

        Arguments:
            mode [boolean]: True is two yellow circles with blue point
                            False is a single purple circle
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
        self.last_center = self.center
        self._center = new_center
        self.update()

    def generatePicture(self):
        if self.pxmode and self.parentItem() is not None:
            origin = self.parentItem().mapToDevice(pg.Point(0, 0))
            pos = self.parentItem().mapToDevice(pg.Point(self.r, self.r))
            r = pos.x() - origin.x()
        else:
            r = self.r

        painter = QPainter(self.picture)
        if self.mode:
            pen = QPen(Qt.yellow, self.thickness, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(0, 0), r, r)

            # pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawEllipse(QPoint(0, 0), 2*r, 2*r)
            painter.setPen(QPen(Qt.blue, 2*self.thickness))
            painter.drawPoint(QPoint(0, 0))
        else:
            pen = QPen(QPen(QColor(128, 0, 128), self.thickness, Qt.SolidLine))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(0, 0), 2*r, 2*r)
        painter.end()

    def paint(self, painter, option, widget=None):
        self.generatePicture()
        painter.translate(self.center.x(), self.center.y())
        if self.pxmode:
            t = painter.transform()
            pts = self.parentItem().mapToDevice(pg.Point(self.center.x(), self.center.y()))
            painter.setTransform(QTransform(1, 0, t.m13(),
                                            t.m21(), 1, t.m23(),
                                            pts.x(), pts.y(), t.m33()))
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if self.pxmode and self.parentItem() is not None:
            origin = self.parentItem().mapToDevice(pg.Point(0, 0))
            pos = self.parentItem().mapToDevice(pg.Point(self.r, self.r))
            r = pos.x() - origin.x()
        else:
            r = self.r

        size = 10
        rect = QRectF(0, 0, size*2*r, size*2*r)

        if self.pxmode:
            origin = self.parentItem().mapToDevice(pg.Point(0, 0))
            pos = self.parentItem().mapFromDevice(pg.Point(rect.width() + origin.x(), rect.height() + origin.y()))
            rect.setWidth(pos.x())
            rect.setHeight(pos.y())

        rect.moveCenter(self.last_center)
        return rect


class HistogramLUTWidget2(pg.HistogramLUTWidget):
    def __init__(self, parent=None, *args, **kwargs):
        pg.HistogramLUTWidget.__init__(self, parent, *args, **kwargs)
        self.item = HistogramLUTItem2(*args, **kwargs)
        self.setCentralItem(self.item)
        self.vb.setMenuEnabled(False)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        modifier = QApplication.keyboardModifiers()
        pos = self.vb.mapSceneToView(event.pos())
        if self.item.movable and modifier == Qt.ControlModifier:
            if event.button() == Qt.LeftButton:
                self.setLevels(pos.y(), self.getLevels()[1])
            elif event.button() == Qt.RightButton:
                self.setLevels(self.getLevels()[0], pos.y())


class HistogramLUTItem2(pg.HistogramLUTItem):
    def __init__(self, *args, **kwargs):
        pg.HistogramLUTItem.__init__(self, *args, **kwargs)
        self.level_images = []
        self.movable = True

    def setImages(self, img):
        """ Store images to automatically set levels that correspond to
            the initial one

        Arguments:
            img: [ImageItem2 or list of ImageItem2]
        """
        if type(img) == ImageItem2:
            self.level_images = [img]
        elif type(img) == list and type(img[0]) == ImageItem2:
            self.level_images = img
        else:
            raise TypeError

    def setMovable(self, boolean):
        """
        Set whether the minimum and maximum values can be changed
        mouse interaction
        """
        self.movable = boolean
        self.region.setMovable(self.movable)

    def paint(self, p, *args):
        # tbh this is an improvement
        pass

    def regionChanging(self):
        super().regionChanging()
        for img in self.level_images:
            img.setLevels(self.getLevels())


class PlateparParameterManager(QWidget):
    """
    QWidget that contains various QDoubleSpinBox's that can be changed to
    manage platepar parameters
    """
    azalt_star_signal = pyqtSignal()
    rot_star_signal = pyqtSignal()
    scale_star_signal = pyqtSignal()
    distortion_signal = pyqtSignal()

    def __init__(self, parent, platepar):
        QWidget.__init__(self, parent)
        self.platepar = platepar

        self.attr_list = {}
        self.setMaximumWidth(300)

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)
        self.setLayout(layout)

        hbox = QHBoxLayout()
        self.az_centre = DoubleSpinBox()
        self.az_centre.setMinimum(-360)
        self.az_centre.setMaximum(360)
        self.az_centre.setDecimals(8)
        self.az_centre.setSingleStep(1)
        self.az_centre.setFixedWidth(100)
        self.az_centre.setValue(self.platepar.az_centre)
        self.az_centre.valueModified.connect(self.azChanged)
        hbox.addWidget(self.az_centre)
        hbox.addWidget(QLabel('degrees', alignment=Qt.AlignLeft))
        layout.addRow(QLabel('Az'), hbox)

        hbox = QHBoxLayout()
        self.alt_centre = DoubleSpinBox()
        self.alt_centre.setMinimum(-360)
        self.alt_centre.setMaximum(360)
        self.alt_centre.setDecimals(8)
        self.alt_centre.setSingleStep(1)
        self.alt_centre.setFixedWidth(100)
        self.alt_centre.setValue(self.platepar.alt_centre)
        self.alt_centre.valueModified.connect(self.altChanged)
        hbox.addWidget(self.alt_centre)
        hbox.addWidget(QLabel('degrees', alignment=Qt.AlignLeft))
        layout.addRow(QLabel('Alt'), hbox)

        hbox = QHBoxLayout()
        self.rotation_from_horiz = DoubleSpinBox()
        self.rotation_from_horiz.setMinimum(-360)
        self.rotation_from_horiz.setMaximum(360)
        self.rotation_from_horiz.setDecimals(8)
        self.rotation_from_horiz.setSingleStep(1)
        self.rotation_from_horiz.setFixedWidth(100)
        self.rotation_from_horiz.setValue(self.platepar.rotation_from_horiz)
        self.rotation_from_horiz.valueModified.connect(self.rotChanged)
        hbox.addWidget(self.rotation_from_horiz)
        hbox.addWidget(QLabel('degrees', alignment=Qt.AlignLeft))
        layout.addRow(QLabel('Rot from horiz'), hbox)

        hbox = QHBoxLayout()
        self.F_scale = DoubleSpinBox()
        self.F_scale.setMinimum(0)
        self.F_scale.setMaximum(1)
        self.F_scale.setDecimals(8)
        self.F_scale.setSingleStep(0.01)
        self.F_scale.setFixedWidth(100)
        self.F_scale.setValue(self.platepar.F_scale/60)
        self.F_scale.valueModified.connect(self.scaleChanged)
        hbox.addWidget(self.F_scale)
        hbox.addWidget(QLabel('arcmin/px', alignment=Qt.AlignLeft))
        layout.addRow(QLabel('Scale'), hbox)

        self.distortion_type = QComboBox(self)
        self.distortion_type.addItems(self.platepar.distortion_type_list)
        self.distortion_type.currentIndexChanged.connect(self.onIndexChanged)
        layout.addRow(QLabel('Distortion'), self.distortion_type)

        self.fit_parameters = ArrayTabWidget(parent=None, platepar=self.platepar)
        self.fit_parameters.valueModified.connect(self.scale_star_signal.emit)  # calls updateStars
        layout.addRow(self.fit_parameters)

    @pyqtSlot()
    def azChanged(self):
        self.platepar.az_centre = self.az_centre.value()
        self.azalt_star_signal.emit()

    @pyqtSlot()
    def altChanged(self):
        self.platepar.alt_centre = self.alt_centre.value()
        self.azalt_star_signal.emit()

    @pyqtSlot()
    def rotChanged(self):
        self.platepar.rotation_from_horiz = self.rotation_from_horiz.value()
        self.rot_star_signal.emit()

    @pyqtSlot()
    def scaleChanged(self):
        self.platepar.F_scale = self.F_scale.value()*60
        self.scale_star_signal.emit()

    @pyqtSlot()
    def onIndexChanged(self):
        text = self.distortion_type.currentText()
        self.platepar.setDistortionType(text, reset_params=False)

        if text == 'poly3+radial':
            self.fit_parameters.changeNumberShown(12)
        elif text == 'radial3':
            self.fit_parameters.changeNumberShown(5)
        elif text == 'radial4':
            self.fit_parameters.changeNumberShown(6)
        elif text == 'radial5':
            self.fit_parameters.changeNumberShown(7)

        self.distortion_signal.emit()

    def updatePlatepar(self):
        """
        Updates QDoubleSpinBox values to the values of the platepar.
        Call this whenever the platepar values are changed
        """
        self.az_centre.setValue(self.platepar.az_centre)
        self.alt_centre.setValue(self.platepar.alt_centre)
        self.rotation_from_horiz.setValue(self.platepar.rotation_from_horiz)
        self.F_scale.setValue(self.platepar.F_scale/60)
        self.fit_parameters.updateValues()


class SettingsWidget(QWidget):
    variablesChanged = pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)


class ArrayTabWidget(QTabWidget):
    """
    Widget to the right which holds the histogram as well as the parameter manager
    This class does not manipulate their values itself, that is done by accessing
    the variables themselves
    """
    valueModified = pyqtSignal()

    def __init__(self, parent=None, platepar=None):
        super(ArrayTabWidget, self).__init__(parent)
        self.platepar = platepar

        self.vars = ['x_poly_rev', 'y_poly_rev', 'x_poly_fwd', 'y_poly_fwd']

        self.tabs = [QWidget() for x in range(4)]
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
        layout = QFormLayout()

        for j in range(12):
            box = ScientificDoubleSpinBox()
            box.setSingleStep(0.5)
            box.setFixedWidth(100)
            box.setValue(getattr(self.platepar, self.vars[i])[j])
            box.valueModified.connect(self.updated(i, j))
            label = QLabel("{}[{}]".format(self.vars[i], j))
            layout.addRow(label, box)
            self.boxes[i].append(box)
            self.labels[i].append(label)

        self.setTabText(i, self.vars[i])
        self.tabs[i].setLayout(layout)
        self.layouts.append(layout)

    def updated(self, i, j):
        def f():
            getattr(self.platepar, self.vars[i])[j] = self.boxes[i][j].value()
            self.valueModified.emit()

        return f

    def updateValues(self):
        for i in range(4):
            for j in range(12):
                self.boxes[i][j].setValue(getattr(self.platepar, self.vars[i])[j])


class RightOptionsTab(QTabWidget):
    def __init__(self, parent=None, platepar=None):
        super(RightOptionsTab, self).__init__(parent)

        self.hist = HistogramLUTWidget2()
        self.param_manager = PlateparParameterManager(parent=None,
                                                      platepar=platepar)

        self.index = 0
        self.maximized = False
        self.setFixedWidth(250)
        self.addTab(self.hist, 'Levels')
        self.addTab(self.param_manager, 'Fit Parameters')

        self.setTabText(0, 'Levels')
        self.setTabText(1, 'Fit Parameters')

        self.setCurrentIndex(self.index)  # redundant
        self.setTabPosition(QTabWidget.East)
        self.setMovable(True)

        self.tabBarClicked.connect(self.onTabBarClicked)

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


# https://jdreaver.com/posts/2014-07-28-scientific-notation-spin-box-pyside.html

_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')


def valid_float_string(string):
    match = _float_re.search(string)
    return match.groups()[0] == string if match else False


class FloatValidator(QValidator):
    def validate(self, string, position):
        if valid_float_string(string):
            state = QValidator.Acceptable
        elif string == "" or string[position - 1] in 'e.-+':
            state = QValidator.Intermediate
        else:
            state = QValidator.Invalid
        return state, string, position

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""


class DoubleSpinBox(QDoubleSpinBox):
    buttonPressed = pyqtSignal()
    valueModified = pyqtSignal()  # press enter or buttonpressed

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
        if e.key() == Qt.Key_Enter - 1:
            self.valueModified.emit()


class ScientificDoubleSpinBox(QDoubleSpinBox):
    buttonPressed = pyqtSignal()
    valueModified = pyqtSignal()

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
        if e.key() == Qt.Key_Enter - 1:
            self.valueModified.emit()


def format_float(value):
    """Modified form of the 'g' format specifier."""
    string = "{:e}".format(value)  # .replace("e+", "e")
    # string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string
