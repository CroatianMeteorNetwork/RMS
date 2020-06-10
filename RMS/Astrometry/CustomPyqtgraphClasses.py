import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt, QLine
from PyQt5.QtGui import QColor, QPicture, QPainter, QPen, QFont, QTransform, QPainterPath, QBrush
from PyQt5.QtWidgets import QApplication


class Plus(QPainterPath):
    def __init__(self):
        QPainterPath.__init__(self)
        points = np.asarray([
            (-0.5, 0),
            (0, 0),
            (0, 0.5),
            (0, 0),
            (0.5, 0),
            (0, 0),
            (0, -0.5),
            (0, 0)
        ])

        self.moveTo(*points[0])
        for x, y in points[1:]:
            self.lineTo(x, y)
        self.closeSubpath()


class Cross(QPainterPath):
    def __init__(self):
        QPainterPath.__init__(self)
        points = np.asarray([
            (-0.5, -0.5),
            (0, 0),
            (-0.5, 0.5),
            (0, 0),
            (0.5, 0.5),
            (0, 0),
            (0.5, -0.5),
            (0, 0)
        ])

        self.moveTo(*points[0])
        for x, y in points[1:]:
            self.lineTo(x, y)
        self.closeSubpath()


class CircleLine(QPainterPath):
    def __init__(self):
        QPainterPath.__init__(self)
        points = np.asarray([(0, -0.5), (0, 0)])
        self.moveTo(*points[0])
        for x, y in points[1:]:
            self.lineTo(x, y)
        self.closeSubpath()
        self.addEllipse(QPoint(0, 0), 0.5, 0.5)


class Crosshair(QPainterPath):
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
    def __init__(self, data=None):
        pg.GraphicsObject.__init__(self)
        self.data = data

        if self.data is None:
            self.data = []

        self.picture = QPicture()
        self.generatePicture()

    def setData(self, data):
        self.data = data
        self.update()

    def generatePicture(self):
        painter = QPainter(self.picture)
        for x0, y0, xnd, ynd, pen in self.data:
            painter.setPen(pen)
            painter.drawLine(QLine(x0, y0, xnd, ynd))
        painter.end()

    def paint(self, painter, option, widget=None):
        self.generatePicture()
        painter.drawPicture(QPoint(0, 0), self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class TextItemList(pg.GraphicsObject):
    """
    Allows for a list of TextItems without having to constantly add items to a widget
    """

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.text_list = []
        self.z = 0

    def getTextItem(self, i):
        return self.text_list[i]

    def addTextItem(self, *args, **kwargs):
        new = TextItem(*args, **kwargs)
        new.setParentItem(self.parentItem())
        new.setZValue(self.z)
        self.text_list.append(new)

    def setZValue(self, z):
        self.z = z
        for text in self.text_list:
            text.setZValue(z)

    def moveText(self, i, x, y):
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

    def setTextItem(self, i, *args, **kwargs):
        self.text_list[i].setParentItem(None)
        self.text_list[i] = TextItem(*args, **kwargs)
        self.text_list[i].setParentItem(self.parentItem())
        self.text_list[i].setZValue(self.z)

    def clear(self):
        while self.text_list:
            self.removeTextItem(0)

    def setParentItem(self, parent):
        super().setParentItem(parent)
        for text in self.text_list:
            text.setParentItem(parent)

    def removeTextItem(self, i):
        item = self.text_list.pop(i)
        item.setParentItem(None)

    def paint(self, painter, option, widget=None):
        for text in self.text_list:
            text.update()

    def boundingRect(self):
        return QRectF()


class TextItem(pg.GraphicsObject):
    def __init__(self, x, y, w, h, text,
                 pen=None, font=None, align=None, pxmode=0,
                 background_brush=None, background_pen=None, margin=None):
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
        return self.wh

    def setBackgroundBrush(self, brush):
        self.background_brush = brush

    def setBackgroundPen(self, pen):
        self.background_pen = pen

    def setPen(self, pen):
        self.pen = pen

    def setFont(self, font):
        self.font = font

    def setText(self, text):
        self.text = text

    def setAlignment(self, align):
        self.align = align

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
        """ example usage
        ImageItem2({'maxpixel':data1,'avepixel':data2},'avepixel'})
        selectImage('maxpixel')
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
        # CHANGED TO IMPLEMENT GAMMA

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
        super().__init__()
        self._center = QPoint(0, 0)
        self.last_center = QPoint(0, 0)
        self._r = r
        self.mode = True
        self.thickness = thickness

        self.pxmode = pxmode
        self.picture = QPicture()
        self.generatePicture()

    def setMode(self, mode):
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
            painter.setPen(QPen(Qt.blue, self.thickness))
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
