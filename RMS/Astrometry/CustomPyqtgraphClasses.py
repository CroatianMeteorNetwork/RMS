from __future__ import division, absolute_import, unicode_literals

import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

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
    def __init__(self, *args, **kwargs):
        pg.ViewBox.__init__(self, *args, **kwargs)

    def keyPressEvent(self, ev):
        """
        The original ViewBox accepts key mappings to + - and =, which prevents their usage.
        This passes the keypressevent on
        """
        ev.ignore()


class ImageItem(pg.ImageItem):
    # ImageItem that allows for a change in gamma
    def __init__(self, image=None, default_key=None, invert=False, **kwargs):
        """
        ex
        ImageItem({'maxpixel':data1,'avepixel':data2}, 'avepixel')
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
        self._center = QtCore.QPoint(0, 0)
        self._r = r
        self.mode = True
        self.thickness = thickness

        self.pxmode = pxmode
        self.picture = QtGui.QPicture()

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
        if self.mode:
            pen = QtGui.QPen(QtCore.Qt.yellow, self.thickness, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPoint(0, 0), r, r)

            # pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawEllipse(QtCore.QPoint(0, 0), 2*r, 2*r)
            painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2*self.thickness))
            painter.drawPoint(QtCore.QPoint(0, 0))
        else:
            pen = QtGui.QPen(QtGui.QColor(128, 0, 128), self.thickness, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPoint(0, 0), 2*r, 2*r)
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
        if self.item.movable and modifier == QtCore.Qt.ControlModifier:
            if event.button() == QtCore.Qt.LeftButton:
                self.setLevels(pos.y(), self.getLevels()[1])
            elif event.button() == QtCore.Qt.RightButton:
                self.setLevels(self.getLevels()[0], pos.y())


class CelestialGrid(pg.PlotCurveItem):
    def __int__(self):
        pg.PlotCurveItem.__init__(self, pen=pg.mkPen((255, 255, 255, 255), style=QtCore.Qt.DotLine))
        # self.


class HistogramLUTItem(pg.HistogramLUTItem):
    def __init__(self, *args, **kwargs):
        pg.HistogramLUTItem.__init__(self, *args, **kwargs)
        self.level_images = []
        self.movable = True

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


class RightOptionsTab(QtWidgets.QTabWidget):
    def __init__(self, gui, parent=None):
        super(RightOptionsTab, self).__init__(parent)

        self.hist = HistogramLUTWidget()
        self.param_manager = PlateparParameterManager(parent=None,
                                                      gui=gui)
        self.settings = SettingsWidget(gui=gui, parent=None)

        self.index = 0
        self.maximized = True
        self.setFixedWidth(250)
        self.addTab(self.hist, 'Levels')
        self.addTab(self.param_manager, 'Fit Parameters')
        self.addTab(self.settings, 'Settings')

        self.setTabText(0, 'Levels')
        self.setTabText(1, 'Fit Parameters')
        self.setTabText(2, 'Settings')

        self.setCurrentIndex(self.index)  # redundant
        self.setTabPosition(QtWidgets.QTabWidget.East)
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

    def __init__(self, gui, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.gui = gui

        full_layout = QtWidgets.QVBoxLayout()
        self.setLayout(full_layout)

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

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        full_layout.addLayout(form)

        self.refraction = QtWidgets.QCheckBox('Refraction')
        self.refraction.released.connect(self.onRefractionToggled)
        form.addWidget(self.refraction)

        self.eqAspect = QtWidgets.QCheckBox('Equal Aspect')
        self.eqAspect.released.connect(self.onEqualAspectToggled)
        form.addWidget(self.eqAspect)

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

        self.fit_parameters = ArrayTabWidget(parent=None, platepar=self.gui.platepar)
        self.fit_parameters.valueModified.connect(self.onFitParametersChanged)
        form.addRow(self.fit_parameters)

        self.updatePlatepar()

    def onRefractionToggled(self):
        self.gui.platepar.refraction = self.refraction.isChecked()
        self.sigRefractionToggled.emit()

    def onEqualAspectToggled(self):
        self.gui.platepar.equal_aspect = self.eqAspect.isChecked()
        self.sigEqAspectToggled.emit()

    def onLatChanged(self):
        self.gui.platepar.lat = self.lat.value()
        # self.gui.view_widget.setFocus()
        self.sigLocationChanged.emit()

    def onLonChanged(self):
        self.gui.platepar.lon = self.lon.value()
        # self.gui.view_widget.setFocus()
        self.sigLocationChanged.emit()

    def onElevChanged(self):
        self.gui.platepar.elev = self.elev.value()
        # self.gui.view_widget.setFocus()
        self.sigElevChanged.emit()

    def onAzChanged(self):
        self.gui.platepar.az_centre = self.az_centre.value()
        # self.gui.view_widget.setFocus()
        self.sigAzAltChanged.emit()

    def onAltChanged(self):
        self.gui.platepar.alt_centre = self.alt_centre.value()
        # self.gui.view_widget.setFocus()
        self.sigAzAltChanged.emit()

    def onRotChanged(self):
        self.gui.platepar.rotation_from_horiz = self.rotation_from_horiz.value()
        # self.gui.view_widget.setFocus()
        self.sigRotChanged.emit()

    def onScaleChanged(self):
        self.gui.platepar.F_scale = self.F_scale.value()*60
        # self.gui.view_widget.setFocus()
        self.sigScaleChanged.emit()

    def onExtinctionChanged(self):
        self.gui.platepar.extinction_scale = self.extinction_scale.value()
        # self.gui.view_widget.setFocus()
        self.sigExtinctionChanged.emit()

    def onFitParametersChanged(self):
        # fit parameter object updates platepar by itself
        # self.gui.view_widget.setFocus()
        self.sigFitParametersChanged.emit()

    def onIndexChanged(self):
        text = self.distortion_type.currentText()
        # self.gui.view_widget.setFocus()
        self.gui.platepar.setDistortionType(text, reset_params=False)

        if text == 'poly3+radial':
            self.fit_parameters.changeNumberShown(12)
        elif text == 'radial3':
            self.fit_parameters.changeNumberShown(5)
        elif text == 'radial4':
            self.fit_parameters.changeNumberShown(6)
        elif text == 'radial5':
            self.fit_parameters.changeNumberShown(7)

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

    def __init__(self, platepar, parent=None):
        super(ArrayTabWidget, self).__init__(parent)
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


class SettingsWidget(QtWidgets.QTabWidget):
    sigMaxAveToggled = QtCore.pyqtSignal()
    sigCatStarsToggled = QtCore.pyqtSignal()
    sigCalStarsToggled = QtCore.pyqtSignal()
    sigDistortionToggled = QtCore.pyqtSignal()
    sigInvertToggled = QtCore.pyqtSignal()
    sigGridToggled = QtCore.pyqtSignal()

    def __init__(self, gui, parent=None):
        QtWidgets.QTabWidget.__init__(self, parent)
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
        form.addRow(QtWidgets.QLabel('Lim Mag'), self.lim_mag)

        self.std = DoubleSpinBox()
        self.std.setSingleStep(0.1)
        self.std.setMinimum(0)
        self.std.setValue(self.gui.stdev_text_filter)
        self.std.valueModified.connect(self.onStdChanged)
        form.addRow(QtWidgets.QLabel('Filter Res Std'), self.std)

    def updateMaxAvePixel(self):
        self.ave_pixel.setChecked(self.gui.img_type_flag == 'avepixel')
        self.max_pixel.setChecked(self.gui.img_type_flag == 'maxpixel')

    def updateShowCatStars(self):
        self.catalog_stars.setChecked(self.gui.catalog_stars_visible)

    def updateShowCalStars(self):
        self.detected_stars.setChecked(self.gui.draw_calstars)

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
