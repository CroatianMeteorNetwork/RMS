from __future__ import division, absolute_import, unicode_literals

import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Import the correct version of pyqtgraph function module, depending on the version
try:
    import pyqtgraph.fn as pgfn
    
except ImportError:
    import pyqtgraph.functions as pgfn
    
from RMS.Astrometry.Conversions import AER2LatLonAlt
from RMS.Formats.FFfile import reconstructFrame as reconstructFrameFF
from RMS.Routines import Image
from RMS.Routines.DebruijnSequence import findAllInDeBruijnSequence, generateDeBruijnSequence
from RMS.Routines.SkyFitHelp import HELP_STYLE, buildHelpHome, buildHelpTopic

import time
import re
import sys


class _CornerHelpOverlay(QtCore.QObject):
    """ Keeps a help button pinned to the top-right corner of a host widget without taking up any
        layout space (so it never shifts the tab content down). """

    def __init__(self, host, button, margin):
        super(_CornerHelpOverlay, self).__init__(host)
        self.host = host
        self.button = button
        self.margin = margin

    def reposition(self):
        self.button.move(self.host.width() - self.button.width() - self.margin, self.margin)
        self.button.raise_()

    def eventFilter(self, obj, event):
        if obj is self.host and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.reposition()
        return False


class ScaledSizeHelper:
    """Helper mixin for calculating sizes that scale with font/DPI settings.

    Use this to replace hardcoded pixel values with font-relative sizes.
    """

    # Reference character width at 96 DPI (typical default)
    _REF_CHAR_WIDTH = 8
    _REF_LINE_HEIGHT = 16

    def scaledWidth(self, chars):
        """Calculate width in pixels for given number of characters."""
        fm = QtGui.QFontMetrics(self.font())
        return int(fm.averageCharWidth() * chars)

    def scaledHeight(self, lines):
        """Calculate height in pixels for given number of lines."""
        fm = QtGui.QFontMetrics(self.font())
        return int(fm.height() * lines)

    def scaledMargins(self, chars_h=0.5, lines_v=0.25):
        """Calculate margins scaled to font size.

        Args:
            chars_h: Horizontal margin in character widths
            lines_v: Vertical margin in line heights

        Returns:
            Tuple of (left, top, right, bottom) margins in pixels
        """
        fm = QtGui.QFontMetrics(self.font())
        h_margin = int(fm.averageCharWidth() * chars_h)
        v_margin = int(fm.height() * lines_v)
        return (h_margin, v_margin, h_margin, v_margin)

    def scaledSpacing(self, fraction=0.5):
        """Calculate spacing scaled to font size.

        Args:
            fraction: Spacing as fraction of line height

        Returns:
            Spacing in pixels
        """
        fm = QtGui.QFontMetrics(self.font())
        return int(fm.height() * fraction)

    def makeHelpButton(self, topic, tooltip="Open the related help page"):
        """ Create a small circular blue "i" info button that opens a Help topic.

        The widget must have a ``self.gui`` attribute exposing ``openHelpTopic(topic_id)``.
        """
        btn = QtWidgets.QToolButton()
        btn.setText("i")
        btn.setToolTip(tooltip)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        d = max(self.scaledHeight(1.0), 13)
        btn.setFixedSize(d, d)
        btn.setStyleSheet(
            "QToolButton { border: none; border-radius: %dpx; background-color: #1a73e8; "
            "color: white; font-weight: bold; font-style: italic; font-family: serif; "
            "font-size: %dpx; } "
            "QToolButton:hover { background-color: #1666c1; }" % (int(d/2), max(int(d*0.62), 8))
        )
        btn.clicked.connect(lambda: self.gui.openHelpTopic(topic))
        return btn

    def addCornerHelpButton(self, topic, tooltip="Open the help page for this tab"):
        """ Pin a small circular "i" help button to the widget's top-right corner.

        The button floats over the content (it is parented to the widget, not added to a layout),
        so it does not take up layout space or shift the tab content down. An event filter keeps it
        in the corner as the widget resizes.
        """
        btn = self.makeHelpButton(topic, tooltip)
        btn.setParent(self)
        margin = max(self.scaledSpacing(0.15), 2)
        overlay = _CornerHelpOverlay(self, btn, margin)
        self.installEventFilter(overlay)
        # Keep a reference so the filter isn't garbage-collected
        self._help_overlay = overlay
        overlay.reposition()
        btn.show()
        return btn


def qmessagebox(message="", title="Error", message_type="warning"):
    msg = QtWidgets.QMessageBox()
    if message_type == "warning":
        msg.setIcon(QtWidgets.QMessageBox.Warning)
    elif message_type == "error":
        msg.setIcon(QtWidgets.QMessageBox.Critical)
    else:
        msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle(title)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()


class QHSeparationLine(QtWidgets.QFrame):
  """ A horizontal separation line """
  def __init__(self):
    super().__init__()
    self.setMinimumWidth(1)
    self.setFixedHeight(20)
    self.setFrameShape(QtWidgets.QFrame.HLine)
    self.setFrameShadow(QtWidgets.QFrame.Sunken)
    self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
    return


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
        text.setVisible(self.isVisible())
        self.text_list.append(text)

    def addNewTextItem(self, *args, **kwargs):
        """
        Has the same arguments as __init__ in TextItem
        """
        new = TextItem(*args, **kwargs)
        new.setParentItem(self.parentItem())
        new.setZValue(self.z)
        new.setVisible(self.isVisible())
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
        # Get scene once for efficiency
        try:
            scene = self.parentItem().scene()
        except:
            scene = None

        # Remove all items (iterate in reverse to avoid O(n²) from pop(0))
        while self.text_list:
            item = self.text_list.pop()  # Pop from end is O(1)
            if scene is not None:
                try:
                    scene.removeItem(item)
                except:
                    pass
            item.setParentItem(None)

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

    def setVisible(self, visible):
        super().setVisible(visible)
        for text in self.text_list:
            text.setVisible(visible)

    def setInteractionEnabled(self, enabled):
        """
        Enable or disable text interaction (hyperlinks) on all TextItems in the list.

        Arguments:
            enabled [bool]: True to enable interaction, False to disable.
        """
        for text in self.text_list:
            if hasattr(text, 'setInteraction'):
                text.setInteraction(enabled)

    def paint(self, painter, option, widget=None):
        for text in self.text_list:
            text.update()

    def boundingRect(self):
        return QtCore.QRectF()


class TextItem(pg.TextItem):
    def __init__(self, text='', color=(200, 200, 200), html=None, anchor=(0, 0),
                 border=None, fill=None, angle=0, rotateAxis=None, interaction=True):
        pg.TextItem.__init__(self, text, color, html, anchor, border, fill, angle, rotateAxis)
        if interaction:
            self.textItem.setOpenExternalLinks(True)
            self.textItem.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        else:
            self.textItem.setOpenExternalLinks(False)
            self.textItem.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)

    def setInteraction(self, enabled):
        """
        Enable or disable text interaction (hyperlinks, selection).

        Arguments:
            enabled [bool]: True to enable interaction, False to disable.
        """
        if enabled:
            self.textItem.setOpenExternalLinks(True)
            self.textItem.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        else:
            self.textItem.setOpenExternalLinks(False)
            self.textItem.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)

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
        self.panning_enabled = True  # Can be disabled for mask editing etc.

    def keyPressEvent(self, ev):
        """
        The original ViewBox accepts key mappings to + - and =, which prevents their usage.
        This passes the keypressevent on
        """
        ev.ignore()

    def mouseReleaseEvent(self, event):
        self.sigMouseReleased.emit(event)
        if self.panning_enabled:
            super().mouseReleaseEvent(event)

    def mousePressEvent(self, event):
        self.sigMousePressed.emit(event)
        if self.panning_enabled:
            super().mousePressEvent(event)
        else:
            event.accept()  


    def wheelEventModified(self, ev, axis=None):

        # Get the range of X and Y of the current view
        x_range, y_range = self.viewRange()


        # Read the wheel direction
        try:
            delta = ev.delta()
        except AttributeError:
            delta = ev.angleDelta().y() 

        # Don't zoom if it's past the limits
        if not ((self.state['limits']['xLimits'] == [round(x, 8) for x in x_range] or
                 self.state['limits']['yLimits'] == [round(y, 8) for y in y_range])
                and delta < 0):

            # On some systems, some versions of code produce an error with this, and work well without
            #   this function
            try:
                super().wheelEvent(ev, axis)
            except AttributeError:
                pass


class ImageItem(pg.ImageItem):
    sigLevelsChanged = QtCore.pyqtSignal()

    # ImageItem that provides an interface around img_handle
    def __init__(self, img_handle=None, saturation_mask=None, **kwargs):
        """
        Makes an image item with img_handle, with the default image of avepixel

        Keyword arguments:
            image_handle: [InputType]
            saturation_mask: [ImageItem] Mask for painting saturation on the screen. None by default.
            invert: [boolean] whether to invert image when displaying
            gamma: [float]
            dark:
            flat_struct:
            kwargs: other __init__ arguments of pg.ImageItem
        """
        self.img_handle = img_handle

        # Display-LUT state. Replaces the old copy-pasted render() override that
        #   reached into pyqtgraph internals (self._effectiveLut etc.), which broke on
        #   pyqtgraph >= 0.13. _base_lut holds any LUT pushed by the histogram (None for
        #   a plain grayscale gradient); gamma and inversion are composed on top of it.
        self._base_lut = None
        self._gamma = 1
        self.invert_img = False

        if 'saturation_threshold' in kwargs:
            self.saturation_threshold = kwargs.pop('saturation_threshold')
        else:
            self.saturation_threshold = None

        pg.ImageItem.__init__(self, image=None, **kwargs)

        self.saturation_mask = saturation_mask

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

        # Apply the initial display LUT now that gamma/inversion are known
        self._applyDisplayLut(update=False)

        if img_handle is not None:
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

        maxpixel = np.swapaxes(maxpixel, 0, 1)

        if self.dark is not None:
            maxpixel = Image.applyDark(maxpixel, self.dark)
        if self.flat_struct is not None:
            maxpixel = Image.applyFlat(maxpixel, self.flat_struct)

        self.setImage(maxpixel)
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

        avepixel = np.swapaxes(avepixel, 0, 1)

        if self.dark is not None:
            avepixel = Image.applyDark(avepixel, self.dark)
        if self.flat_struct is not None:
            avepixel = Image.applyFlat(avepixel, self.flat_struct)

        self.setImage(avepixel)
        self.img_showing = 'avepixel'


    def setImage(self, *args, **kwargs):

        # Set the saturation mask
        if len(args) > 0:

            img = args[0]

            # Apply a saturation mask, if given
            if self.saturation_mask is not None:
                
                # Use the saturation threshold passed from SkyFit, or fallback to config/default
                saturation_threshold = None
                
                if self.saturation_threshold is not None:
                    saturation_threshold = self.saturation_threshold
                elif self.img_handle is not None and hasattr(self.img_handle, 'config') and hasattr(self.img_handle.config, 'bit_depth'):
                    saturation_threshold = int(round(0.98*(2**self.img_handle.config.bit_depth - 1)))
                else:
                    saturation_threshold = int(round(0.98*(2**(8*img.itemsize) - 1)))

                saturates = img > saturation_threshold

                self.saturation_mask.image[:, :] = 0
                
                # Set red colour on for saturation
                self.saturation_mask.image[saturates, 0] = 255
                self.saturation_mask.image[saturates, 1] = 0
                self.saturation_mask.image[saturates, 2] = 0

                # Set alpha on to turn on the mask, just a light shading
                self.saturation_mask.image[saturates, 3] = 32

                self.saturation_mask.setImage(self.saturation_mask.image)


        super().setImage(*args, **kwargs)


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

        # Load the frame
        frame = self.img_handle.loadFrame(avepixel=True)

        if frame is not None:

            # Adding background to FR files
            if self.img_handle.name()[:2] == 'FR':

                # Save the original frame number
                current_frame = self.img_handle.current_frame

                original_index = self.img_handle.current_ff_index
                original_time = self.img_handle.currentTime()

                # Find the FF file which corresponds to the FR file
                for index in range(len(self.img_handle.ff_list)):
                    if index == original_index:
                        continue

                    # If there is an FF file present, add the FF avepixel to the reconstructed FR frame
                    #   background
                    self.img_handle.current_ff_index = index
                    if (original_time == self.img_handle.currentTime()) \
                        and (self.img_handle.name()[:2] == 'FF'):

                        # Load the FF file
                        ff = self.img_handle.loadChunk()

                        # Reconstruct the given frame
                        ff_frame = reconstructFrameFF(ff, current_frame, avepixel=True)

                        # Old method where only the avepixel was pasted
                        # ff_frame = self.img_handle.loadChunk().avepixel

                        # Paste the background to pixels outside the FR cutout
                        frame = frame + ff_frame*(frame == 0)
                        break

                self.img_handle.current_ff_index = original_index

            frame = np.swapaxes(frame, 0, 1)

            if self.dark is not None:
                frame = Image.applyDark(frame, self.dark)
            if self.flat_struct is not None:
                frame = Image.applyFlat(frame, self.flat_struct)

            self.setImage(frame)
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

    def getAutolevels(self, lower=0.1, upper=99.95, ignoretopperc=10):

        if self.image is None or self.image.size == 0:
            return 0, 255

        # Ignore the top 10% of the image pixel brightness (from the maximum) to avoid auto leveling on
        #  saturated pixels
        max_level = np.max(self.image)
        ignore_level = (100 - ignoretopperc)*max_level/100
        
        img_filtered = self.image[self.image < ignore_level]

        # Validation: If the image is saturated or flat, the filtered image might be empty
        if img_filtered.size == 0:
            return np.percentile(self.image, lower), np.percentile(self.image, upper)

        return np.percentile(img_filtered, lower), np.percentile(img_filtered, upper)

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

        self._applyDisplayLut()

    def updateGamma(self, factor):
        self.setGamma(self.gamma*factor)

    def invert(self):
        self.invert_img = not self.invert_img
        self._applyDisplayLut()

    def setLevels(self, levels, update=True):
        super().setLevels(levels, update)
        self.sigLevelsChanged.emit()

    def setLookupTable(self, lut, update=True):
        # The histogram pushes its gradient LUT here (None for a plain grayscale
        #   gradient). Keep it as the base so gamma/inversion can be re-composed on top
        #   whenever they change, then hand pyqtgraph the effective LUT.
        self._base_lut = lut
        super().setLookupTable(self._composeDisplayLut(lut), update=update)

    def _applyDisplayLut(self, update=True):
        """ Rebuild and apply the effective display LUT from the current base LUT,
            gamma and inversion. Used instead of overriding render(), so we no longer
            depend on pyqtgraph render internals. """
        super().setLookupTable(self._composeDisplayLut(self._base_lut), update=update)

    def _composeDisplayLut(self, base_lut):
        """ Build the grayscale lookup table that bakes in gamma and inversion.

        The old render() applied gamma to the post-levels 8-bit luminance and forced
        R = G = B. pyqtgraph maps levels -> LUT index, so a 256-entry grayscale ramp
        carrying gamma/inversion reproduces gamma(rescale(value)) exactly (for 8-bit the
        LUT index equals the rescaled value). An optional base LUT (e.g. a non-trivial
        histogram gradient) is composed underneath.
        """
        if base_lut is None:
            n = 256
            rgb = np.repeat(np.linspace(0, 255, n)[:, None], 3, axis=1)
            alpha = None
        else:
            base_lut = np.asarray(base_lut)
            rgb = base_lut[:, :3].astype(float)
            alpha = base_lut[:, 3:4] if base_lut.shape[1] == 4 else None

        # Gamma on the displayed luminance, then optional inversion
        out = np.clip(np.power(rgb/255.0, 1.0/self._gamma)*255.0, 0, 255)
        if self.invert_img:
            out = 255.0 - out
        out = out.astype(np.ubyte)

        if alpha is not None:
            out = np.concatenate([out, alpha.astype(np.ubyte)], axis=1)

        return out


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

        rect = QtCore.QRect(-3*int(self.r), -3*int(self.r), 6*int(self.r), 6*int(self.r))
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


class PointingIndicator(pg.GraphicsObject):
    """ A HUD glyph pinned at the optical-axis marker that shows:

          - a zenith arrow: points toward the zenith on screen (its angle conveys the camera roll),
          - an elevation notch: a tick sliding along the arrow from the centre (horizon, 0 deg) to the
            tip (zenith, 90 deg), marking the apparent elevation of the optical centre,
          - an Az/Alt readout: the apparent azimuth and altitude of the optical centre,
          - a horizon bar through the optical axis (along the East-West horizon, so it doubles as a
            horizon/roll indicator). Its length is one WASD pan step on screen, and it carries an azimuth
            compass: the middle is North, little notches mark West/East, and a notch slides to the
            current azimuth.

        The arrow and readout are a constant pixel size (ItemIgnoresTransformations), while the bar and
        its notches scale with the image zoom; the whole glyph stays anchored to the optical-axis pixel.
    """

    def __init__(self, arrow_length=52):
        """
        Arguments:
            arrow_length: [float] Length of the zenith arrow in screen pixels.
        """
        super().__init__()

        # Keep the glyph a constant device-pixel size, anchored at setPos()
        self.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, True)

        self.arrow_length = float(arrow_length)

        # State (screen frame: +x right, +y up)
        self.angle = 90.0          # zenith direction (deg)
        self.east_angle = 0.0      # screen direction of increasing azimuth (East along the horizon, deg)
        self.azimuth = 0.0         # apparent azimuth of the optical centre (deg)
        self.elevation = 0.0       # apparent elevation of the optical centre (deg)
        self.step_px = 0.0         # one WASD step in screen pixels
        self.precision = 0         # decimals for the Az/Alt readout (FOV-dependent)
        self.valid_zenith = True   # False when the centre is at/near the zenith

        # Colours (muted, so the glyph stays unobtrusive over the image)
        self.arrow_color = QtGui.QColor(170, 175, 180)
        self.notch_color = QtGui.QColor(210, 215, 220)
        self.step_color = QtGui.QColor(255, 255, 255, 90)
        self.text_color = QtGui.QColor(185, 190, 195)

        # Compact, fixed-size font for the Az/Alt readout (independent of the system default)
        self.font = QtGui.QFont()
        self.font.setPointSize(9)


    def setData(self, angle, east_angle, azimuth, elevation, step_px, precision, valid_zenith):
        self.angle = float(angle)
        self.east_angle = float(east_angle)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.step_px = float(step_px)
        self.precision = int(precision)
        self.valid_zenith = bool(valid_zenith)
        self.prepareGeometryChange()
        self.update()


    def refresh(self):
        """ Recompute geometry and repaint. Connect this to the view's range-change signal so the
            step bar (which scales with zoom) is redrawn with up-to-date bounds. """
        self.prepareGeometryChange()
        self.update()


    def _deviceScaleX(self):
        """ Device pixels per one image pixel (data unit) along X at the current zoom. Used to scale the
            step bar, which represents a real on-screen distance, while the arrow/notch stay fixed size. """
        parent = self.parentItem()
        if parent is None:
            return 1.0
        try:
            o = parent.mapToDevice(pg.Point(0.0, 0.0))
            ex = parent.mapToDevice(pg.Point(1.0, 0.0))
            if (o is None) or (ex is None):
                return 1.0
            return abs(ex.x() - o.x())
        except Exception:
            return 1.0


    def boundingRect(self):
        # Cover the arrow (plus arrowhead and text margin) and the horizontal step bar (zoom-scaled)
        bar_half = 0.5*self.step_px*self._deviceScaleX()
        reach = max(self.arrow_length + 34.0, bar_half + 24.0, 122.0)
        return QtCore.QRectF(-reach, -reach, 2*reach, 2*reach)


    def paint(self, painter, option, widget=None):

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        L = self.arrow_length

        # Step-size / horizon / azimuth bar: a line through the optical-axis centre, oriented along the
        # horizon (East-West). It doubles as (a) a horizon/roll indicator, (b) a pan step-size gauge -- its
        # length is one WASD step on screen, so it scales with the image zoom -- and (c) an azimuth compass:
        # the middle is North, little notches mark West and East (+/- 90 deg), and a longer notch slides to
        # the current azimuth. The compass spans +/-180 deg over the bar (the ends are South).
        if self.step_px > 0.5:
            half = 0.5*self.step_px*self._deviceScaleX()
            ea = np.radians(self.east_angle)
            ux, uy = np.cos(ea), -np.sin(ea)         # unit toward East along the horizon (device coords)
            px_, py_ = -uy, ux                       # unit perpendicular (for the notch ticks)

            pen = QtGui.QPen(self.step_color, 2.0, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(-half*ux, -half*uy), QtCore.QPointF(half*ux, half*uy))
            # End caps
            cap = 4.0
            for s in (-1.0, 1.0):
                painter.drawLine(QtCore.QPointF(s*half*ux - cap*px_, s*half*uy - cap*py_),
                                 QtCore.QPointF(s*half*ux + cap*px_, s*half*uy + cap*py_))

            if self.valid_zenith:
                def _notch(frac, length, pen):
                    # Perpendicular tick at the given fraction (-1..1) along the bar from the centre (N)
                    f = min(max(frac, -1.0), 1.0)
                    bx, by = f*half*ux, f*half*uy
                    painter.setPen(pen)
                    painter.drawLine(QtCore.QPointF(bx - length*px_, by - length*py_),
                                     QtCore.QPointF(bx + length*px_, by + length*py_))

                # Little reference notches: West (-90 deg) and East (+90 deg) at +/- half the bar
                ref_pen = QtGui.QPen(self.step_color, 1.5, QtCore.Qt.SolidLine)
                _notch(-0.5, 3.0, ref_pen)
                _notch(+0.5, 3.0, ref_pen)

                # Current-azimuth notch (azimuth wrapped to +/-180 deg, mapped over the bar)
                az_frac = (((self.azimuth + 180.0)%360.0) - 180.0)/180.0
                _notch(az_frac, 6.0, QtGui.QPen(self.notch_color, 2.2, QtCore.Qt.SolidLine))

        # Az/Alt readout, in the middle just left of the optical-axis plus (right-aligned, two lines).
        # 'Az' = azimuth (+E of due N), 'Alt' = altitude/elevation. The decimal precision scales with the
        # FOV (coarser for wide/all-sky, finer for narrow fields).
        dec = max(self.precision, 0)
        az_str = ("Az {:." + str(dec) + "f}°").format(self.azimuth)
        alt_str = ("Alt {:." + str(dec) + "f}°").format(self.elevation)
        tw = 100.0
        th = 20.0           # tall enough to not clip glyph descenders / the degree sign
        tx = -tw - 15.0     # right edge clears the optical-axis plus arm
        painter.setFont(self.font)
        painter.setPen(QtGui.QPen(self.text_color))
        # Two well-separated lines straddling the centre (Alt above, Az below)
        painter.drawText(QtCore.QRectF(tx, -30.0, tw, th),
                         QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, alt_str)
        painter.drawText(QtCore.QRectF(tx, 2.0, tw, th),
                         QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, az_str)

        # Direction in device coordinates (screen-up = device -Y)
        ang = np.radians(self.angle)
        dx, dy = np.cos(ang), -np.sin(ang)

        if not self.valid_zenith:
            # Near the zenith the direction is undefined: draw a dashed ring instead of an arrow
            pen = QtGui.QPen(self.arrow_color, 1.6, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(0.0, 0.0), 0.5*L, 0.5*L)
            return

        # Shaft (originates from the centre of the optical-axis plus, points toward the zenith)
        x1, y1 = L*dx, L*dy
        pen = QtGui.QPen(self.arrow_color, 2.2, QtCore.Qt.SolidLine)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QtCore.QPointF(0.0, 0.0), QtCore.QPointF(x1, y1))

        # Arrowhead (two short barbs at the tip)
        head = 9.0
        spread = np.radians(26.0)
        for s in (+1, -1):
            ba = ang + np.pi + s*spread
            bx, by = x1 + head*np.cos(ba), y1 - head*np.sin(ba)
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(bx, by))

        # Elevation notch: tick perpendicular to the shaft, sliding centre->tip with elevation 0->90 deg
        f = min(max(self.elevation/90.0, 0.0), 1.0)
        # Position along the shaft
        px, py = f*L*dx, f*L*dy
        # Perpendicular direction (screen) rotated 90 deg
        perp = ang + np.pi/2.0
        pwx, pwy = np.cos(perp), -np.sin(perp)
        nw = 6.0
        painter.setPen(QtGui.QPen(self.notch_color, 2.2, QtCore.Qt.SolidLine))
        painter.drawLine(QtCore.QPointF(px - nw*pwx, py - nw*pwy),
                         QtCore.QPointF(px + nw*pwx, py + nw*pwy))


class HistogramLUTWidget(pg.HistogramLUTWidget):
    def __init__(self, gui, parent=None, *args, **kwargs):

        self.gui = gui

        pg.HistogramLUTWidget.__init__(self, parent, *args, **kwargs)
        self.item = HistogramLUTItem(*args, **kwargs)
        self.setCentralItem(self.item)
        self.vb.setMenuEnabled(False)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        modifier = QtWidgets.QApplication.keyboardModifiers()
        pos = self.vb.mapSceneToView(event.pos())
        if self.item.region.movable and modifier == QtCore.Qt.ControlModifier:
            self.item.exitAutoLevels()
            if event.button() == QtCore.Qt.LeftButton:
                self.setLevels(pos.y(), self.getLevels()[1])
            elif event.button() == QtCore.Qt.RightButton:
                self.setLevels(self.getLevels()[0], pos.y())

        # Set focus back on the image window
        self.gui.view_widget.setFocus()


class HistogramLUTItem(pg.HistogramLUTItem):

    # Emitted with the new state whenever auto levels is toggled (button or Ctrl+A)
    sigAutoLevelsToggled = QtCore.pyqtSignal(bool)

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

        self.imageItem().sigLevelsChanged.connect(
            lambda: [i.setLevels(self.imageItem().getLevels()) for i in self.level_images])

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
        self.sigAutoLevelsToggled.emit(self.auto_levels)

    def exitAutoLevels(self):
        """
        Drop out of auto levels mode, keeping the auto-determined levels as the manual levels
        """
        if self.auto_levels:
            self.auto_levels = False
            self.saved_manual_levels = self.getLevels()
            self.sigAutoLevelsToggled.emit(self.auto_levels)

    def paint(self, p, *args):
        # tbh this is an improvement
        pass

    def regionChanging(self):
        # Doesn't update the image when moving the region, only on release

        # A user drag on the region while auto levels are on drops to manual mode, keeping the
        #   auto-determined levels as the starting point
        if self.auto_levels and (self.region.moving or any(l.moving for l in self.region.lines)):
            self.exitAutoLevels()

    def imageChanged(self, autoLevel=False, autoRange=False):
        if not self.auto_levels:
            self.saved_manual_levels = self.getLevels()
        super().imageChanged(autoLevel, autoRange)
        if self.auto_levels:
            self.setLevels(*self.imageItem().getAutolevels())
        else:
            self.setLevels(*self.saved_manual_levels)


class RightOptionsTab(QtWidgets.QTabWidget, ScaledSizeHelper):
    """
    Tab widget which initializes and holds each of the tabs. They can be accessed with
    self.hist, self.param_manager, self.debruijn and self.settings
    """
    # Signal emitted when tab changes: (old_index, new_index)
    sigTabChanged = QtCore.pyqtSignal(int, int)

    # Tab width in characters (scales with font)
    TAB_WIDTH_CHARS = 32
    TAB_MINIMIZED_CHARS = 2

    # The Help tab is shown at this multiple of the normal width (more readable docs)
    HELP_WIDTH_MULT = 2

    def __init__(self, gui):
        super(RightOptionsTab, self).__init__()

        self.gui = gui

        self.hist = HistogramLUTWidget(gui)
        self.param_manager = PlateparParameterManager(gui)
        self.geolocation = GeolocationWidget(gui)
        self.star_detection = StarDetectionWidget(gui)
        self.mask = MaskWidget(gui)
        self.settings = SettingsWidget(gui)
        self.help = HelpWidget(gui)
        self.debruijn = DebruijnSequenceManager(gui)

        self.index = 0
        self.maximized = True
        self.setFixedWidth(self.scaledWidth(self.TAB_WIDTH_CHARS))

        # Levels tab: auto-levels toggle button above the histogram. The button state stays in
        # sync with the Ctrl+A shortcut through sigAutoLevelsToggled.
        self.levels_tab = QtWidgets.QWidget()
        levels_layout = QtWidgets.QVBoxLayout()
        levels_layout.setContentsMargins(*self.scaledMargins(0.3, 0.2))
        levels_layout.setSpacing(self.scaledSpacing(0.3))
        self.auto_levels_button = QtWidgets.QPushButton('Auto Levels')
        self.auto_levels_button.setCheckable(True)
        self.auto_levels_button.setToolTip('Toggle automatic image levels (Ctrl+A)')
        self.auto_levels_button.clicked.connect(lambda checked: self.hist.item.toggleAutoLevels())
        self.hist.item.sigAutoLevelsToggled.connect(self.auto_levels_button.setChecked)
        levels_layout.addWidget(self.auto_levels_button)
        levels_layout.addWidget(self.hist)
        self.levels_tab.setLayout(levels_layout)

        # Wrap the form-style tabs in scroll areas so that short windows scroll the content
        # instead of compressing it into unreadability. Levels stretches naturally with the
        # window and Help scrolls by itself, so they stay unwrapped.
        self._tab_scroll_wrappers = {}
        for panel in (self.param_manager, self.geolocation, self.star_detection,
                      self.mask, self.settings, self.debruijn):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scroll.setWidget(panel)

            # The tab column has a FIXED width; as direct tab pages the panels simply
            # compressed their content when their minimum width exceeded it. A scroll area
            # would instead enforce the panel's minimum width and clip the right edge
            # (corner help button first). The Ignored horizontal policy makes the scroll
            # area size the panel to the viewport width unconditionally, restoring the
            # graceful compression, while the vertical direction still scrolls.
            panel.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                panel.sizePolicy().verticalPolicy())
            self._tab_scroll_wrappers[panel] = scroll

        self.addTab(self.levels_tab, 'Levels')
        self.addTab(self.tabWidgetFor(self.param_manager), 'Fit Parameters')
        self.addTab(self.tabWidgetFor(self.geolocation), 'Station')
        self.addTab(self.tabWidgetFor(self.star_detection), 'Star Detection')
        self.addTab(self.tabWidgetFor(self.mask), 'Mask')
        self.addTab(self.tabWidgetFor(self.settings), 'Settings')
        self.addTab(self.help, 'ⓘ Help')

        self.setCurrentIndex(self.index)  # redundant
        self.setTabPosition(QtWidgets.QTabWidget.East)

        self.tabBarClicked.connect(self.onTabBarClicked)

    def keyPressEvent(self, event):
        """ Pressing escape when you're focused on any widget on the right focuses
            on the main widget
        """

        if event.key() == QtCore.Qt.Key_Escape:
            self.gui.view_widget.setFocus()


    def maximizedWidthChars(self):
        """ Maximized panel width (in characters). The Help tab gets extra width for readability;
            all other tabs use the normal width. """
        if 0 <= self.index < self.count() and self.widget(self.index) is self.help:
            return self.TAB_WIDTH_CHARS*self.HELP_WIDTH_MULT
        return self.TAB_WIDTH_CHARS

    def applyTabWidth(self):
        """ Resize the panel to match the current maximized/minimized state and selected tab. """
        if self.maximized:
            self.setFixedWidth(self.scaledWidth(self.maximizedWidthChars()))
        else:
            self.setFixedWidth(self.scaledWidth(self.TAB_MINIMIZED_CHARS))

    def onTabBarClicked(self, index):
        old_index = self.index
        if index != self.index:
            self.index = index
            self.maximized = True
            # Wider for Help, normal for everything else
            self.applyTabWidth()
            # Emit signal for tab change
            self.sigTabChanged.emit(old_index, index)
        else:
            self.maximized = not self.maximized
            self.applyTabWidth()

        # Always set the focus back to the image window
        self.gui.view_widget.setFocus()


    def tabWidgetFor(self, panel):
        """ Return the widget actually inserted in the tab bar for the given panel
            (its scroll wrapper if it has one, the panel itself otherwise).
        """
        return self._tab_scroll_wrappers.get(panel, panel)

    def tabIndexOf(self, panel):
        """ Return the tab index of the given panel, looking through scroll wrappers. """
        return self.indexOf(self.tabWidgetFor(panel))

    def onSkyFit(self):

        # Remove ManualReduction-specific tabs
        self.removeTabText('Debruijn')

        # Add Skyfit-specific tabs
        self.insertTab(1, self.tabWidgetFor(self.param_manager), "Fit Parameters")
        self.insertTab(2, self.tabWidgetFor(self.geolocation), "Station")
        self.settings.onSkyFit()

        self.setCurrentIndex(self.index)

    def onManualReduction(self):

        # Remove Skyfit-specific tabs
        self.removeTabText("Fit Parameters")
        self.removeTabText("Station")
        self.settings.onManualReduction()

        # Add ManualReduction-specific tabs
        if self.gui.img.img_handle.input_type == 'dfn':
            self.insertTab(1, self.tabWidgetFor(self.debruijn), 'Debruijn')

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


class HelpWidget(QtWidgets.QWidget, ScaledSizeHelper):
    """ Read-only Help tab with mode- and feature-aware documentation.

    Uses progressive disclosure: a Home page with an intro and a triage list of links, each
    opening a detailed topic page. Content is built in RMS.Routines.SkyFitHelp from the current
    GUI state (mode + enabled features). Call updateHelp() to rebuild after a mode/feature change.
    """

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)

        self.gui = gui

        # Stack of visited topic ids (for the Back button)
        self._history = []

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setContentsMargins(*self.scaledMargins(0.5, 0.25))
        layout.setSpacing(self.scaledSpacing(0.25))
        self.setLayout(layout)

        # Navigation buttons
        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(self.scaledSpacing(0.25))
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.setToolTip("Back to the help topic list")
        self.home_button.clicked.connect(self.showHome)
        self.back_button = QtWidgets.QPushButton("Back")
        self.back_button.clicked.connect(self.goBack)
        nav.addWidget(self.home_button)
        nav.addWidget(self.back_button)
        layout.addLayout(nav)

        # Search box: filters the home topic list as you type
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search help...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._onSearchChanged)
        layout.addWidget(self.search_box)

        # Read-only rich-text view. Internal "topic:" links and external http(s) links are handled
        # manually in onAnchorClicked, so disable Qt's own link following.
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self.onAnchorClicked)

        # Apply the help stylesheet to the document (persists across setHtml calls)
        self.browser.document().setDefaultStyleSheet(HELP_STYLE)

        layout.addWidget(self.browser)

        self.showHome()


    def showHome(self):
        """ Show the intro + triage page for the current mode/features (unfiltered). """
        self._history = []
        # Reset any active search without re-triggering a render
        if self.search_box.text():
            self.search_box.blockSignals(True)
            self.search_box.clear()
            self.search_box.blockSignals(False)
        self._renderHome(None)
        self._updateBackButton()


    def _onSearchChanged(self, text):
        """ Re-render the home page filtered by the search query. """
        self._history = []
        self._renderHome(text.strip() or None)
        self._updateBackButton()


    def _renderHome(self, query):
        """ Build and display the home page, guarded so a content error can't blank the tab. """
        try:
            html = buildHelpHome(self.gui, query=query)
        except Exception as e:
            html = "<h2>Help</h2><p>Could not render the help page: {:s}</p>".format(str(e))
        self.browser.setHtml(html)
        self.browser.verticalScrollBar().setValue(0)


    def updateHelp(self):
        """ Rebuild the help content from the current GUI state (mode/feature change). """
        self.showHome()


    def showTopic(self, topic_id):
        """ Show one detailed topic page. """
        html = buildHelpTopic(self.gui, topic_id)
        if html is None:
            return
        self._history.append(topic_id)
        self.browser.setHtml(html)
        self.browser.verticalScrollBar().setValue(0)
        self._updateBackButton()


    def goBack(self):
        """ Step back to the previous topic, or Home. """
        if self._history:
            self._history.pop()
        if self._history:
            self.browser.setHtml(buildHelpTopic(self.gui, self._history[-1]))
            self.browser.verticalScrollBar().setValue(0)
        else:
            self.showHome()
        self._updateBackButton()


    def _updateBackButton(self):
        self.back_button.setEnabled(len(self._history) > 0)


    def onAnchorClicked(self, url):
        """ Route internal topic links and open external links in the system browser. """
        scheme = url.scheme()
        if scheme in ("http", "https"):
            QtGui.QDesktopServices.openUrl(url)
            return

        topic_id = url.toString()
        if topic_id.startswith("topic:"):
            topic_id = topic_id[len("topic:"):]
        self.showTopic(topic_id)


class DebruijnSequenceManager(QtWidgets.QWidget, ScaledSizeHelper):
    # this whole thing could use some huge lower level changes
    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui
        self.sequence = generateDeBruijnSequence(2, 9)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # table
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setFixedWidth(self.scaledWidth(26))  # ~205px at 8px/char
        self.table.setColumnWidth(0, self.scaledWidth(6))   # break
        self.table.setColumnWidth(1, self.scaledWidth(10))  # time
        self.table.setColumnWidth(2, self.scaledWidth(5))   # value
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
            table.setColumnWidth(0, self.scaledWidth(8))
            table.setColumnWidth(3, self.scaledWidth(22))
            table.verticalHeader().hide()
            table.currentCellChanged.connect(lambda: msg.buttons()[0].setDisabled(False))
            table.setFixedWidth(self.scaledWidth(56))   # ~450px at 8px/char
            table.setFixedHeight(self.scaledHeight(18))  # ~300px at 16px/line
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


class GeolocationWidget(QtWidgets.QWidget, ScaledSizeHelper):

    sigLocationChanged = QtCore.pyqtSignal()
    sigReloadGeoPoints = QtCore.pyqtSignal()
    sigFitPressed = QtCore.pyqtSignal()

    # Width in characters for coordinate input boxes
    COORD_INPUT_CHARS = 15

    def __init__(self, gui):
        """ QWidget contains station information. """

        QtWidgets.QWidget.__init__(self)
        self.gui = gui

        full_layout = QtWidgets.QVBoxLayout()
        full_layout.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(full_layout)

        # Tab help button (top-right)
        self.addCornerHelpButton('station', "Help: station & geo points")

        # Station geo position input boxes
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        group = QtWidgets.QGroupBox('Station coordinates')
        group.setLayout(form)
        full_layout.addWidget(group)


        hbox = QtWidgets.QHBoxLayout()
        self.lat = DoubleSpinBox()
        self.lat.setMinimum(-90)
        self.lat.setMaximum(90)
        self.lat.setDecimals(8)
        self.lat.setSingleStep(0.00001)
        self.lat.setFixedWidth(self.scaledWidth(self.COORD_INPUT_CHARS))
        self.lat.valueModified.connect(self.onLatChanged)
        hbox.addWidget(self.lat)
        hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Lat'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.lon = DoubleSpinBox()
        self.lon.setMinimum(-180)
        self.lon.setMaximum(180)
        self.lon.setDecimals(8)
        self.lon.setSingleStep(0.00001)
        self.lon.setFixedWidth(self.scaledWidth(self.COORD_INPUT_CHARS))
        self.lon.valueModified.connect(self.onLonChanged)
        hbox.addWidget(self.lon)
        hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Lon'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.elev = DoubleSpinBox()
        self.elev.setMinimum(-1000)
        self.elev.setMaximum(1000000)
        self.elev.setDecimals(3)
        self.elev.setSingleStep(1)
        self.elev.setFixedWidth(self.scaledWidth(self.COORD_INPUT_CHARS))
        self.elev.valueModified.connect(self.onElevChanged)
        hbox.addWidget(self.elev)
        hbox.addWidget(QtWidgets.QLabel('m', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Elev'), hbox)

        form.addRow(QtWidgets.QLabel("Press Enter to accept value"))



        # Add buttons to move station coordinates
        box = QtWidgets.QVBoxLayout()
        grid = QtWidgets.QGridLayout()
        box.addLayout(grid)

        group = QtWidgets.QGroupBox('Move station coordinates')
        group.setLayout(box)
        full_layout.addWidget(group)


        # Up button
        self.up_button = QtWidgets.QPushButton("Up")
        self.up_button.clicked.connect(self.onUpPressed)
        grid.addWidget(self.up_button, 0, 1)

        # Forward button
        self.fwd_button = QtWidgets.QPushButton("Forward")
        self.fwd_button.clicked.connect(self.onForwardPressed)
        grid.addWidget(self.fwd_button, 1, 1)

        # Left button
        self.left_button = QtWidgets.QPushButton("Left")
        self.left_button.clicked.connect(self.onLeftPressed)
        grid.addWidget(self.left_button, 2, 0)


        # Right button
        self.right_button = QtWidgets.QPushButton("Right")
        self.right_button.clicked.connect(self.onRightPressed)
        grid.addWidget(self.right_button, 2, 2)

        # Reverse button
        self.reverse_button = QtWidgets.QPushButton("Reverse")
        self.reverse_button.clicked.connect(self.onReversePressed)
        grid.addWidget(self.reverse_button, 3, 1)

        # Down button
        self.down_button = QtWidgets.QPushButton("Down")
        self.down_button.clicked.connect(self.onDownPressed)
        grid.addWidget(self.down_button, 4, 1)


        # movement distance box, use a default of 2 cm
        self.distance = 0.02
        hbox = QtWidgets.QHBoxLayout()
        self.dist_box = DoubleSpinBox()
        self.dist_box.setMinimum(0)
        self.dist_box.setMaximum(1000)
        self.dist_box.setDecimals(3)
        self.dist_box.setSingleStep(1)
        self.dist_box.setFixedWidth(self.scaledWidth(12))  # ~90px at 8px/char
        self.dist_box.setValue(self.distance)
        self.dist_box.valueModified.connect(self.onDistanceChanged)
        hbox.addWidget(QtWidgets.QLabel('Distance'))
        hbox.addWidget(self.dist_box)
        hbox.addWidget(QtWidgets.QLabel('m', alignment=QtCore.Qt.AlignLeft))
        box.addLayout(hbox)
        box.addWidget(QtWidgets.QLabel("Press Enter to accept value"))

        # Checkbox to auto refit astrometry when the station is moved
        self.auto_fit_checkbox = QtWidgets.QCheckBox("Auto refit astrometry")
        self.auto_fit_checkbox.released.connect(self.onAutoFitToggled)
        box.addWidget(self.auto_fit_checkbox)
        if self.gui.geo_points_obj is None:
            self.auto_fit_checkbox.hide()


        # Label to show residuals
        self.residuals_label = QtWidgets.QLabel("Residuals:\n")
        # Wrap instead of clipping - font metrics differ across OSes
        self.residuals_label.setWordWrap(True)
        box.addWidget(self.residuals_label)
        if self.gui.geo_points_obj is None:
            self.residuals_label.hide()


        # Reload geo points
        self.reload_geo_points_button = QtWidgets.QPushButton("Reload geo points")
        self.reload_geo_points_button.clicked.connect(self.sigReloadGeoPoints.emit)
        #box.addWidget(self.reload_geo_points_button)
        full_layout.addWidget(self.reload_geo_points_button)
        if self.gui.geo_points_obj is None:
            self.reload_geo_points_button.hide()

        # Fit astrometry with changed station
        self.fit_astrometry_button = QtWidgets.QPushButton("Fit astrometry")
        self.fit_astrometry_button.clicked.connect(self.sigFitPressed.emit)
        full_layout.addWidget(self.fit_astrometry_button)
        if self.gui.geo_points_obj is None:
            self.fit_astrometry_button.hide()


        self.updatePlatepar()



    def onAutoFitToggled(self):
        """ Refit astrometry once the station is moved. """
        
        self.gui.station_moved_auto_refit = self.auto_fit_checkbox.isChecked()
        

    def autoRefit(self):
        """ Run auto refitting, if enabled. """

        if self.gui.station_moved_auto_refit:
            self.sigFitPressed.emit()


    def updateGeoCoordinatesFromAzimChange(self, azim):
        """ Given an azimuth, update the geo coordinates in the platepar by moving long the azimuth by the
            distance given in the GUI. 
        """

        # Compute geo coordinates of the points along the azimuth for the given range, assume 0 elevation
        lat2, lon2, elev2 = AER2LatLonAlt(azim, 0.0, self.distance, self.gui.platepar.lat, \
            self.gui.platepar.lon, self.gui.platepar.elev)

        self.gui.platepar.lat, self.gui.platepar.lon, self.gui.platepar.elev = lat2, lon2, elev2


    def onLatChanged(self):
        self.gui.platepar.lat = self.lat.value()
        self.sigLocationChanged.emit()
        self.autoRefit()

    def onLonChanged(self):
        self.gui.platepar.lon = self.lon.value()
        self.sigLocationChanged.emit()
        self.autoRefit()

    def onElevChanged(self):
        self.gui.platepar.elev = self.elev.value()
        self.sigLocationChanged.emit()
        self.autoRefit()



    def onUpPressed(self):

        # Read the distance box
        self.onDistanceChanged()

        # Compute the new elevation
        self.gui.platepar.elev += self.distance

        self.sigLocationChanged.emit()
        self.updatePlatepar()

    def onForwardPressed(self):

        # Read the distance box
        self.onDistanceChanged()

        # Move observed along azimuth - forward
        self.updateGeoCoordinatesFromAzimChange(self.gui.platepar.az_centre)

        self.sigLocationChanged.emit()
        self.updatePlatepar()

    def onLeftPressed(self):

        # Read the distance box
        self.onDistanceChanged()

        # Move observed along azimuth - left
        self.updateGeoCoordinatesFromAzimChange((self.gui.platepar.az_centre - 90)%360)

        self.sigLocationChanged.emit()
        self.updatePlatepar()

    def onRightPressed(self):
        
        # Read the distance box
        self.onDistanceChanged()

        # Move observed along azimuth - right
        self.updateGeoCoordinatesFromAzimChange((self.gui.platepar.az_centre + 90)%360)

        self.sigLocationChanged.emit()
        self.updatePlatepar()

    def onReversePressed(self):
        
        # Read the distance box
        self.onDistanceChanged()

        # Move observed along azimuth - reverse
        self.updateGeoCoordinatesFromAzimChange((self.gui.platepar.az_centre - 180)%360)

        self.sigLocationChanged.emit()
        self.updatePlatepar()

    def onDownPressed(self):

        # Read the distance box
        self.onDistanceChanged()

        # Compute the new elevation
        self.gui.platepar.elev -= self.distance

        self.sigLocationChanged.emit()
        self.updatePlatepar()

    def onDistanceChanged(self):
        self.distance = self.dist_box.value()
        pass



    def updatePlatepar(self):
        """
        Updates QDoubleSpinBox values to the values of the platepar.
        Call this whenever the platepar values are changed
        """
        self.lat.setValue(self.gui.platepar.lat)
        self.lon.setValue(self.gui.platepar.lon)
        self.elev.setValue(self.gui.platepar.elev)

        self.autoRefit()



class PlateparParameterManager(QtWidgets.QWidget, ScaledSizeHelper):
    """
    QWidget that contains various QDoubleSpinBox's that can be changed to
    manage platepar parameters
    """
    sigAzAltChanged = QtCore.pyqtSignal()
    sigRotChanged = QtCore.pyqtSignal()
    sigScaleChanged = QtCore.pyqtSignal()
    sigFitParametersChanged = QtCore.pyqtSignal()
    sigLocationChanged = QtCore.pyqtSignal()
    sigExtinctionChanged = QtCore.pyqtSignal()
    sigVignettingChanged = QtCore.pyqtSignal()

    sigFitPressed = QtCore.pyqtSignal()
    sigAutoFitPressed = QtCore.pyqtSignal()
    sigFindPairsPressed = QtCore.pyqtSignal()
    sigComputeResidualsPressed = QtCore.pyqtSignal()
    sigValidateFitPressed = QtCore.pyqtSignal()
    sigShowNightPairsToggled = QtCore.pyqtSignal(bool)
    sigRefitNightPressed = QtCore.pyqtSignal()
    sigQuickAlignPressed = QtCore.pyqtSignal()
    sigFindBestFramePressed = QtCore.pyqtSignal()
    sigAstrometryPressed = QtCore.pyqtSignal()
    sigPhotometryPressed = QtCore.pyqtSignal()
    sigResetDistortionPressed = QtCore.pyqtSignal()

    sigRefractionToggled = QtCore.pyqtSignal()
    sigEqAspectToggled = QtCore.pyqtSignal()
    sigAsymmetryCorrToggled = QtCore.pyqtSignal()
    sigForceDistortionToggled = QtCore.pyqtSignal()
    sigOnVignettingFixedToggled = QtCore.pyqtSignal()
    sigFitOnlyPointingToggled = QtCore.pyqtSignal()
    sigRestoreDefaultsPressed = QtCore.pyqtSignal()

    # Default settings for SkyFit2 Fit Parameters tab
    DEFAULT_FIXED_SCALE = False
    DEFAULT_REFRACTION = True
    DEFAULT_EQUAL_ASPECT = True
    DEFAULT_ASYMMETRY_CORR = True
    DEFAULT_FORCE_DISTORTION_CENTRE = False
    DEFAULT_DISTORTION_TYPE = "radial7-odd"
    DEFAULT_EXTINCTION_SCALE = 0.6
    DEFAULT_VIGNETTING_FIXED = True

    # Width in characters for parameter input boxes
    PARAM_INPUT_CHARS = 13

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui

        # Stash for coefficients that get hidden when reducing coefficient count
        # This allows restoring them when toggling flags back
        self._coeff_stash = {
            'x_fwd': {}, 'x_rev': {}, 'y_fwd': {}, 'y_rev': {}
        }

        full_layout = QtWidgets.QVBoxLayout()
        full_layout.setContentsMargins(*self.scaledMargins(0.5, 0.25))
        self.setLayout(full_layout)

        # Tab help button (top-right)
        self.addCornerHelpButton('astrometry', "Help: fitting the astrometry")

        # buttons
        box = QtWidgets.QVBoxLayout()
        box.setContentsMargins(*self.scaledMargins(0.5, 0.25))
        box.setSpacing(self.scaledSpacing(0.25))

        # Best Frame button row
        best_frame_hbox = QtWidgets.QHBoxLayout()
        best_frame_hbox.setSpacing(self.scaledSpacing(0.25))
        self.find_best_frame_button = QtWidgets.QPushButton("Find Best Frame")
        self.find_best_frame_button.setToolTip(
            "Find the best frame for calibration: star distribution and quality,\n"
            "plus sky condition (darkest, most uniform background and sharpest stars,\n"
            "ranked against the rest of the night)")
        self.find_best_frame_button.clicked.connect(self.sigFindBestFramePressed.emit)
        best_frame_hbox.addWidget(self.find_best_frame_button)
        box.addLayout(best_frame_hbox)

        # Fit buttons in a horizontal layout
        fit_hbox = QtWidgets.QHBoxLayout()
        fit_hbox.setSpacing(self.scaledSpacing(0.25))
        self.fit_astrometry_button = QtWidgets.QPushButton("Fit")
        self.fit_astrometry_button.setToolTip(
            "Fit the platepar to the current star pairs (needs pairs)")
        self.fit_astrometry_button.clicked.connect(self.sigFitPressed.emit)
        fit_hbox.addWidget(self.fit_astrometry_button)

        self.auto_fit_button = QtWidgets.QPushButton("Auto Fit")
        self.auto_fit_button.setToolTip("Automatic plate solving using astrometry.net (Ctrl+X)")
        self.auto_fit_button.clicked.connect(self.sigAutoFitPressed.emit)
        fit_hbox.addWidget(self.auto_fit_button)
        box.addLayout(fit_hbox)

        # Individual fit steps: pair finding and residuals, runnable independently of the fit
        steps_hbox = QtWidgets.QHBoxLayout()
        steps_hbox.setSpacing(self.scaledSpacing(0.25))
        self.find_pairs_button = QtWidgets.QPushButton("Find Pairs")
        self.find_pairs_button.setToolTip(
            "Match detected stars to catalog stars using the current platepar.\n"
            "Replaces the current pairs; does not fit anything (needs a decent platepar).")
        self.find_pairs_button.clicked.connect(self.sigFindPairsPressed.emit)
        steps_hbox.addWidget(self.find_pairs_button)

        self.compute_residuals_button = QtWidgets.QPushButton("Residuals")
        self.compute_residuals_button.setToolTip(
            "Compute residuals of the current pairs against the current platepar.\n"
            "Does not fit anything (needs pairs).")
        self.compute_residuals_button.clicked.connect(self.sigComputeResidualsPressed.emit)
        steps_hbox.addWidget(self.compute_residuals_button)
        box.addLayout(steps_hbox)

        # Quick Align button row
        quick_align_hbox = QtWidgets.QHBoxLayout()
        quick_align_hbox.setSpacing(self.scaledSpacing(0.25))
        self.quick_align_button = QtWidgets.QPushButton("Auto Pointing")
        self.quick_align_button.setToolTip(
            "Automatically re-estimate pointing from detected stars (existing distortion kept). "
            "Falls back to astrometry.net. Does not use your picked stars.")
        self.quick_align_button.clicked.connect(self.sigQuickAlignPressed.emit)
        quick_align_hbox.addWidget(self.quick_align_button)
        box.addLayout(quick_align_hbox)

        # Frame budget for the cross-frame validation/refit subset. Frames are picked
        # greedily for spatial coverage, so a modest budget covers the image; validating
        # thousands of frames takes minutes for no accuracy gain
        validate_frames_hbox = QtWidgets.QHBoxLayout()
        validate_frames_label = QtWidgets.QLabel("Validation max frames")
        self.validate_max_frames_spin = QtWidgets.QSpinBox()
        self.validate_max_frames_spin.setRange(10, 5000)
        self.validate_max_frames_spin.setValue(100)
        self.validate_max_frames_spin.setToolTip(
            "Frame budget for the coverage-selected subset used by Validate Across Frames\n"
            "and Refit W/ Night. Frames are picked greedily so their union of detected stars\n"
            "covers the image; the corner cells are always topped up, even past this budget.\n"
            "Budget left over after coverage saturates is spent on frames spread evenly\n"
            "across the night, so temporal variation is sampled too.")
        validate_frames_label.setToolTip(self.validate_max_frames_spin.toolTip())
        validate_frames_hbox.addWidget(validate_frames_label)
        validate_frames_hbox.addStretch()
        validate_frames_hbox.addWidget(self.validate_max_frames_spin)
        box.addLayout(validate_frames_hbox)

        # Cross-frame validation buttons (stacked - side by side they overflow the panel)
        self.validate_fit_button = QtWidgets.QPushButton("Validate Across Frames")
        self.validate_fit_button.setToolTip(
            "Check how well the fit generalizes to other frames of the night, especially the\n"
            "image corners: matches detected stars (CALSTARS) to the catalog on a coverage-\n"
            "selected frame subset, refits only the pointing per frame so mount drift is\n"
            "separated from distortion error, and reports residuals by radius.")
        self.validate_fit_button.clicked.connect(self.sigValidateFitPressed.emit)
        box.addWidget(self.validate_fit_button)

        self.refit_night_button = QtWidgets.QPushButton("Refit W/ Night")
        self.refit_night_button.setToolTip(
            "Complement the astrometric fit with the validated cross-frame star pairs\n"
            "(spatially balanced, corner pairs kept in full, per-frame pointing drift\n"
            "compensated). The refit is validated against the night and only kept if it\n"
            "improves on the current platepar. Photometry is not affected - it must come\n"
            "from a single frame. Run Validate Across Frames first.")
        self.refit_night_button.setEnabled(False)
        self.refit_night_button.clicked.connect(self.sigRefitNightPressed.emit)
        box.addWidget(self.refit_night_button)

        self.show_night_pairs_checkbox = QtWidgets.QCheckBox('Show Night Pairs')
        self.show_night_pairs_checkbox.setToolTip(
            "Overlay the validated cross-frame star pairs on the current frame: circles mark\n"
            "detections from other frames of the night (the camera is fixed, so their pixel\n"
            "positions apply directly), needles point to their catalog positions projected at\n"
            "each pair's own frame time (exaggerated). This is the pair set Refit W/ Night\n"
            "would fit - useful to eyeball corner coverage before refitting.")
        self.show_night_pairs_checkbox.setEnabled(False)
        self.show_night_pairs_checkbox.toggled.connect(self.sigShowNightPairsToggled.emit)
        box.addWidget(self.show_night_pairs_checkbox)


        box.addWidget(QtWidgets.QLabel("Residuals:"))

        # RMSD display label with color coding. Shows the simple px RMSD; the forward/reverse
        # consistency and held-out overfitting checks run internally on every fit and turn this
        # label red when either trips (the detailed numbers are printed to the console).
        self.rmsd_label = QtWidgets.QLabel("--")
        self.rmsd_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        # Wrap instead of clipping: font metrics differ across OSes, and the suffixes
        # ("(on N of M)", health-check flags) can outgrow the fixed-width panel
        self.rmsd_label.setWordWrap(True)
        box.addWidget(self.rmsd_label)

        # Max round-trip (forward vs reverse mapping) disagreement across the whole image,
        # updated with the error overlay. Complements the RMSD, which only covers matched stars.
        self.roundtrip_label = QtWidgets.QLabel("")
        self.roundtrip_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.roundtrip_label.setWordWrap(True)
        box.addWidget(self.roundtrip_label)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setSpacing(self.scaledSpacing(0.25))  # Reduce spacing between buttons
        self.astrometry_button = QtWidgets.QPushButton('Astrometry')
        self.astrometry_button.clicked.connect(self.sigAstrometryPressed.emit)
        hbox.addWidget(self.astrometry_button)

        self.photometry_button = QtWidgets.QPushButton('Photometry')
        self.photometry_button.clicked.connect(self.sigPhotometryPressed.emit)
        hbox.addWidget(self.photometry_button)

        # Small circular "i" button: opens the Help page on reading the residual plots
        self.residuals_help_button = self.makeHelpButton('residuals', "How to read the residual plots")
        hbox.addWidget(self.residuals_help_button)
        box.addLayout(hbox)

        self.updatePairedStars()
        group = QtWidgets.QGroupBox("Calibration")
        # Dynamic stylesheet with scaled padding
        pad_top = self.scaledHeight(0.75)
        pad_side = self.scaledWidth(0.25)
        group.setStyleSheet(f"QGroupBox {{ padding-top: {pad_top}px; padding-left: {pad_side}px; padding-right: {pad_side}px; }}")
        group.setLayout(box)
        full_layout.addWidget(group)

        hline = QHSeparationLine()
        full_layout.addWidget(hline)
        full_layout.addWidget(QtWidgets.QLabel("Astrometry parameters"))

        # Restore defaults button at top of section
        self.restore_defaults_button = QtWidgets.QPushButton("Restore Defaults")
        self.restore_defaults_button.clicked.connect(self.onRestoreDefaults)
        full_layout.addWidget(self.restore_defaults_button)

        # check boxes
        self.fit_only_pointing = QtWidgets.QCheckBox('Only fit pointing')
        self.fit_only_pointing.released.connect(self.onFitOnlyPointingToggled)
        full_layout.addWidget(self.fit_only_pointing)

        self.fixed_scale = QtWidgets.QCheckBox('Fixed scale')
        self.fixed_scale.released.connect(self.onFixScaleToggled)
        full_layout.addWidget(self.fixed_scale)

        self.refraction = QtWidgets.QCheckBox('Refraction')
        self.refraction.released.connect(self.onRefractionToggled)
        full_layout.addWidget(self.refraction)

        self.eqAspect = QtWidgets.QCheckBox('Equal Aspect')
        self.eqAspect.released.connect(self.onEqualAspectToggled)
        full_layout.addWidget(self.eqAspect)
        if not self.gui.platepar.distortion_type.startswith('radial'):
            self.eqAspect.hide()

        self.asymmetryCorr = QtWidgets.QCheckBox('Asymmetry Correction')
        self.asymmetryCorr.released.connect(self.onAsymmetryCorrToggled)
        full_layout.addWidget(self.asymmetryCorr)
        if not self.gui.platepar.distortion_type.startswith('radial'):
            self.asymmetryCorr.hide()

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
        self.az_centre.setMinimum(0.0)
        self.az_centre.setMaximum(360)
        self.az_centre.setDecimals(8)
        self.az_centre.setSingleStep(1)
        self.az_centre.setFixedWidth(self.scaledWidth(self.PARAM_INPUT_CHARS))
        self.az_centre.valueModified.connect(self.onAzChanged)
        hbox.addWidget(self.az_centre)
        hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Azim'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.alt_centre = DoubleSpinBox()
        self.alt_centre.setMinimum(-90)
        self.alt_centre.setMaximum(90)
        self.alt_centre.setDecimals(8)
        self.alt_centre.setSingleStep(1)
        self.alt_centre.setFixedWidth(self.scaledWidth(self.PARAM_INPUT_CHARS))
        self.alt_centre.valueModified.connect(self.onAltChanged)
        hbox.addWidget(self.alt_centre)
        hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Alt'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.rotation_from_horiz = DoubleSpinBox()
        self.rotation_from_horiz.setMinimum(-360)
        self.rotation_from_horiz.setMaximum(360)
        self.rotation_from_horiz.setDecimals(8)
        self.rotation_from_horiz.setSingleStep(1)
        self.rotation_from_horiz.setFixedWidth(self.scaledWidth(self.PARAM_INPUT_CHARS))
        self.rotation_from_horiz.valueModified.connect(self.onRotChanged)
        hbox.addWidget(self.rotation_from_horiz)
        hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Horiz rot'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        self.F_scale = DoubleSpinBox()
        self.F_scale.setMinimum(0)
        self.F_scale.setMaximum(50)
        self.F_scale.setDecimals(8)
        self.F_scale.setSingleStep(0.1)
        self.F_scale.setFixedWidth(self.scaledWidth(self.PARAM_INPUT_CHARS))
        self.F_scale.valueModified.connect(self.onScaleChanged)
        hbox.addWidget(self.F_scale)
        hbox.addWidget(QtWidgets.QLabel('\'/px', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Scale'), hbox)

        form.addRow(QtWidgets.QLabel("Press Enter to accept value"))

        # hbox = QtWidgets.QHBoxLayout()
        # self.lat = DoubleSpinBox()
        # self.lat.setMinimum(-360)
        # self.lat.setMaximum(360)
        # self.lat.setDecimals(8)
        # self.lat.setSingleStep(1)
        # self.lat.setFixedWidth(100)
        # self.lat.valueModified.connect(self.onLatChanged)
        # hbox.addWidget(self.lat)
        # hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        # form.addRow(QtWidgets.QLabel('Lat'), hbox)

        # hbox = QtWidgets.QHBoxLayout()
        # self.lon = DoubleSpinBox()
        # self.lon.setMinimum(-360)
        # self.lon.setMaximum(360)
        # self.lon.setDecimals(8)
        # self.lon.setSingleStep(1)
        # self.lon.setFixedWidth(100)
        # self.lon.valueModified.connect(self.onLonChanged)
        # hbox.addWidget(self.lon)
        # hbox.addWidget(QtWidgets.QLabel(u"\N{DEGREE SIGN}", alignment=QtCore.Qt.AlignLeft))
        # form.addRow(QtWidgets.QLabel('Lon'), hbox)

        # hbox = QtWidgets.QHBoxLayout()
        # self.elev = DoubleSpinBox()
        # self.elev.setMinimum(0)
        # self.elev.setMaximum(1000000)
        # self.elev.setDecimals(8)
        # self.elev.setSingleStep(100)
        # self.elev.setFixedWidth(100)
        # self.elev.valueModified.connect(self.onElevChanged)
        # hbox.addWidget(self.elev)
        # hbox.addWidget(QtWidgets.QLabel('m', alignment=QtCore.Qt.AlignLeft))
        # form.addRow(QtWidgets.QLabel('Elev'), hbox)

        self.distortion_type = QtWidgets.QComboBox(self)
        self.distortion_type.addItems(self.gui.platepar.distortion_type_list)
        self.distortion_type.currentIndexChanged.connect(self.onIndexChanged)
        form.addRow(QtWidgets.QLabel('Distortion'), self.distortion_type)


        # Distortion buttons row - use widget container to prevent clipping
        distortion_widget = QtWidgets.QWidget()
        distortion_widget.setMinimumHeight(28)
        distortion_buttons = QtWidgets.QHBoxLayout(distortion_widget)
        distortion_buttons.setSpacing(4)
        distortion_buttons.setContentsMargins(0, 0, 0, 0)

        self.reset_distortion_button = QtWidgets.QPushButton("Reset")
        self.reset_distortion_button.clicked.connect(self.sigResetDistortionPressed.emit)
        distortion_buttons.addWidget(self.reset_distortion_button)

        self.edit_distortion_button = QtWidgets.QPushButton("Coefficients...")
        self.edit_distortion_button.clicked.connect(self.openDistortionDialog)
        distortion_buttons.addWidget(self.edit_distortion_button)

        form.addRow(distortion_widget)

        # Create the distortion dialog (hidden initially)
        self.distortion_dialog = DistortionDialog(self.gui, self.gui.platepar)
        self.distortion_dialog.valueModified.connect(self.onFitParametersChanged)

        # Keep reference for compatibility with existing code that accesses fit_parameters
        self.fit_parameters = self.distortion_dialog.fit_parameters

        # Add horizontal line
        hline = QHSeparationLine()
        form.addRow(hline)
        form.addRow(QtWidgets.QLabel("Photometry parameters"))

        hbox = QtWidgets.QHBoxLayout()
        self.extinction_scale = DoubleSpinBox()
        self.extinction_scale.setMinimum(0)
        self.extinction_scale.setMaximum(100)
        self.extinction_scale.setDecimals(4)
        self.extinction_scale.setSingleStep(0.1)
        self.extinction_scale.setFixedWidth(self.scaledWidth(self.PARAM_INPUT_CHARS - 4))
        self.extinction_scale.valueModified.connect(self.onExtinctionChanged)
        hbox.addWidget(self.extinction_scale)
        hbox.addWidget(QtWidgets.QLabel('', alignment=QtCore.Qt.AlignLeft))
        form.addRow(QtWidgets.QLabel('Extinction'), hbox)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setAlignment(QtCore.Qt.AlignVCenter)
        self.vignetting_coeff = DoubleSpinBox()
        self.vignetting_coeff.setMinimum(0)
        self.vignetting_coeff.setMaximum(0.1)
        self.vignetting_coeff.setDecimals(5)
        self.vignetting_coeff.setSingleStep(0.0001)
        self.vignetting_coeff.setFixedWidth(self.scaledWidth(self.PARAM_INPUT_CHARS - 3))
        self.vignetting_coeff.valueModified.connect(self.onVignettingChanged)
        hbox.addWidget(self.vignetting_coeff)
        hbox.addWidget(QtWidgets.QLabel('r/px'))
        vignetting_info = QtWidgets.QToolButton()
        vignetting_info.setText("ⓘ")
        info_font = vignetting_info.font()
        info_font.setPointSize(info_font.pointSize() + 2)
        info_font.setBold(True)
        vignetting_info.setFont(info_font)
        vignetting_info.setStyleSheet("QToolButton { color: #0066cc; border: none; } QToolButton:hover { color: #0044aa; }")
        vignetting_info.setCursor(QtCore.Qt.PointingHandCursor)
        vignetting_info.clicked.connect(self.showVignettingInfo)
        hbox.addWidget(vignetting_info)
        form.addRow(QtWidgets.QLabel("Vignetting"), hbox)

        hbox_fixed = QtWidgets.QHBoxLayout()
        hbox_fixed.setAlignment(QtCore.Qt.AlignVCenter)
        self.vignetting_fixed = QtWidgets.QCheckBox('Fixed vignetting')
        self.vignetting_fixed.setChecked(True)
        self.vignetting_fixed.released.connect(self.onVignettingFixedToggled)
        hbox_fixed.addWidget(self.vignetting_fixed)
        vignetting_fixed_info = QtWidgets.QToolButton()
        vignetting_fixed_info.setText("ⓘ")
        info_font = vignetting_fixed_info.font()
        info_font.setPointSize(info_font.pointSize() + 2)
        info_font.setBold(True)
        vignetting_fixed_info.setFont(info_font)
        vignetting_fixed_info.setStyleSheet("QToolButton { color: #0066cc; border: none; } QToolButton:hover { color: #0044aa; }")
        vignetting_fixed_info.setCursor(QtCore.Qt.PointingHandCursor)
        vignetting_fixed_info.clicked.connect(self.showVignettingFixedInfo)
        hbox_fixed.addWidget(vignetting_fixed_info)
        hbox_fixed.addStretch()
        form.addRow(hbox_fixed)

        # Add stretch to absorb extra vertical space at the bottom
        full_layout.addStretch()

        self.updatePlatepar()
        self.updateRestoreDefaultsButton()


    def onFixScaleToggled(self):
        self.gui.fixed_scale = self.fixed_scale.isChecked()
        self.sigFitParametersChanged.emit()
        self.updatePairedStars(min_fit_stars=self.gui.getMinFitStars())

        if self.gui.fixed_scale:
            self.F_scale.setDisabled(True)
        else:
            self.F_scale.setDisabled(False)

        self.updateRestoreDefaultsButton()

    def onRefractionToggled(self):
        self.gui.platepar.refraction = self.refraction.isChecked()
        self.updateRestoreDefaultsButton()
        self.sigRefractionToggled.emit()

    def _stashCurrentCoeffs(self):
        """Stash current coefficient values for later restoration.

        This saves non-zero coefficients to the stash so they can be restored
        when switching back to a distortion type/flags that need them.
        """
        pp = self.gui.platepar

        # Only works for radial distortion types
        if not pp.distortion_type.startswith("radial"):
            return

        # Extract current coefficients and update stash with non-zero values
        x_coeffs_fwd = pp.extractRadialCoeffs(pp.x_poly_fwd)
        x_coeffs_rev = pp.extractRadialCoeffs(pp.x_poly_rev)
        y_coeffs_fwd = pp.extractRadialCoeffs(pp.y_poly_fwd)
        y_coeffs_rev = pp.extractRadialCoeffs(pp.y_poly_rev)

        if x_coeffs_fwd is not None:
            for key, val in x_coeffs_fwd.items():
                if val != 0.0:
                    self._coeff_stash['x_fwd'][key] = val
            for key, val in x_coeffs_rev.items():
                if val != 0.0:
                    self._coeff_stash['x_rev'][key] = val
            for key, val in y_coeffs_fwd.items():
                if val != 0.0:
                    self._coeff_stash['y_fwd'][key] = val
            for key, val in y_coeffs_rev.items():
                if val != 0.0:
                    self._coeff_stash['y_rev'][key] = val

    def _restoreCoeffsFromStash(self):
        """Restore any zero coefficients from the stash.

        This restores stashed coefficient values where current values are zero.
        """
        pp = self.gui.platepar

        # Only works for radial distortion types
        if not pp.distortion_type.startswith("radial"):
            return

        # Extract current coefficients
        x_coeffs_fwd = pp.extractRadialCoeffs(pp.x_poly_fwd)
        x_coeffs_rev = pp.extractRadialCoeffs(pp.x_poly_rev)
        y_coeffs_fwd = pp.extractRadialCoeffs(pp.y_poly_fwd)
        y_coeffs_rev = pp.extractRadialCoeffs(pp.y_poly_rev)

        if x_coeffs_fwd is not None:
            # Restore stashed values where current value is zero
            for key, val in self._coeff_stash['x_fwd'].items():
                if x_coeffs_fwd.get(key, 0.0) == 0.0:
                    x_coeffs_fwd[key] = val
            for key, val in self._coeff_stash['x_rev'].items():
                if x_coeffs_rev.get(key, 0.0) == 0.0:
                    x_coeffs_rev[key] = val
            for key, val in self._coeff_stash['y_fwd'].items():
                if y_coeffs_fwd.get(key, 0.0) == 0.0:
                    y_coeffs_fwd[key] = val
            for key, val in self._coeff_stash['y_rev'].items():
                if y_coeffs_rev.get(key, 0.0) == 0.0:
                    y_coeffs_rev[key] = val

            # Rebuild coefficient arrays with restored values
            pp.x_poly_fwd = pp.buildRadialCoeffs(x_coeffs_fwd, pp.distortion_type)
            pp.x_poly_rev = pp.buildRadialCoeffs(x_coeffs_rev, pp.distortion_type)
            pp.y_poly_fwd = pp.buildRadialCoeffs(y_coeffs_fwd, pp.distortion_type)
            pp.y_poly_rev = pp.buildRadialCoeffs(y_coeffs_rev, pp.distortion_type)
            pp.x_poly = pp.x_poly_fwd
            pp.y_poly = pp.y_poly_fwd

    def _remapCoeffsWithStash(self, flag_name, new_value):
        """Remap coefficients when toggling a flag, using the stash to restore hidden coefficients.

        This method stashes the current coefficient values before remapping, then restores
        any values that would otherwise be zeros after toggling back.

        Arguments:
            flag_name: [str] The flag being changed
            new_value: [bool] The new value for the flag
        """
        pp = self.gui.platepar

        # Only works for radial distortion types
        if not pp.distortion_type.startswith("radial"):
            setattr(pp, flag_name, new_value)
            return

        # Stash current coefficients before remapping
        self._stashCurrentCoeffs()

        # Do the standard remap
        pp.remapCoeffsForFlagChange(flag_name, new_value)

        # Restore any zeros from stash
        self._restoreCoeffsFromStash()

    def _changeDistortionTypeWithStash(self, new_dist_type):
        """Change distortion type while preserving coefficients via stash.

        This method stashes the current coefficient values before changing the
        distortion type, then restores any values that would otherwise be zeros.

        Arguments:
            new_dist_type: [str] The new distortion type (e.g., 'radial3-odd', 'radial5-odd')
        """
        pp = self.gui.platepar

        # Stash current coefficients before changing type (only for radial types)
        self._stashCurrentCoeffs()

        # Change the distortion type
        pp.setDistortionType(new_dist_type, reset_params=False)

        # Restore any zeros from stash (only for radial types)
        self._restoreCoeffsFromStash()

    def onEqualAspectToggled(self):
        new_value = self.eqAspect.isChecked()
        # Remap coefficients with stash to preserve/restore values when toggling
        self._remapCoeffsWithStash('equal_aspect', new_value)

        # Update GUI to reflect new poly_length and coefficient values
        self.fit_parameters.changeNumberShown(self.gui.platepar.poly_length)
        self.fit_parameters.updateValues()

        # Update restore defaults button state
        self.updateRestoreDefaultsButton()

        # Emit signal to trigger display updates (connected to onFitParametersChanged)
        self.sigEqAspectToggled.emit()

    def onAsymmetryCorrToggled(self):
        new_value = self.asymmetryCorr.isChecked()
        # Remap coefficients with stash to preserve/restore values when toggling
        self._remapCoeffsWithStash('asymmetry_corr', new_value)

        # Update GUI to reflect new poly_length and coefficient values
        self.fit_parameters.changeNumberShown(self.gui.platepar.poly_length)
        self.fit_parameters.updateValues()

        # Update restore defaults button state
        self.updateRestoreDefaultsButton()

        # Emit signal to trigger display updates (connected to onFitParametersChanged)
        self.sigAsymmetryCorrToggled.emit()

    def onForceDistortionToggled(self):
        new_value = self.fdistortion.isChecked()
        # Remap coefficients with stash to preserve/restore values when toggling
        self._remapCoeffsWithStash('force_distortion_centre', new_value)

        # Update GUI to reflect new poly_length and coefficient values
        self.fit_parameters.changeNumberShown(self.gui.platepar.poly_length)
        self.fit_parameters.updateValues()

        # Update restore defaults button state
        self.updateRestoreDefaultsButton()

        # Emit signal to trigger display updates (connected to onFitParametersChanged)
        self.sigForceDistortionToggled.emit()

    # def onLatChanged(self):
    #     self.gui.platepar.lat = self.lat.value()
    #     self.sigLocationChanged.emit()

    # def onLonChanged(self):
    #     self.gui.platepar.lon = self.lon.value()
    #     self.sigLocationChanged.emit()

    # def onElevChanged(self):
    #     self.gui.platepar.elev = self.elev.value()
    #     self.sigLocationChanged.emit()

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
        self.gui.platepar.F_scale = 60/self.F_scale.value()
        self.sigScaleChanged.emit()

    def onExtinctionChanged(self):
        self.gui.platepar.extinction_scale = self.extinction_scale.value()
        self.updateRestoreDefaultsButton()
        self.sigExtinctionChanged.emit()

    def onVignettingChanged(self):
        self.gui.platepar.vignetting_coeff = self.vignetting_coeff.value()
        self.sigVignettingChanged.emit()

    def showVignettingInfo(self):
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("Vignetting Coefficient")
        msg.setText("Vignetting coefficient (radians per pixel)")
        msg.setInformativeText(
            "The displayed value comes from the loaded platepar.\n\n"
            "For new platepars, defaults are calibrated for 4mm f/0.95 lens "
            "and scaled by resolution:\n"
            "• 720p (1280×720): 0.001 r/px\n"
            "• 1080p (1920×1080): ~0.00068 r/px\n\n"
            "For other lenses, consider fitting the coefficient\n"
            "(see info next to Fixed Vignetting), or reach out to\n"
            "the GMN community for known good values for your lens."
        )
        msg.exec_()

    def showVignettingFixedInfo(self):
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("Measuring Vignetting Coefficient")
        msg.setText("Measuring vignetting requires ideal conditions:")
        msg.setInformativeText(
            "• Moonless, perfectly cloudless sky\n"
            "• No haze or atmospheric gradients\n"
            "• Camera pointed at high elevation\n"
            "• Deep into the night (no skyglow)\n"
            "• Well-distributed stars across FOV\n"
            "• Results averaged across several nights\n\n"
            "If these conditions are not met, it is best to leave\n"
            "Fixed Vignetting checked and use the default value."
        )
        msg.exec_()

    def onVignettingFixedToggled(self):
        self.gui.platepar.vignetting_fixed = self.vignetting_fixed.isChecked()
        self.sigOnVignettingFixedToggled.emit()

        # If the vignetting is fixed, allow setting manual values
        self.vignetting_coeff.setDisabled(not self.gui.platepar.vignetting_fixed)

        # Update restore defaults button state
        self.updateRestoreDefaultsButton()

    def updateRoundtripError(self, max_err):
        """Show the maximum forward/reverse round-trip error across the image (px)."""
        self.roundtrip_label.setText("Round-trip max: {:.2f} px".format(max_err))

    def updateRMSD(self, rmsd_img, rmsd_angular, angular_error_label, fwdrev_mismatch=False, clean_info=None,
                   overfit=False):
        """Update the RMSD display with color coding based on pixel RMSD.

        The label shows the plain RMSD (reverse residual in px and forward residual in angular
        units). Two health checks run internally on every fit and, when either trips, override the
        color to red so a good-looking RMSD can't hide a broken fit (the detailed numbers behind
        both checks are printed to the console):
            - fwdrev_mismatch: the forward and reverse distortion mappings disagree, so the catalog
              overlay will be off even though the reverse RMSD looks fine.
            - overfit: the held-out (cross-validated) RMSD is much worse than in-sample, i.e. the
              model is fitting centroid noise rather than the true distortion.

        Thresholds are normalized to 1280x720 resolution:
            - < 0.2 px: Excellent (green)
            - < 0.3 px: Good (light green)
            - < 0.4 px: Acceptable (yellow)
            - < 0.5 px: Marginal (orange)
            - >= 0.5 px: Poor (red)
        """
        text = "{:.2f} px, {:.2f} {:s}".format(rmsd_img, rmsd_angular, angular_error_label)
        if clean_info:
            text += "  ({:s})".format(clean_info)

        # Scale thresholds by resolution (reference: 720p)
        scale = self.gui.platepar.Y_res / 720.0

        if rmsd_img <= 0.2 * scale:
            color = "#228B22"  # Forest green - excellent
        elif rmsd_img <= 0.3 * scale:
            color = "#32CD32"  # Lime green - good
        elif rmsd_img <= 0.4 * scale:
            color = "#DAA520"  # Goldenrod - acceptable
        elif rmsd_img <= 0.5 * scale:
            color = "#FF8C00"  # Dark orange - marginal
        else:
            color = "#DC143C"  # Crimson - poor

        # Internal health checks override the color to red regardless of how good the RMSD looks.
        flags = []
        if fwdrev_mismatch:
            flags.append("MAPPING MISMATCH")
        if overfit:
            flags.append("OVERFIT")
        if flags:
            text += "  " + " / ".join(flags)
            color = "#DC143C"

        self.rmsd_label.setText(text)
        self.rmsd_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: {};".format(color))

    def onFitParametersChanged(self):
        # fit parameter object updates platepar by itself
        self.sigFitParametersChanged.emit()

    def openDistortionDialog(self):
        """Open the distortion parameters dialog."""
        # Update the dialog with current platepar before showing
        self.distortion_dialog.updatePlatepar(self.gui.platepar)
        self.distortion_dialog.show()
        self.distortion_dialog.raise_()
        self.distortion_dialog.activateWindow()

    def onIndexChanged(self):
        text = self.distortion_type.currentText()

        # Use stash-aware method to preserve coefficients when switching distortion types
        self._changeDistortionTypeWithStash(text)

        # Set the number of shown poly parameters in the GUI
        self.fit_parameters.changeNumberShown(self.gui.platepar.poly_length)
        self.fit_parameters.updateValues()

        if self.gui.platepar.distortion_type.startswith('radial'):
            self.eqAspect.show()
            self.asymmetryCorr.show()
            self.fdistortion.show()
        else:
            self.eqAspect.hide()
            self.asymmetryCorr.hide()
            self.fdistortion.hide()

        # Update restore defaults button state
        self.updateRestoreDefaultsButton()

        self.sigFitParametersChanged.emit()

    def updatePlatepar(self):
        """
        Updates QDoubleSpinBox values to the values of the platepar.
        Call this whenever the platepar values are changed
        """
        self.az_centre.setValue(self.gui.platepar.az_centre)
        self.alt_centre.setValue(self.gui.platepar.alt_centre)
        self.rotation_from_horiz.setValue(self.gui.platepar.rotation_from_horiz)
        self.F_scale.setValue(60/self.gui.platepar.F_scale)
        # Update platepar reference in distortion dialog in case a new platepar was loaded
        self.distortion_dialog.updatePlatepar(self.gui.platepar)
        self.distortion_type.setCurrentIndex(
            self.gui.platepar.distortion_type_list.index(self.gui.platepar.distortion_type))
        self.extinction_scale.setValue(self.gui.platepar.extinction_scale)
        
        self.vignetting_coeff.setValue(self.gui.platepar.vignetting_coeff)
        self.vignetting_fixed.setChecked(self.gui.platepar.vignetting_fixed)

        # If the vignetting is fixed, allow setting manual values
        self.vignetting_coeff.setDisabled(not self.gui.platepar.vignetting_fixed)

        self.refraction.setChecked(self.gui.platepar.refraction)
        self.fit_only_pointing.setChecked(self.gui.fit_only_pointing)
        self.eqAspect.setChecked(self.gui.platepar.equal_aspect)
        self.asymmetryCorr.setChecked(self.gui.platepar.asymmetry_corr)
        self.fdistortion.setChecked(self.gui.platepar.force_distortion_centre)

        if self.gui.platepar.distortion_type.startswith('radial'):
            self.eqAspect.show()
            self.asymmetryCorr.show()
            self.fdistortion.show()
        else:
            self.eqAspect.hide()
            self.asymmetryCorr.hide()
            self.fdistortion.hide()

        # Update restore defaults button state
        self.updateRestoreDefaultsButton()

    def isAtDefaults(self):
        """Check if current settings match the defaults."""
        pp = self.gui.platepar
        gui = self.gui
        return (gui.fixed_scale == self.DEFAULT_FIXED_SCALE and
                gui.fit_only_pointing == False and
                pp.refraction == self.DEFAULT_REFRACTION and
                pp.equal_aspect == self.DEFAULT_EQUAL_ASPECT and
                pp.asymmetry_corr == self.DEFAULT_ASYMMETRY_CORR and
                pp.force_distortion_centre == self.DEFAULT_FORCE_DISTORTION_CENTRE and
                pp.distortion_type == self.DEFAULT_DISTORTION_TYPE and
                pp.extinction_scale == self.DEFAULT_EXTINCTION_SCALE and
                pp.vignetting_fixed == self.DEFAULT_VIGNETTING_FIXED)

    def updateRestoreDefaultsButton(self):
        """Update restore defaults button color based on current settings.

        Green when at defaults, amber when not at defaults.
        """
        at_defaults = self.isAtDefaults()
        if at_defaults:
            # Green background when at defaults
            self.restore_defaults_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; }"
                "QPushButton:hover { background-color: #45a049; }"
            )
        else:
            # Amber/orange background when not at defaults
            self.restore_defaults_button.setStyleSheet(
                "QPushButton { background-color: #FF9800; color: white; }"
                "QPushButton:hover { background-color: #F57C00; }"
            )
        self.restore_defaults_button.setEnabled(not at_defaults)

    def onFitOnlyPointingToggled(self):
        self.gui.fit_only_pointing = self.fit_only_pointing.isChecked()
        self.updatePairedStars(min_fit_stars=self.gui.getMinFitStars())
        self.sigFitOnlyPointingToggled.emit()

    def onRestoreDefaults(self):
        """Restore all default settings for the Fit Parameters tab."""
        pp = self.gui.platepar
        gui = self.gui

        # Restore fixed scale
        if gui.fixed_scale != self.DEFAULT_FIXED_SCALE:
            gui.fixed_scale = self.DEFAULT_FIXED_SCALE
            self.fixed_scale.setChecked(self.DEFAULT_FIXED_SCALE)
            self.F_scale.setDisabled(self.DEFAULT_FIXED_SCALE)

        # Restore fit only pointing
        if gui.fit_only_pointing:
            gui.fit_only_pointing = False
            self.fit_only_pointing.setChecked(False)

        # Restore refraction
        if pp.refraction != self.DEFAULT_REFRACTION:
            pp.refraction = self.DEFAULT_REFRACTION
            self.refraction.setChecked(self.DEFAULT_REFRACTION)

        # Restore equal aspect
        if pp.equal_aspect != self.DEFAULT_EQUAL_ASPECT:
            self._remapCoeffsWithStash('equal_aspect', self.DEFAULT_EQUAL_ASPECT)
            self.eqAspect.setChecked(self.DEFAULT_EQUAL_ASPECT)

        # Restore asymmetry correction
        if pp.asymmetry_corr != self.DEFAULT_ASYMMETRY_CORR:
            self._remapCoeffsWithStash('asymmetry_corr', self.DEFAULT_ASYMMETRY_CORR)
            self.asymmetryCorr.setChecked(self.DEFAULT_ASYMMETRY_CORR)

        # Restore force distortion centre
        if pp.force_distortion_centre != self.DEFAULT_FORCE_DISTORTION_CENTRE:
            self._remapCoeffsWithStash('force_distortion_centre', self.DEFAULT_FORCE_DISTORTION_CENTRE)
            self.fdistortion.setChecked(self.DEFAULT_FORCE_DISTORTION_CENTRE)

        # Restore distortion type
        if pp.distortion_type != self.DEFAULT_DISTORTION_TYPE:
            self._changeDistortionTypeWithStash(self.DEFAULT_DISTORTION_TYPE)
            self.distortion_type.setCurrentIndex(
                pp.distortion_type_list.index(self.DEFAULT_DISTORTION_TYPE))
            self.fit_parameters.changeNumberShown(pp.poly_length)
            self.fit_parameters.updateValues()

        # Restore extinction scale
        if pp.extinction_scale != self.DEFAULT_EXTINCTION_SCALE:
            pp.extinction_scale = self.DEFAULT_EXTINCTION_SCALE
            self.extinction_scale.setValue(self.DEFAULT_EXTINCTION_SCALE)

        # Restore vignetting fixed
        if pp.vignetting_fixed != self.DEFAULT_VIGNETTING_FIXED:
            pp.vignetting_fixed = self.DEFAULT_VIGNETTING_FIXED
            self.vignetting_fixed.setChecked(self.DEFAULT_VIGNETTING_FIXED)
            self.vignetting_coeff.setDisabled(not self.DEFAULT_VIGNETTING_FIXED)

        # Show/hide radial-specific options
        if pp.distortion_type.startswith('radial'):
            self.eqAspect.show()
            self.asymmetryCorr.show()
            self.fdistortion.show()
        else:
            self.eqAspect.hide()
            self.asymmetryCorr.hide()
            self.fdistortion.hide()

        # Update button state
        self.updateRestoreDefaultsButton()

        # Emit signal to update the GUI
        self.sigRestoreDefaultsPressed.emit()

    def updatePairedStars(self, min_fit_stars=4):
        """
        Updates QPushButtons to be enabled/disabled based on the number of paired stars
        Call whenever paired_stars is changed
        """
        n_pairs = len(self.gui.paired_stars)
        self.astrometry_button.setEnabled(n_pairs > 0)
        self.photometry_button.setEnabled(n_pairs >= 2)
        self.fit_astrometry_button.setEnabled(n_pairs >= min_fit_stars)

        # Pair finding needs a platepar to project the catalog; residuals need a platepar and pairs
        has_platepar = getattr(self.gui, 'platepar', None) is not None
        self.find_pairs_button.setEnabled(has_platepar)
        self.compute_residuals_button.setEnabled(has_platepar and (n_pairs > 0))


    def setFitButtonBusy(self, busy):
        """Set the Fit button to show busy state.

        Arguments:
            busy: [bool] If True, show busy state. If False, restore normal state.
        """
        if busy:
            self.fit_astrometry_button.setText("Fitting...")
            self.fit_astrometry_button.setEnabled(False)
            self.auto_fit_button.setEnabled(False)
            self.find_pairs_button.setEnabled(False)
            self.compute_residuals_button.setEnabled(False)
            # Force visual update
            self.fit_astrometry_button.repaint()
            self.auto_fit_button.repaint()
        else:
            self.fit_astrometry_button.setText("Fit")
            # Re-enable based on paired stars count
            self.updatePairedStars(min_fit_stars=self.gui.getMinFitStars())
            self.auto_fit_button.setEnabled(True)


    def setAutoFitButtonBusy(self, busy):
        """Set the Auto Fit button to show busy state.

        Arguments:
            busy: [bool] If True, show busy state. If False, restore normal state.
        """
        if busy:
            self.auto_fit_button.setText("Fitting...")
            self.auto_fit_button.setEnabled(False)
            self.fit_astrometry_button.setEnabled(False)
            self.quick_align_button.setEnabled(False)
            self.find_pairs_button.setEnabled(False)
            self.compute_residuals_button.setEnabled(False)
            # Force visual update
            self.auto_fit_button.repaint()
            self.fit_astrometry_button.repaint()
            self.quick_align_button.repaint()
        else:
            self.auto_fit_button.setText("Auto Fit")
            self.auto_fit_button.setEnabled(True)
            self.quick_align_button.setEnabled(True)
            # Re-enable fit button based on paired stars count
            self.updatePairedStars(min_fit_stars=self.gui.getMinFitStars())

    def setQuickAlignButtonBusy(self, busy):
        """Set the Quick Align button to show busy state.

        Arguments:
            busy: [bool] If True, show busy state. If False, restore normal state.
        """
        if busy:
            self.quick_align_button.setText("Aligning...")
            self.quick_align_button.setEnabled(False)
            self.auto_fit_button.setEnabled(False)
            self.fit_astrometry_button.setEnabled(False)
            # Force visual update
            self.quick_align_button.repaint()
            self.auto_fit_button.repaint()
            self.fit_astrometry_button.repaint()
        else:
            self.quick_align_button.setText("Auto Pointing")
            self.quick_align_button.setEnabled(True)
            self.auto_fit_button.setEnabled(True)
            # Re-enable fit button based on paired stars count
            self.updatePairedStars(min_fit_stars=self.gui.getMinFitStars())


class ArrayTabWidget(QtWidgets.QTabWidget, ScaledSizeHelper):
    """
    Widget to the right which holds the histogram as well as the parameter manager
    This class does not manipulate their values itself, that is done by accessing
    the variables themselves
    """
    valueModified = QtCore.pyqtSignal()

    # Width in characters for coefficient input boxes
    COEFF_INPUT_CHARS = 15

    def __init__(self, platepar):
        super(ArrayTabWidget, self).__init__()
        self.platepar = platepar

        self.vars = ['x_poly_rev', 'y_poly_rev', 'x_poly_fwd', 'y_poly_fwd']
        # Shorter display names for tabs
        self.tab_names = ['x_rev', 'y_rev', 'x_fwd', 'y_fwd']

        # Maximum number of parameters that can be shown
        self.max_n_shown = max(self.platepar.distortion_type_poly_length)

        self.tabs = [QtWidgets.QWidget() for x in range(4)]
        self.layouts = []
        self.boxes = [[], [], [], []]
        self.labels = [[], [], [], []]

        self.n_shown = self.platepar.poly_length

        for i in range(len(self.vars)):
            self.addTab(self.tabs[i], self.tab_names[i])
            self.setupTab(i)

    def changeNumberShown(self, n):
        """
        Change the number of QDoubleSpinBoxes visible

        Arguments:
            n [int]: Number of QDoubleSpinBoxes to be visible
        """
        assert 0 <= n <= self.max_n_shown
        if n == self.n_shown:
            return

        elif n > self.n_shown:
            for i in range(len(self.vars)):
                for j in range(self.n_shown, n):
                    self.layouts[i].insertRow(j, self.labels[i][j], self.boxes[i][j])
                    self.labels[i][j].show()
                    self.boxes[i][j].show()

        elif n < self.n_shown:
            for i in range(len(self.vars)):
                for j in range(n, self.max_n_shown):
                    self.labels[i][j].hide()
                    self.boxes[i][j].hide()
                    self.layouts[i].removeWidget(self.labels[i][j])
                    self.layouts[i].removeWidget(self.boxes[i][j])

        self.n_shown = n

    def setupTab(self, i):
        """ Setup all boxes with polynomial values. """

        layout = QtWidgets.QFormLayout()

        for j in range(self.max_n_shown):
            box = ScientificDoubleSpinBox()
            box.setSingleStep(0.5)
            box.setFixedWidth(self.scaledWidth(self.COEFF_INPUT_CHARS))

            # Set the value to the box from the platepar polynomial
            poly_arr = getattr(self.platepar, self.vars[i])
            if len(poly_arr) > j:
                box.setValue(poly_arr[j])
            else:
                box.setValue(0)

            box.valueModified.connect(self.onFitParameterChanged(i, j))
            # Just show index since tab name identifies the array
            label = QtWidgets.QLabel("[{}]".format(j))

            # Only add to layout if within n_shown; otherwise just store the widgets hidden
            if j < self.n_shown:
                layout.addRow(label, box)
            else:
                # Hide widgets that are beyond the current poly_length
                label.hide()
                box.hide()

            self.boxes[i].append(box)
            self.labels[i].append(label)

        self.setTabText(i, self.tab_names[i])
        self.tabs[i].setLayout(layout)
        self.layouts.append(layout)

    def onFitParameterChanged(self, i, j):
        def f():
            getattr(self.platepar, self.vars[i])[j] = self.boxes[i][j].value()
            self.valueModified.emit()

        return f

    def updateValues(self):
        for i in range(len(self.vars)):
            poly_arr = getattr(self.platepar, self.vars[i])
            for j in range(self.n_shown):
                if len(poly_arr) > j:
                    self.boxes[i][j].setValue(poly_arr[j])


class DistortionDialog(QtWidgets.QDialog, ScaledSizeHelper):
    """
    Non-modal dialog for editing lens distortion parameters.
    Allows manual editing of polynomial coefficients with live preview.
    """
    valueModified = QtCore.pyqtSignal()

    def __init__(self, parent, platepar):
        super(DistortionDialog, self).__init__(parent)
        self.platepar = platepar
        self.parent_widget = parent

        self.setWindowTitle("Lens Distortion Parameters")
        self.setModal(False)  # Non-modal for live preview

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info_label = QtWidgets.QLabel(
            "Edit polynomial coefficients for lens distortion correction.\n"
            "Changes are applied immediately to the image display."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(info_label)

        # Create the array tab widget for coefficients
        self.fit_parameters = ArrayTabWidget(platepar=self.platepar)
        self.fit_parameters.valueModified.connect(self.onValueModified)
        layout.addWidget(self.fit_parameters)

        # Button row
        button_layout = QtWidgets.QHBoxLayout()

        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.resetToZero)
        button_layout.addWidget(self.reset_button)

        button_layout.addStretch()

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        # Set reasonable size (wider to accommodate coefficient values)
        self.resize(self.scaledWidth(40), self.scaledHeight(30))

    def onValueModified(self):
        """Forward signal when a coefficient is modified."""
        self.valueModified.emit()

    def resetToZero(self):
        """Reset distortion coefficients, preserving center/offset terms."""
        # Determine how many leading indices to preserve (center coefficients)
        # For radial distortion without force_distortion_centre, indices 0 and 1 are center x and y
        # For polynomial distortion, index 0 is the offset
        if self.platepar.distortion_type.startswith("radial") and not self.platepar.force_distortion_centre:
            preserve_count = 2  # Preserve x_poly[0] and x_poly[1] for radial center
        else:
            preserve_count = 1  # Preserve index 0 only

        for var in ['x_poly_rev', 'y_poly_rev', 'x_poly_fwd', 'y_poly_fwd']:
            poly = getattr(self.platepar, var)
            # For y_poly in radial, all values should be zero (no center there)
            start_idx = preserve_count if var.startswith('x_') else 1
            for i in range(start_idx, len(poly)):
                poly[i] = 0.0
        self.fit_parameters.updateValues()
        self.valueModified.emit()

    def updatePlatepar(self, platepar):
        """Update the platepar reference and refresh display."""
        self.platepar = platepar
        self.fit_parameters.platepar = platepar
        self.fit_parameters.changeNumberShown(platepar.poly_length)
        self.fit_parameters.updateValues()

    def changeNumberShown(self, n):
        """Change the number of coefficients shown."""
        self.fit_parameters.changeNumberShown(n)

    def updateValues(self):
        """Update displayed values from platepar."""
        self.fit_parameters.updateValues()


class StarDetectionWidget(QtWidgets.QWidget, ScaledSizeHelper):
    """
    Widget for testing and adjusting star detection parameters.
    Allows overriding CALSTARS detection settings to find optimal parameters.
    """
    sigRedetectStars = QtCore.pyqtSignal()
    sigRedetectAllImages = QtCore.pyqtSignal()
    sigTuneParameters = QtCore.pyqtSignal()
    sigSaveCalstarsPressed = QtCore.pyqtSignal()
    sigSaveToConfig = QtCore.pyqtSignal()
    sigCatalogLMChanged = QtCore.pyqtSignal(float)
    sigUseOverrideToggled = QtCore.pyqtSignal()
    sigStarGateFactorChanged = QtCore.pyqtSignal(float)
    sigNeighborhoodSizeChanged = QtCore.pyqtSignal(int)
    sigMaxStarsChanged = QtCore.pyqtSignal(int)
    sigConfigMaxStarsChanged = QtCore.pyqtSignal(int)
    sigMaxGlobalIntensityChanged = QtCore.pyqtSignal(int)
    sigGammaChanged = QtCore.pyqtSignal(float)
    sigSegmentRadiusChanged = QtCore.pyqtSignal(int)
    sigMaxFeatureRatioChanged = QtCore.pyqtSignal(float)
    sigRoundnessThresholdChanged = QtCore.pyqtSignal(float)

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui

        # While True, programmatic slider seeding bypasses the snap-to-100 handlers, so a
        # config value like max_stars=150 survives loading exactly instead of being snapped
        # (and then counting as an unsaved modification with zero user input)
        self._seeding = False

        # The station config last loaded into the sliders - Reset to Defaults returns the
        # station-bound values (gamma, config max stars) to it rather than global defaults
        self._loaded_config = None

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(*self.scaledMargins(1, 0.5))
        layout.setSpacing(self.scaledSpacing(0.3))
        self.setLayout(layout)

        # Tab help button (top-right)
        self.addCornerHelpButton('stardetect', "Help: star detection override")

        # Title
        title = QtWidgets.QLabel('Star Detection Override')
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)

        # Detection parameters, split into what Save Config persists to the station config
        # and what only applies to this SkyFit session
        slider_data = [
            # (key, label, min, max, default, default label, callback, group)
            ('star_gate_factor', 'Adaptive Gate Factor', 15, 60, 30, '3.0', self.onStarGateFactorChanged, 'config'),
            ('neighborhood_size', 'Neighborhood Size', 5, 40, 10, '10', self.onNeighborhoodSizeChanged, 'config'),
            ('config_max_stars', 'Max Stars', 100, 2000, 400, '400', self.onConfigMaxStarsChanged, 'config'),
            ('max_global_intensity', 'Max Global Intensity', 30, 255, 230, '230', self.onMaxGlobalIntensityChanged, 'config'),
            ('gamma', 'Gamma', 45, 200, 100, '1.00', self.onGammaChanged, 'config'),
            ('segment_radius', 'Segment Radius', 2, 20, 4, '4', self.onSegmentRadiusChanged, 'config'),
            ('max_feature_ratio', 'Max Feature Ratio', 50, 200, 80, '0.80', self.onMaxFeatureRatioChanged, 'config'),
            ('roundness_threshold', 'Roundness Threshold', 30, 90, 50, '0.50', self.onRoundnessThresholdChanged, 'config'),
            ('skyfit_max_stars', 'Max Stars', 100, 5000, 2000, '2000', self.onMaxStarsChanged, 'session'),
        ]

        self.sliders = {}
        self.slider_labels = {}
        self.slider_defaults = {}

        # One group box per parameter scope
        group_grids = {}
        group_rows = {}
        for group_key, group_title, group_tip in [
                ('config', 'Station Config',
                 'Written to the station config file by Save Config - these control the nightly pipeline'),
                ('session', 'SkyFit Session Only',
                 'Only used by SkyFit re-detection in this session - never written to the config')]:

            box = QtWidgets.QGroupBox(group_title)
            box.setToolTip(group_tip)
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(self.scaledSpacing(0.3))
            # Keep the group box padding tight so the label column doesn't get clipped
            grid.setContentsMargins(*self.scaledMargins(0.4, 0.3))
            grid.setColumnStretch(0, 1)  # Label column stretches
            grid.setColumnStretch(1, 0)  # Value column fixed
            box.setLayout(grid)
            layout.addWidget(box)

            group_grids[group_key] = grid
            group_rows[group_key] = 0

        for key, name, min_val, max_val, default, default_str, callback, group in slider_data:

            grid = group_grids[group]
            row = group_rows[group]

            # Row with label and value
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            val_label = QtWidgets.QLabel(default_str)
            val_label.setFixedSize(self.scaledWidth(6), self.scaledHeight(1))
            val_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(val_label, row, 1)
            row += 1

            # Row with slider spanning both columns
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.valueChanged.connect(callback)
            grid.addWidget(slider, row, 0, 1, 2)  # span 2 columns
            row += 1

            self.sliders[key] = slider
            self.slider_labels[key] = val_label
            self.slider_defaults[key] = default

            # Keep the Reset to Defaults button color-coded as values move
            slider.valueChanged.connect(self.updateDefaultsButton)

            # Add gamma preset buttons right after gamma slider
            if key == 'gamma':
                gamma_presets = QtWidgets.QHBoxLayout()
                gamma_presets.setSpacing(self.scaledSpacing(0.25))
                gamma_presets.setContentsMargins(0, 0, 0, self.scaledSpacing(0.3))

                btn_gamma_22 = QtWidgets.QPushButton('1/2.2')
                btn_gamma_22.clicked.connect(lambda: self.setGammaPreset(1/2.2))
                gamma_presets.addWidget(btn_gamma_22)

                btn_gamma_18 = QtWidgets.QPushButton('1/1.8')
                btn_gamma_18.clicked.connect(lambda: self.setGammaPreset(1/1.8))
                gamma_presets.addWidget(btn_gamma_18)

                btn_gamma_lin = QtWidgets.QPushButton('Linear')
                btn_gamma_lin.clicked.connect(lambda: self.setGammaPreset(1.0))
                gamma_presets.addWidget(btn_gamma_lin)

                grid.addLayout(gamma_presets, row, 0, 1, 2)
                grid.setRowMinimumHeight(row, self.scaledHeight(1.75))
                row += 1

            group_rows[group] = row

        # Create named references for compatibility
        self.star_gate_factor_slider = self.sliders['star_gate_factor']
        self.star_gate_factor_label = self.slider_labels['star_gate_factor']
        self.neighborhood_size_slider = self.sliders['neighborhood_size']
        self.neighborhood_size_label = self.slider_labels['neighborhood_size']
        self.max_stars_slider = self.sliders['skyfit_max_stars']
        self.max_stars_label = self.slider_labels['skyfit_max_stars']
        self.config_max_stars_slider = self.sliders['config_max_stars']
        self.config_max_stars_label = self.slider_labels['config_max_stars']
        self.max_global_intensity_slider = self.sliders['max_global_intensity']
        self.max_global_intensity_label = self.slider_labels['max_global_intensity']
        self.gamma_slider = self.sliders['gamma']
        self.gamma_label = self.slider_labels['gamma']
        self.segment_radius_slider = self.sliders['segment_radius']
        self.segment_radius_label = self.slider_labels['segment_radius']
        self.max_feature_ratio_slider = self.sliders['max_feature_ratio']
        self.max_feature_ratio_label = self.slider_labels['max_feature_ratio']
        self.roundness_threshold_slider = self.sliders['roundness_threshold']
        self.roundness_threshold_label = self.slider_labels['roundness_threshold']

        # Both max stars budgets move in increments of 100
        for s in (self.max_stars_slider, self.config_max_stars_slider):
            s.setSingleStep(100)
            s.setPageStep(500)

        # Tooltips distinguishing the two star count budgets
        self.max_stars_slider.setToolTip(
            'Number of star candidates used by SkyFit re-detection in this session.\n'
            'Initial calibration benefits from a deep sample - this value is never saved.')
        self.config_max_stars_slider.setToolTip(
            'max_stars value written to the station config by Save Config.\n'
            'This bounds the star extraction cost of the nightly pipeline - 400 recommended.')
        self.gamma_slider.setToolTip(
            'Camera gamma used for detection and photometry.\n'
            'Written to the config [Capture] section by Save Config, and also stored in the platepar.')

        layout.addSpacing(self.scaledSpacing(1))

        # Buttons in their own layout with spacing
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(self.scaledSpacing(0.5))

        self.redetect_button = QtWidgets.QPushButton('Re-Detect Current')
        self.redetect_button.clicked.connect(self.sigRedetectStars.emit)
        btn_layout.addWidget(self.redetect_button)

        self.redetect_all_button = QtWidgets.QPushButton('Re-Detect All')
        self.redetect_all_button.clicked.connect(self.sigRedetectAllImages.emit)
        btn_layout.addWidget(self.redetect_all_button)

        self.tune_button = QtWidgets.QPushButton('Tune')
        self.tune_button.setToolTip('Auto-find optimal threshold and segment radius')
        self.tune_button.clicked.connect(self.sigTuneParameters.emit)
        btn_layout.addWidget(self.tune_button)

        self.defaults_button = QtWidgets.QPushButton('Reset to Defaults')
        self.defaults_button.setToolTip('Reset all star detection parameters to the recommended defaults')
        self.defaults_button.clicked.connect(self.resetToDefaults)
        self.updateDefaultsButton()
        btn_layout.addWidget(self.defaults_button)

        self.save_config_button = QtWidgets.QPushButton('Save Config...')
        self.save_config_button.setToolTip('Open File Manager to save star detection settings')
        self.save_config_button.setEnabled(False)
        self.save_config_button.clicked.connect(self.sigSaveToConfig.emit)
        btn_layout.addWidget(self.save_config_button)

        self.save_calstars_button = QtWidgets.QPushButton('Save CALSTARS...')
        self.save_calstars_button.setToolTip(
            'Write the current star data (CALSTARS merged with any re-detected frames)\n'
            'to a CALSTARS file, e.g. for sharing or offline analysis')
        self.save_calstars_button.clicked.connect(self.sigSaveCalstarsPressed.emit)
        btn_layout.addWidget(self.save_calstars_button)

        layout.addLayout(btn_layout)
        layout.addSpacing(self.scaledSpacing(0.6))

        # Catalog Limiting Magnitude (synced with Settings panel)
        lm_layout = QtWidgets.QHBoxLayout()
        lm_layout.addWidget(QtWidgets.QLabel('Lim Mag'))
        self.catalog_lm_spinbox = DoubleSpinBox()
        self.catalog_lm_spinbox.setSingleStep(0.1)
        self.catalog_lm_spinbox.setMinimum(3.0)
        self.catalog_lm_spinbox.setMaximum(12.0)
        self.catalog_lm_spinbox.setDecimals(1)
        self.catalog_lm_spinbox.valueModified.connect(self.onCatalogLMChanged)
        lm_layout.addWidget(self.catalog_lm_spinbox)
        lm_layout.addStretch()
        layout.addLayout(lm_layout)

        layout.addSpacing(self.scaledSpacing(0.3))

        # Checkbox
        self.use_override_checkbox = QtWidgets.QCheckBox('Use Override Detections')
        self.use_override_checkbox.released.connect(self.sigUseOverrideToggled.emit)
        layout.addWidget(self.use_override_checkbox)

        layout.addSpacing(self.scaledSpacing(0.3))

        # Status
        self.status_label = QtWidgets.QLabel('Using original CALSTARS')
        self.status_label.setStyleSheet("color: gray; font-size: 10pt;")
        # Wrap instead of clipping - "(N stars, M candidates)" can outgrow the panel
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()


    def onStarGateFactorChanged(self, value):
        factor = value/10.0
        self.star_gate_factor_label.setText(f'{factor:.1f}')
        self.sigStarGateFactorChanged.emit(factor)

    def onNeighborhoodSizeChanged(self, value):
        self.neighborhood_size_label.setText(str(value))
        self.sigNeighborhoodSizeChanged.emit(value)

    @staticmethod
    def _snapTo100(value):
        """Snap a max stars value to the nearest 100."""
        return max(100, int(round(value/100.0))*100)

    def onMaxStarsChanged(self, value):
        # Snap only user motion - programmatic seeding keeps the exact config value
        if not self._seeding:
            snapped = self._snapTo100(value)
            if snapped != value:
                # Re-fires this handler with the snapped value
                self.max_stars_slider.setValue(snapped)
                return
        self.max_stars_label.setText(str(value))
        self.sigMaxStarsChanged.emit(value)

    def onConfigMaxStarsChanged(self, value):
        # Snap only user motion - programmatic seeding keeps the exact config value
        if not self._seeding:
            snapped = self._snapTo100(value)
            if snapped != value:
                # Re-fires this handler with the snapped value
                self.config_max_stars_slider.setValue(snapped)
                return
        self.config_max_stars_label.setText(str(value))
        self.sigConfigMaxStarsChanged.emit(value)

    # Values set by the auto-tuner: Reset to Defaults must not clobber them (the
    # catalog LM spinbox is tuned too and is likewise left untouched)
    TUNED_KEYS = ('segment_radius',)

    def _sliderRestoreTarget(self, key):
        """ The value Reset to Defaults would set for a slider (in slider units), or
            None for tuned values which are never restored. """

        if key in self.TUNED_KEYS:
            return None

        cfg = getattr(self, '_loaded_config', None)

        # Station-bound values return to the loaded config instead: gamma is a hardware
        # property of the camera (resetting it to 1.0 would corrupt photometry the
        # moment Save Config is pressed), and the config max stars is the station's
        # pipeline budget, not a tuning preference
        if key == 'gamma':
            if cfg is not None and hasattr(cfg, 'gamma'):
                return int(round(cfg.gamma*100))
            return self.slider_defaults[key]

        if key == 'config_max_stars':
            if cfg is not None and hasattr(cfg, 'max_stars'):
                return int(cfg.max_stars)
            return self.slider_defaults[key]

        return self.slider_defaults[key]

    def isAtDefaults(self):
        """ True when every non-tuned slider sits at its restore target. """

        for key in self.slider_defaults:
            target = self._sliderRestoreTarget(key)
            if target is None:
                continue
            if self.sliders[key].value() != target:
                return False

        return True

    def updateDefaultsButton(self):
        """ Color-code Reset to Defaults: green at defaults, amber when values differ
            (same convention as the Fit Parameters tab). """

        if not hasattr(self, 'defaults_button'):
            return

        at_defaults = self.isAtDefaults()
        if at_defaults:
            self.defaults_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; }"
                "QPushButton:hover { background-color: #45a049; }"
            )
        else:
            self.defaults_button.setStyleSheet(
                "QPushButton { background-color: #FF9800; color: white; }"
                "QPushButton:hover { background-color: #F57C00; }"
            )
        self.defaults_button.setEnabled(not at_defaults)

    def resetToDefaults(self):
        """Reset every non-tuned slider to its restore target (widget default, or the
        loaded config for station-bound values). Tuned values (TUNED_KEYS and the
        catalog LM) are left untouched - retuning is the only way they should move.
        """

        for key in self.slider_defaults:
            target = self._sliderRestoreTarget(key)
            if target is None:
                continue

            if key == 'config_max_stars':
                self._seedSlider(self.config_max_stars_slider, target)
                continue

            # setValue triggers each slider's callback, so labels and override
            # values in SkyFit update through the normal signal path
            self.sliders[key].setValue(target)

        self.updateDefaultsButton()

    def _seedSlider(self, slider, value):
        """ Set a slider value programmatically, bypassing the snap-to-100 handlers so the
            exact value survives (the change signal still fires normally). Extends the
            slider range if needed so out-of-range config values are not clamped. """
        if value < slider.minimum():
            slider.setMinimum(value)
        if value > slider.maximum():
            slider.setMaximum(value)
        self._seeding = True
        try:
            slider.setValue(value)
        finally:
            self._seeding = False

    def onMaxGlobalIntensityChanged(self, value):
        self.max_global_intensity_label.setText(str(value))
        self.sigMaxGlobalIntensityChanged.emit(value)

    def onGammaChanged(self, value):
        gamma = value / 100.0
        self.gamma_label.setText(f'{gamma:.2f}')
        self.sigGammaChanged.emit(gamma)

    def setGammaPreset(self, gamma):
        """Set gamma to a preset value."""
        self.gamma_slider.setValue(int(gamma * 100))

    def onSegmentRadiusChanged(self, value):
        self.segment_radius_label.setText(str(value))
        self.sigSegmentRadiusChanged.emit(value)

    def onMaxFeatureRatioChanged(self, value):
        ratio = value / 100.0
        self.max_feature_ratio_label.setText(f'{ratio:.2f}')
        self.sigMaxFeatureRatioChanged.emit(ratio)

    def onRoundnessThresholdChanged(self, value):
        threshold = value / 100.0
        self.roundness_threshold_label.setText(f'{threshold:.2f}')
        self.sigRoundnessThresholdChanged.emit(threshold)

    def onCatalogLMChanged(self):
        """Handle catalog LM spinbox change."""
        self.sigCatalogLMChanged.emit(self.catalog_lm_spinbox.value())

    def setCatalogLM(self, value):
        """Update the catalog LM spinbox without triggering signal."""
        self.catalog_lm_spinbox.blockSignals(True)
        self.catalog_lm_spinbox.setValue(value)
        self.catalog_lm_spinbox.blockSignals(False)

    def updateStatus(self, using_override, star_count=None, candidate_count=None):
        """Update the status label to show current detection source."""
        pad = self.scaledSpacing(0.3)
        if using_override:
            if star_count is not None:
                text = f'Using override detection ({star_count} stars'
                if candidate_count is not None:
                    text += f', {candidate_count} candidates'
                text += ')'
                self.status_label.setText(text)
                self.status_label.setStyleSheet(f"color: green; font-size: 9pt; padding: {pad}px; font-weight: bold;")
            else:
                self.status_label.setText('Using override detection')
                self.status_label.setStyleSheet(f"color: green; font-size: 9pt; padding: {pad}px; font-weight: bold;")
        else:
            self.status_label.setText('Using original CALSTARS')
            self.status_label.setStyleSheet(f"color: gray; font-size: 9pt; padding: {pad}px;")

    def loadFromConfig(self, config):
        """Initialize sliders from config values."""

        self._loaded_config = config

        if hasattr(config, 'star_gate_factor'):
            self.star_gate_factor_slider.setValue(int(round(config.star_gate_factor*10)))

        self.updateDefaultsButton()
        if hasattr(config, 'neighborhood_size'):
            self.neighborhood_size_slider.setValue(config.neighborhood_size)
        if hasattr(config, 'max_stars'):
            # Only the CONFIG budget tracks the station config (it is what Save Config
            # writes back). The session budget is a desktop compute limit for calibration
            # work - seeding it from the Pi's pipeline budget would drag the deep session
            # default down on every folder load. Seed without snapping so a config value
            # like 150 loads exactly - otherwise the seeded value counts as an unsaved
            # config modification with zero user input, and Save Config would write the
            # snapped value back to the station config
            self._seedSlider(self.config_max_stars_slider, config.max_stars)
        if hasattr(config, 'max_global_intensity'):
            self.max_global_intensity_slider.setValue(config.max_global_intensity)
        if hasattr(config, 'gamma'):
            self.gamma_slider.setValue(int(config.gamma * 100))
        if hasattr(config, 'segment_radius'):
            self.segment_radius_slider.setValue(config.segment_radius)
        if hasattr(config, 'max_feature_ratio'):
            self.max_feature_ratio_slider.setValue(int(config.max_feature_ratio * 100))
        if hasattr(config, 'roundness_threshold'):
            self.roundness_threshold_slider.setValue(int(config.roundness_threshold * 100))


class BrushCursorItem(pg.GraphicsObject):
    """Circle outline that follows the mouse in brush mask mode.

    The radius is in image coordinates (scales with zoom) but the pen is cosmetic
    (always 1 px on screen) so the outline stays crisp at any zoom level.
    """

    def __init__(self):
        super().__init__()
        self._radius = 20.0
        self._pen = QtGui.QPen(QtGui.QColor(0, 255, 255, 200))
        self._pen.setCosmetic(True)
        self._pen.setWidth(2)

    def setRadius(self, r):
        self._radius = float(r)
        self.prepareGeometryChange()
        self.update()

    def setCenter(self, pos):
        self.setPos(pos)

    def paint(self, painter, option, widget=None):
        painter.setPen(self._pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(0, 0), self._radius, self._radius)

    def boundingRect(self):
        r = self._radius + 2
        return QtCore.QRectF(-r, -r, 2*r, 2*r)


class MaskWidget(QtWidgets.QWidget, ScaledSizeHelper):
    """
    Widget for creating and editing mask polygons and brush strokes.
    Click to add points, right-click to close polygon.
    """
    sigDrawModeToggled = QtCore.pyqtSignal()
    sigClearPolygons = QtCore.pyqtSignal()
    sigSaveMask = QtCore.pyqtSignal()
    sigLoadMask = QtCore.pyqtSignal()
    sigShowOverlayToggled = QtCore.pyqtSignal(bool)
    sigUnsavedChanged = QtCore.pyqtSignal()
    sigUseFlatToggled = QtCore.pyqtSignal(bool)
    sigInvertMask = QtCore.pyqtSignal()
    sigBrushModeToggled = QtCore.pyqtSignal()
    sigClearBrushStrokes = QtCore.pyqtSignal()
    sigBrushSizeChanged = QtCore.pyqtSignal(int)
    sigUndoBrushStroke = QtCore.pyqtSignal()

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui
        self.unsaved = False

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(*self.scaledMargins(1, 0.5))
        layout.setSpacing(self.scaledSpacing(0.5))
        self.setLayout(layout)

        # Tab help button (top-right)
        self.addCornerHelpButton('mask', "Help: drawing a mask")

        # ── Header ────────────────────────────────────────────────────────────

        title = QtWidgets.QLabel('Mask Editor')
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)

        # Status sits at the top so the save state is always visible
        self.status_label = QtWidgets.QLabel('No mask')
        self.status_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.status_label)

        layout.addSpacing(self.scaledSpacing(0.6))

        # ── Mode selection ────────────────────────────────────────────────────
        # Draw Polygon and Paint Brush are mutually exclusive modes.
        # Side-by-side layout makes the exclusivity obvious at a glance.

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.setSpacing(self.scaledSpacing(0.25))

        self.draw_button = QtWidgets.QPushButton('Draw Polygon')
        self.draw_button.setCheckable(True)
        self.draw_button.clicked.connect(self.onDrawToggled)
        mode_layout.addWidget(self.draw_button)

        self.brush_button = QtWidgets.QPushButton('Paint Brush')
        self.brush_button.setCheckable(True)
        self.brush_button.clicked.connect(self.onBrushToggled)
        mode_layout.addWidget(self.brush_button)

        layout.addLayout(mode_layout)

        # Instructions update dynamically when the active mode changes
        self.instructions = QtWidgets.QLabel()
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        self._updateInstructions()

        layout.addSpacing(self.scaledSpacing(0.4))

        # ── Brush controls ────────────────────────────────────────────────────
        # These controls are greyed out when polygon mode is active.
        # Brush size is also adjustable via Shift+scroll on the image.

        brush_size_layout = QtWidgets.QHBoxLayout()
        brush_size_layout.setSpacing(self.scaledSpacing(0.25))

        self.brush_size_label = QtWidgets.QLabel('Brush size:')
        brush_size_layout.addWidget(self.brush_size_label)

        self.brush_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_size_slider.setRange(1, 200)
        self.brush_size_slider.setValue(20)
        self.brush_size_slider.valueChanged.connect(self._onBrushSizeChanged)
        brush_size_layout.addWidget(self.brush_size_slider)

        self.brush_size_value = QtWidgets.QLabel('20')
        self.brush_size_value.setMinimumWidth(self.scaledSpacing(2))
        brush_size_layout.addWidget(self.brush_size_value)

        layout.addLayout(brush_size_layout)

        self.undo_brush_button = QtWidgets.QPushButton('Undo Last Stroke  Ctrl+Z')
        self.undo_brush_button.setEnabled(False)
        self.undo_brush_button.clicked.connect(self.sigUndoBrushStroke.emit)
        layout.addWidget(self.undo_brush_button)

        # Disable brush controls until brush mode is activated
        self._setBrushSectionEnabled(False)

        layout.addSpacing(self.scaledSpacing(0.6))

        # ── Mask operations ───────────────────────────────────────────────────
        # Invert flips the entire mask (both polygons and paint layer).
        # Clear All removes everything; Clear Brush removes only paint strokes.

        self.invert_button = QtWidgets.QPushButton('Invert Mask')
        self.invert_button.clicked.connect(self.sigInvertMask.emit)
        layout.addWidget(self.invert_button)

        clear_layout = QtWidgets.QHBoxLayout()
        clear_layout.setSpacing(self.scaledSpacing(0.25))

        self.clear_button = QtWidgets.QPushButton('Clear All')
        self.clear_button.clicked.connect(self.onClearAll)
        clear_layout.addWidget(self.clear_button)

        self.clear_brush_button = QtWidgets.QPushButton('Clear Brush')
        self.clear_brush_button.clicked.connect(self.onClearBrush)
        clear_layout.addWidget(self.clear_brush_button)

        layout.addLayout(clear_layout)

        layout.addSpacing(self.scaledSpacing(0.6))

        # ── File operations ───────────────────────────────────────────────────

        file_layout = QtWidgets.QHBoxLayout()
        file_layout.setSpacing(self.scaledSpacing(0.25))

        self.load_button = QtWidgets.QPushButton('Load')
        self.load_button.clicked.connect(self.sigLoadMask.emit)
        file_layout.addWidget(self.load_button)

        self.save_button = QtWidgets.QPushButton('Save Mask...')
        self.save_button.setToolTip('Open File Manager to save mask')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.sigSaveMask.emit)
        file_layout.addWidget(self.save_button)

        layout.addLayout(file_layout)

        layout.addSpacing(self.scaledSpacing(0.6))

        # ── Display options ───────────────────────────────────────────────────

        self.show_overlay = QtWidgets.QCheckBox('Show Mask Overlay')
        self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self.sigShowOverlayToggled.emit)
        layout.addWidget(self.show_overlay)

        self.use_flat = QtWidgets.QCheckBox('Use Flat as Background')
        self.use_flat.setChecked(False)  # toggled to True by checkAndSetupFlatForMask if flat.bmp exists
        self.use_flat.toggled.connect(self.sigUseFlatToggled.emit)
        layout.addWidget(self.use_flat)
        self.flat_available = False

        layout.addStretch()

    def _setBrushSectionEnabled(self, enabled):
        """Grey out brush size controls when not in brush mode."""
        self.brush_size_label.setEnabled(enabled)
        self.brush_size_slider.setEnabled(enabled)
        self.brush_size_value.setEnabled(enabled)
        if not enabled:
            self.undo_brush_button.setEnabled(False)

    def _updateInstructions(self):
        """Update instructions text to match the currently active mode."""

        if hasattr(self, 'brush_button') and self.brush_button.isChecked():
            self.instructions.setText(
                '<b>Paint Brush mode:</b><br>'
                '&bull; Left-click drag to mask<br>'
                '&bull; Right-click drag to erase<br>'
                '&bull; Shift+scroll to resize brush<br>'
                '&bull; Ctrl+Z to undo last stroke')

        elif hasattr(self, 'draw_button') and self.draw_button.isChecked():
            self.instructions.setText(
                '<b>Draw Polygon mode:</b><br>'
                '&bull; Click to add points<br>'
                '&bull; Space/Enter to close polygon<br><br>'
                '<b>Edit existing polygons:</b><br>'
                '&bull; Drag vertices to move<br>'
                '&bull; Right-click vertex to delete<br>'
                '&bull; Ctrl+click edge to add vertex<br><br>'
                'Vertices near image border<br>'
                'will snap to the edge.')

        else:
            # No mode active — show a brief overview of the two options
            self.instructions.setText(
                'Select a mode above to start masking.<br><br>'
                '<b>Draw Polygon</b> — click to place<br>'
                'vertices, then close to fill a region.<br><br>'
                '<b>Paint Brush</b> — freehand paint<br>'
                '(left-click) or erase (right-click).')

    def onDrawToggled(self):
        """Handle draw button toggle."""
        if self.draw_button.isChecked():
            self.draw_button.setText('Drawing... (Space to close)')
            self.draw_button.setStyleSheet("background-color: #FFA500;")
            if self.brush_button.isChecked():
                self.brush_button.setChecked(False)
                self.brush_button.setText('Paint Brush')
                self.brush_button.setStyleSheet("")
                self.sigBrushModeToggled.emit()
        else:
            self.draw_button.setText('Draw Polygon')
            self.draw_button.setStyleSheet("")
        self._updateInstructions()
        self._setBrushSectionEnabled(self.brush_button.isChecked())
        self.sigDrawModeToggled.emit()

    def setDrawMode(self, enabled):
        """Set draw mode from external call."""
        self.draw_button.setChecked(enabled)
        if enabled:
            self.draw_button.setText('Drawing... (Space to close)')
            self.draw_button.setStyleSheet("background-color: #FFA500;")
        else:
            self.draw_button.setText('Draw Polygon')
            self.draw_button.setStyleSheet("")
        self._updateInstructions()
        self._setBrushSectionEnabled(self.brush_button.isChecked())

    def onBrushToggled(self):
        """Handle brush button toggle."""
        if self.brush_button.isChecked():
            self.brush_button.setText('Painting...')
            self.brush_button.setStyleSheet("background-color: #00BFFF;")
            if self.draw_button.isChecked():
                self.draw_button.setChecked(False)
                self.draw_button.setText('Draw Polygon')
                self.draw_button.setStyleSheet("")
                self.sigDrawModeToggled.emit()
        else:
            self.brush_button.setText('Paint Brush')
            self.brush_button.setStyleSheet("")
        self._updateInstructions()
        self._setBrushSectionEnabled(self.brush_button.isChecked())
        self.sigBrushModeToggled.emit()

    def setBrushMode(self, enabled):
        """Set brush mode from external call."""
        self.brush_button.setChecked(enabled)
        if enabled:
            self.brush_button.setText('Painting...')
            self.brush_button.setStyleSheet("background-color: #00BFFF;")
        else:
            self.brush_button.setText('Paint Brush')
            self.brush_button.setStyleSheet("")
        self._updateInstructions()
        self._setBrushSectionEnabled(enabled)

    def _onBrushSizeChanged(self, value):
        """Handle brush size slider change."""
        self.brush_size_value.setText(str(value))
        self.sigBrushSizeChanged.emit(value)

    def onClearAll(self):
        """Confirm and clear all polygons."""
        reply = QtWidgets.QMessageBox.question(self, 'Clear All',
            'Delete all mask polygons?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.sigClearPolygons.emit()

    def onClearBrush(self):
        """Confirm and clear all brush strokes."""
        reply = QtWidgets.QMessageBox.question(self, 'Clear Brush Strokes',
            'Delete all brush strokes?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.sigClearBrushStrokes.emit()

    def updateStatus(self, polygon_count, drawing_points=0, has_brush_strokes=False):
        """Update the status label."""

        if drawing_points > 0:
            self.status_label.setText(f'Drawing: {drawing_points} points')
            self.status_label.setStyleSheet("color: orange; font-size: 9pt;")
        else:
            has_data = polygon_count > 0 or has_brush_strokes

            if self.unsaved and has_data:
                self.status_label.setText('Mask modified (unsaved)')
                self.status_label.setStyleSheet("color: orange; font-size: 9pt;")
            elif self.unsaved:
                self.status_label.setText('Mask cleared (unsaved)')
                self.status_label.setStyleSheet("color: orange; font-size: 9pt;")
            elif has_data:
                self.status_label.setText('Mask saved')
                self.status_label.setStyleSheet("color: green; font-size: 9pt;")
            else:
                self.status_label.setText('No mask')
                self.status_label.setStyleSheet("color: gray; font-size: 9pt;")

    def setUndoEnabled(self, enabled):
        """Enable or disable the undo brush button."""
        self.undo_brush_button.setEnabled(enabled)

    def setUnsaved(self, unsaved=True):
        """Mark polygons as having unsaved changes."""
        self.unsaved = unsaved
        self.save_button.setEnabled(unsaved)
        self.sigUnsavedChanged.emit()

    def setFlatAvailable(self, available, use_by_default=True):
        """Set whether flat.bmp is available and optionally use it by default."""
        self.flat_available = available
        self.use_flat.setEnabled(available)
        if available:
            self.use_flat.setText('Use Flat as Background')
            if use_by_default:
                self.use_flat.setChecked(True)
        else:
            self.use_flat.setText('Use Flat as Background (not found)')
            self.use_flat.setChecked(False)


class SettingsWidget(QtWidgets.QWidget, ScaledSizeHelper):
    """
    QWidget which displays all of the visual values of the gui. Changing any parameters
    here will not affect the functionality of the gui and will not be saved with savestate.
    """
    sigMaxAveToggled = QtCore.pyqtSignal()
    sigCatStarsToggled = QtCore.pyqtSignal()
    sigSpectralTypeToggled = QtCore.pyqtSignal()
    sigStarNamesToggled = QtCore.pyqtSignal()
    sigApparentMagCorrToggled = QtCore.pyqtSignal()
    sigLabelMagLimitChanged = QtCore.pyqtSignal(float)
    sigConstellationToggled = QtCore.pyqtSignal()
    sigCalStarsToggled = QtCore.pyqtSignal()
    sigDistortionToggled = QtCore.pyqtSignal()
    sigMeasGroundPointsToggled = QtCore.pyqtSignal()
    sigInvertToggled = QtCore.pyqtSignal()
    sigGridToggled = QtCore.pyqtSignal()
    sigSelStarsToggled = QtCore.pyqtSignal()
    sigPicksToggled = QtCore.pyqtSignal()
    sigGreatCircleToggled = QtCore.pyqtSignal()
    sigRegionToggled = QtCore.pyqtSignal()
    sigSingleClickPhotometryToggled = QtCore.pyqtSignal()
    sigErrorOverlayToggled = QtCore.pyqtSignal()
    sigErrorOverlayThresholdChanged = QtCore.pyqtSignal(float)
    sigSatTracksToggled = QtCore.pyqtSignal()
    sigAutoComputeSatTracksToggled = QtCore.pyqtSignal()
    sigLoadTLEPressed = QtCore.pyqtSignal()
    sigClearTLEPressed = QtCore.pyqtSignal()
    sigRedrawSatTracksPressed = QtCore.pyqtSignal()
    sigCatalogChanged = QtCore.pyqtSignal(str)  # Emits the selected catalog filename

    def __init__(self, gui):
        QtWidgets.QWidget.__init__(self)
        self.gui = gui

        vbox = QtWidgets.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(vbox)

        # Tab help button (top-right)
        self.addCornerHelpButton('settings', "Help: the Settings tab")

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

        self.show_spectral_type = QtWidgets.QCheckBox('Show Spectral Type')
        self.show_spectral_type.released.connect(self.sigSpectralTypeToggled.emit)
        self.updateShowSpectralType()
        vbox.addWidget(self.show_spectral_type)

        self.show_star_names = QtWidgets.QCheckBox('Show Star Names')
        self.show_star_names.released.connect(self.sigStarNamesToggled.emit)
        self.updateShowStarNames()
        vbox.addWidget(self.show_star_names)

        # Label magnitude limit spinbox (horizontal layout with label)
        label_mag_layout = QtWidgets.QHBoxLayout()
        label_mag_layout.addWidget(QtWidgets.QLabel('Label Mag Limit:'))
        self.label_mag_spinbox = QtWidgets.QDoubleSpinBox()
        self.label_mag_spinbox.setRange(0.0, 12.0)
        self.label_mag_spinbox.setValue(5.0)
        self.label_mag_spinbox.setSingleStep(0.5)
        self.label_mag_spinbox.setDecimals(1)
        self.label_mag_spinbox.setToolTip('Show labels for stars brighter than this magnitude')
        self.label_mag_spinbox.valueChanged.connect(self.sigLabelMagLimitChanged.emit)
        label_mag_layout.addWidget(self.label_mag_spinbox)
        label_mag_layout.addStretch()
        vbox.addLayout(label_mag_layout)

        self.show_constellations = QtWidgets.QCheckBox('Show Constellation Lines')
        self.show_constellations.released.connect(self.sigConstellationToggled.emit)
        self.updateShowConstellations()
        vbox.addWidget(self.show_constellations)

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

        self.great_circle = QtWidgets.QCheckBox('Show Great Circle Line')
        self.great_circle.released.connect(self.sigGreatCircleToggled.emit)
        self.updateShowGreatCircle()
        self.great_circle.hide()
        vbox.addWidget(self.great_circle)

        self.region = QtWidgets.QCheckBox('Show Photometry Highlight')
        self.region.released.connect(self.sigRegionToggled.emit)
        self.updateShowRegion()
        self.region.hide()
        vbox.addWidget(self.region)

        self.single_click_photometry = QtWidgets.QCheckBox('Single Click Photometry')
        self.single_click_photometry.released.connect(self.sigSingleClickPhotometryToggled.emit)
        self.updateSingleClickPhotometry()
        vbox.addWidget(self.single_click_photometry)

        self.distortion = QtWidgets.QCheckBox('Show Distortion')
        self.distortion.released.connect(self.sigDistortionToggled.emit)
        self.updateShowDistortion()
        vbox.addWidget(self.distortion)

        # Round-trip error overlay controls
        self.error_overlay_chk = QtWidgets.QCheckBox('Show Round-Trip Error Overlay')
        self.error_overlay_chk.setToolTip(
            'Heatmap of the disagreement (px) between the forward and reverse\n'
            'astrometric mappings across the image. Bright areas mark where the\n'
            'two distortion polynomials are inconsistent.')
        self.error_overlay_chk.released.connect(self.sigErrorOverlayToggled.emit)
        self.error_overlay_chk.setChecked(True)  # Default ON
        vbox.addWidget(self.error_overlay_chk)

        error_overlay_hbox = QtWidgets.QHBoxLayout()
        error_overlay_hbox.addWidget(QtWidgets.QLabel('Threshold:'))

        self.error_overlay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.error_overlay_slider.setMinimum(1)    # 0.01 px
        self.error_overlay_slider.setMaximum(300)  # 3.0 px
        self.error_overlay_slider.setValue(50)     # Default 0.5 px
        self.error_overlay_slider.setToolTip(
            'Transparency threshold - errors below this (px) are invisible')
        self.error_overlay_slider.valueChanged.connect(
            lambda v: self.sigErrorOverlayThresholdChanged.emit(v/100.0))
        error_overlay_hbox.addWidget(self.error_overlay_slider)

        self.error_overlay_value_label = QtWidgets.QLabel('0.50 px')
        self.error_overlay_value_label.setMinimumWidth(self.scaledWidth(6))
        error_overlay_hbox.addWidget(self.error_overlay_value_label)
        vbox.addLayout(error_overlay_hbox)

        self.invert = QtWidgets.QCheckBox('Invert Colors')
        self.invert.released.connect(self.sigInvertToggled.emit)
        try:
            self.updateInvertColours()
        except AttributeError:
            self.invert.setChecked(False)
        vbox.addWidget(self.invert)


        self.meas_ground_points = QtWidgets.QCheckBox('Measure ground points')
        self.meas_ground_points.released.connect(self.sigMeasGroundPointsToggled.emit)
        self.updateMeasGroundPoints()
        self.meas_ground_points.hide()
        vbox.addWidget(self.meas_ground_points)

        vbox.addWidget(QHSeparationLine())

        self.sat_tracks = QtWidgets.QCheckBox('Show Satellite Tracks')
        self.sat_tracks.released.connect(self.sigSatTracksToggled.emit)
        # self.sat_tracks.hide() # Always show it so user can turn it on
        vbox.addWidget(self.sat_tracks)

        self.auto_compute_sat_tracks = QtWidgets.QCheckBox(' Automatically compute tracks')
        self.auto_compute_sat_tracks.released.connect(self.sigAutoComputeSatTracksToggled.emit)
        self.auto_compute_sat_tracks.setChecked(False)  # Default: disabled to avoid performance issues
        self.updateShowSatTracks()  # Initialize after both checkboxes are created
        vbox.addWidget(self.auto_compute_sat_tracks)

        self.tle_label = QtWidgets.QLabel("TLE: latest downloaded")
        self.tle_label.setWordWrap(True)
        vbox.addWidget(self.tle_label)

        self.load_tle_btn = QtWidgets.QPushButton("Load TLE File")
        self.load_tle_btn.released.connect(self.sigLoadTLEPressed.emit)
        vbox.addWidget(self.load_tle_btn)

        self.clear_tle_btn = QtWidgets.QPushButton("Reset TLE Selection")
        self.clear_tle_btn.released.connect(self.sigClearTLEPressed.emit)
        vbox.addWidget(self.clear_tle_btn)

        self.redraw_tracks_btn = QtWidgets.QPushButton("Redraw Satellite Tracks")
        self.redraw_tracks_btn.released.connect(self.sigRedrawSatTracksPressed.emit)
        vbox.addWidget(self.redraw_tracks_btn)

        vbox.addWidget(QHSeparationLine())


        vbox.addWidget(QtWidgets.QLabel('Grid:'))
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(2)
        grid_group = QtWidgets.QButtonGroup()
        self.grid = []
        for i, text in enumerate(['None', 'Ra/Dec', 'Az/Alt']):
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

        self.apparent_mag_corr = QtWidgets.QCheckBox('Correct Mag for Ext./Vign.')
        self.apparent_mag_corr.setToolTip(
            'Correct catalog magnitudes for atmospheric extinction and\n'
            'lens vignetting before applying the Lim Mag filter'
        )
        self.apparent_mag_corr.setChecked(False)
        self.apparent_mag_corr.released.connect(self.sigApparentMagCorrToggled.emit)
        form.addRow(self.apparent_mag_corr)

        self.std = DoubleSpinBox()
        self.std.setSingleStep(0.1)
        self.std.setMinimum(0)
        self.std.setValue(self.gui.stdev_text_filter)
        self.std.valueModified.connect(self.onStdChanged)
        self.std_label = QtWidgets.QLabel('Filter Mag Err')
        form.addRow(self.std_label, self.std)

        form.addRow(QtWidgets.QLabel("Press Enter to accept value"))

        # Star catalog selector - on separate lines for full width
        vbox.addWidget(QtWidgets.QLabel('Star Catalog:'))
        self.catalog_combo = QtWidgets.QComboBox()
        self.catalog_combo.setToolTip("Select star catalog (catalogs with star names show names when enabled)")
        self.populateCatalogList()
        self.catalog_combo.currentTextChanged.connect(self.onCatalogChanged)
        vbox.addWidget(self.catalog_combo)

    def updateMaxAvePixel(self):
        self.ave_pixel.setChecked(self.gui.img_type_flag == 'avepixel')
        self.max_pixel.setChecked(self.gui.img_type_flag == 'maxpixel')

    def updateShowCatStars(self):
        self.catalog_stars.setChecked(self.gui.catalog_stars_visible)

    def updateShowSpectralType(self):
        self.show_spectral_type.setChecked(self.gui.show_spectral_type)

    def updateShowStarNames(self):
        self.show_star_names.setChecked(self.gui.show_star_names)

    def updateShowConstellations(self):
        self.show_constellations.setChecked(self.gui.show_constellations)

    def updateShowCalStars(self):
        self.detected_stars.setChecked(self.gui.draw_calstars)

    def updateShowSelStars(self):
        self.selected_stars.setChecked(self.gui.selected_stars_visible)

    def updateShowPicks(self):
        self.picks.setChecked(self.gui.pick_marker.isVisible())

    def updateShowGreatCircle(self):
        self.great_circle.setChecked(self.gui.great_circle_line.isVisible())

    def updateShowRegion(self):
        self.region.setChecked(self.gui.region.isVisible())

    def updateShowDistortion(self):
        self.distortion.setChecked(self.gui.draw_distortion)

    def updateMeasGroundPoints(self):
        self.meas_ground_points.setChecked(self.gui.meas_ground_points)

    def updateShowGrid(self):
        for i, button in enumerate(self.grid):
            button.setChecked(self.gui.grid_visible == i)

    def updateInvertColours(self):
        self.invert.setChecked(self.gui.img.invert_img)

    def updateSingleClickPhotometry(self):
        self.single_click_photometry.setChecked(self.gui.single_click_photometry)

    def updateShowSatTracks(self):
        self.sat_tracks.setChecked(self.gui.show_sattracks)
        # Enable/disable auto-compute checkbox based on satellite tracks visibility
        self.auto_compute_sat_tracks.setEnabled(self.gui.show_sattracks)

    def updateAutoComputeSatTracks(self):
        self.auto_compute_sat_tracks.setChecked(self.gui.auto_compute_sattracks)

    def updateTLELabel(self, text):
        self.tle_label.setText("TLE: " + text)

    def updateImageGamma(self):
        self.img_gamma.setValue(self.gui.img.gamma)

    def updateLimMag(self):
        self.lim_mag.setValue(self.gui.cat_lim_mag)
        # Sync with Star Detection panel (if it exists - may not during init)
        if hasattr(self.gui, 'tab') and hasattr(self.gui.tab, 'star_detection'):
            self.gui.tab.star_detection.setCatalogLM(self.gui.cat_lim_mag)

    def updateApparentMagCorr(self):
        self.apparent_mag_corr.setChecked(self.gui.apparent_mag_corr_enabled)

    def onGammaChanged(self):
        gamma_value = self.img_gamma.value()
        self.gui.img.setGamma(gamma_value)
        self.gui.img_zoom.setGamma(gamma_value)
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

        # Sync with Star Detection panel
        self.gui.tab.star_detection.setCatalogLM(self.gui.cat_lim_mag)

    def populateCatalogList(self):
        """Populate the catalog combo box with available catalogs."""
        import os
        self.catalog_combo.blockSignals(True)
        self.catalog_combo.clear()

        # Add "Config Default" option
        self.catalog_combo.addItem("(Config Default)")

        # Find available catalogs
        catalog_path = self.gui.config.star_catalog_path
        if not os.path.isdir(catalog_path):
            catalog_path = os.path.join(self.gui.config.rms_root_dir, 'Catalogs')

        if os.path.isdir(catalog_path):
            for f in sorted(os.listdir(catalog_path)):
                if f.endswith('.npy') or f.endswith('.bin'):
                    self.catalog_combo.addItem(f)

        # Set current selection to match config
        current_catalog = self.gui.config.star_catalog_file
        index = self.catalog_combo.findText(current_catalog)
        if index >= 0:
            self.catalog_combo.setCurrentIndex(index)

        self.catalog_combo.blockSignals(False)

    def onCatalogChanged(self, catalog_name):
        """Handle catalog selection change."""
        if catalog_name == "(Config Default)":
            # Reset to config default - reload config value
            self.gui.config.star_catalog_file = self.gui._original_catalog_file
        else:
            self.gui.config.star_catalog_file = catalog_name

        # Restore the config's original band ratios so the magnitude filter
        # uses the same bands as on startup. Both GMN (.bin) and Sky2000 (.npy)
        # catalogs handle these ratios correctly internally.
        self.gui.config.star_catalog_band_ratios = self.gui._original_band_ratios

        # Reload catalog stars with new catalog
        self.gui.catalog_stars = self.gui.loadCatalogStars(self.gui.cat_lim_mag)
        self.gui.updateLeftLabels()
        self.gui.updateStars()

        self.sigCatalogChanged.emit(catalog_name)

    def updateCatalogSelection(self):
        """Update the combo box to match the current catalog."""
        self.catalog_combo.blockSignals(True)
        current_catalog = self.gui.config.star_catalog_file
        index = self.catalog_combo.findText(current_catalog)
        if index >= 0:
            self.catalog_combo.setCurrentIndex(index)
        else:
            self.catalog_combo.setCurrentIndex(0)  # Config Default
        self.catalog_combo.blockSignals(False)

    def onStdChanged(self):
        self.gui.stdev_text_filter = self.std.value()
        self.gui.photometry()

    def onSkyFit(self):
        self.lim_mag.show()
        self.lim_mag_label.show()
        self.apparent_mag_corr.show()
        self.std.show()
        self.std_label.show()
        self.detected_stars.show()
        self.distortion.show()
        self.error_overlay_chk.show()
        self.error_overlay_slider.show()
        self.error_overlay_value_label.show()
        self.selected_stars.show()
        self.picks.hide()
        self.great_circle.hide()
        self.region.hide()
        self.single_click_photometry.hide()
        self.meas_ground_points.hide()

        # Always show satellite options in SkyFit mode
        self.sat_tracks.show()
        self.tle_label.show()
        self.load_tle_btn.show()
        self.clear_tle_btn.show()
        self.redraw_tracks_btn.show()

        self.gui.selected_stars_visible = False
        self.sigSelStarsToggled.emit()  # toggle makes it true
        self.updateShowSelStars()

        self.gui.draw_calstars = False
        self.sigCalStarsToggled.emit()  # toggle makes it true
        self.updateShowCalStars()

    def onManualReduction(self):
        self.lim_mag.hide()
        self.lim_mag_label.hide()
        self.apparent_mag_corr.hide()
        self.std.hide()
        self.std_label.hide()
        self.detected_stars.hide()
        self.distortion.hide()
        self.error_overlay_chk.hide()
        self.error_overlay_slider.hide()
        self.error_overlay_value_label.hide()
        self.selected_stars.hide()
        self.picks.show()
        self.great_circle.show()
        self.region.show()
        self.single_click_photometry.show()

        self.meas_ground_points.show()

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
        if (e.key() == QtCore.Qt.Key_Return) or (e.key() == QtCore.Qt.Key_Enter):
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
        # Get current value and add step
        current_value = self.value()
        new_value = current_value + steps * self.singleStep()
        # Use format_float to maintain consistent display
        self.lineEdit().setText(format_float(new_value))

        self.buttonPressed.emit()
        self.valueModified.emit()

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        if e.key() == QtCore.Qt.Key_Return:
            self.valueModified.emit()


def format_float(value):
    """Format float to match platepar format - fixed decimal for larger values, scientific for small."""
    if abs(value) > 1e-4 or value == 0:
        return "{:+.6f}".format(value)
    else:
        return "{:+.3e}".format(value)
