""" Definition of an FF file structure. This part is separated to avoid circular dependencies of modules. """

from __future__ import print_function, division, absolute_import

import numpy as np


# Image planes stored in an FF file, in file order
FF_PLANES = ('maxpixel', 'maxframe', 'avepixel', 'stdpixel')


def stdpixelFloat(ff):
    """ The per-pixel standard deviation in codes as float32: the full-precision stdpixel16 plane
        when the FF carries one, otherwise the legacy 8-bit plane (integer codes, floored at 1).
        Every consumer of the noise (thresholding, centroid weights, SNR) should go through this.
    """

    stdpixel16 = getattr(ff, 'stdpixel16', None)
    if stdpixel16 is not None:
        return np.asarray(stdpixel16).astype(np.float32)/np.float32(256.0)

    return ff.stdpixel


def selectPlanes(planes, array):
    """ Validate a plane selection for the FF readers.

    Arguments:
        planes: [iterable of str or None] Requested plane names, or None for all planes.
        array: [bool] True if the caller wants ff.array populated, which needs every plane.

    Return:
        (load_all, planes): [bool, frozenset] Whether every plane is loaded, and the set of planes to
            load when not.
    """

    if (planes is None) or array:
        return True, frozenset(FF_PLANES)

    planes = frozenset(planes)

    unknown = planes - frozenset(FF_PLANES)
    if unknown:
        raise ValueError("Unknown FF planes: {}. Valid planes: {}".format(
            sorted(unknown), list(FF_PLANES)))

    return False, planes


class FFStruct:
    """ Default structure for an FF file.
    """
    
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        
        # 2*nbits compressed frames (OLD format)
        self.nbits = 0

        # Number of compressed frames (NEW format)
        self.nframes = -1

        self.first = 0
        self.camno = 0

         # Decimation factor (NEW format)
        self.decimation_fact = 0

        # Interleave flag (0=prog, 1=even/odd, 2=odd/even) (NEW format)
        self.interleave_flag = 0

        self.fps = -1
        
        self.maxpixel = None
        self.maxframe = None
        self.avepixel = None
        self.stdpixel = None

        # Average pixel image at full precision, in 8.8 fixed point (uint16, units of 1/256 ADU).
        # None if the FF file only carries the 8-bit average. avepixel is its rounding to whole
        # ADU, (avepixel16 + 128) >> 8; FITS files store avepixel as the legacy plane and the
        # sub-ADU residual in an extra AVEFRAC HDU that older readers never look at
        self.avepixel16 = None

        # Standard deviation at full precision, 8.8 fixed point (uint16, units of 1/256 code),
        # floored at half a code. None if the file only carries the 8-bit stdpixel, which is its
        # rounding (stdpixel16 + 128) >> 8; FITS files store the fractional byte in an extra STDFRAC
        # HDU that older readers never look at
        self.stdpixel16 = None

        # Camera gamma used to average avepixel16 in the linear domain (the stored plane is
        # re-encoded, i.e. stays in the gamma-encoded domain). 1.0 means encoded-domain averaging
        self.avegamma = 1.0

        # Id of the camera response table used for the linear-domain averaging ('' = power law)
        self.averesp = ''

        # Optional camera SoC die temperature [degC] from the RMSP provenance SEI (a proxy for
        # housing/ambient temperature, NOT lens temperature). None when unknown (e.g. XM cameras)
        self.soctemp = None

        # Optional per-block photometric provenance from the RMSP SEI (None when unknown):
        # frame exposure [s] (block mean / min / max); sensor analog, sensor digital and ISP
        # digital gain (x, block mean); whether exposure and all gains were constant within the
        # block; encoder mean/max QP (codec-quality indicator); white balance R/B gains (colour
        # term); number of frames in the block that carried the SEI
        self.exptime = None
        self.expmin = None
        self.expmax = None
        self.again = None
        self.dgain = None
        self.ispdgain = None
        self.seistabl = None
        self.qpmean = None
        self.qpmax = None
        self.wbr = None
        self.wbb = None
        self.seinfrm = None

        # Timing provenance: 'sei' (camera integration-start, us-class) or 'legacy' (GStreamer
        # origin, ~30 ms-class); SEI-minus-legacy block-median offset [ms]; interpolated frames
        self.timesrc = None
        self.timeoffs = None
        self.timeinterp = None

        self.array = None


        # False if dark and flat weren't applied, True otherwise (False be default)
        self.calibrated = False


    def __repr__(self):

        out  = ''
        out += 'N rows: {:d}\n'.format(self.nrows)
        out += 'N cols: {:d}\n'.format(self.ncols)
        out += 'N bits: {:d}\n'.format(self.nbits)
        out += 'N frames: {:d}\n'.format(self.nframes)
        out += 'First frame: {:d}\n'.format(self.first)
        out += 'Camera ID: {:s}\n'.format(str(self.camno))
        out += 'Decimation factor: {:d}\n'.format(self.decimation_fact)
        out += 'Interleave flag: {:d}\n'.format(self.interleave_flag)
        out += 'FPS: {:.2f}\n'.format(self.fps)

        return out