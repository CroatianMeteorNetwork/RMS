""" Definition of an FF file structure. This part is separated to avoid circual dependencies of modules. """

from __future__ import print_function, division, absolute_import

class FFStruct:
    """ Default structure for a FF file.
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