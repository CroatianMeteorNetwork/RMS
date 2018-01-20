""" Definition of an FF file structure. This part is separated to avoid circual dependencies of modules. """

from __future__ import print_function, division, absolute_import

class FFStruct:
    """ Default structure for a FF file.
    """
    
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.nbits = 0
        self.first = 0
        self.camno = None

        self.nframes = -1
        self.fps = -1
        
        self.maxpixel = None
        self.maxframe = None
        self.avepixel = None
        self.stdpixel = None
        
        self.array = None


    def __repr__(self):

        out  = ''
        out += 'N rows: {:d}\n'.format(self.nrows)
        out += 'N cols: {:d}\n'.format(self.ncols)
        out += 'N bits: {:d}\n'.format(self.nbits)
        out += 'First frame: {:d}\n'.format(self.first)
        out += 'Camera ID: {:s}\n'.format(str(self.camno))

        out += 'N frames: {:d}\n'.format(self.nframes)
        out += 'FPS: {:d}\n'.format(self.fps)

        return out