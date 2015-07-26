import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from RMS.Formats import FFbin
from RMS import VideoExtraction

def plot(points, x_dim, y_dim):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    
    x = points[1]
    y = points[0]
    z = points[2]
    
    # Plot points in 3D
    ax.scatter(x, y, z, s = 5)

    # Set axes limits
    ax.set_zlim(0, 255)
    plt.xlim([0,x_dim])
    plt.ylim([0,y_dim])
    plt.show()


if __name__ == "__main__":
    ff = FFbin.read(sys.argv[1], sys.argv[2], array=True)
    
    ve = VideoExtraction.Extractor()
    ve.frames = np.empty((256, ff.nrows, ff.ncols))
    ve.compressed = ff.array
    
    points = np.array(ve.findPoints())
    
    plot(points, ff.nrows, ff.ncols)