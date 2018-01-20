import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from RMS.Formats import FFfile
from RMS import VideoExtraction
import RMS.ConfigReader as cr

def view(ff):
    config = cr.parse(".config")
    
    ve = VideoExtraction.Extractor(config)
    ve.frames = np.empty((256, ff.nrows, ff.ncols))
    ve.compressed = ff.array
    
    points = np.array(ve.findPoints())
    
    plot(points, ff.nrows//config.f, ff.ncols//config.f)

def plot(points, y_dim, x_dim):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    plt.title(name)
    
    y = points[:,0]
    x = points[:,1]
    z = points[:,2]
    
    # Plot points in 3D
    ax.scatter(x, y, z)

    # Set axes limits
    ax.set_zlim(0, 255)
    plt.xlim([0, x_dim])
    plt.ylim([0, y_dim])
    
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_zlabel("Time")
    
    plt.show()


if __name__ == "__main__":
    ff = FFfile.read(sys.argv[1], sys.argv[2], array=True)
    
    view(ff)