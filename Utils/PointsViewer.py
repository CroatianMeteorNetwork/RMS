import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, os
from RMS.Formats import FFbin
from RMS import VideoExtraction
import RMS.ConfigReader as cr

def plot(points, x_dim, y_dim, name):
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
    config = cr.parse(".config")
    
    ve = VideoExtraction.Extractor(config)
    
    ff_list = [ff for ff in os.listdir(sys.argv[1]) if ff[0:2]=="FF" and ff[-3:]=="bin"]
    
    for ff_file in ff_list:
        ffbin = FFbin.read(sys.argv[1], ff_file, array=True)
        
        ve = VideoExtraction.Extractor(config)
        ve.frames = np.empty((256, ffbin.nrows, ffbin.ncols))
        ve.frames = np.empty((256, ffbin.nrows, ffbin.ncols))
        ve.compressed = ffbin.array
        
        points = np.array(ve.findPoints())
        
        plot(points, ffbin.ncols//config.f, ffbin.nrows//config.f, ff_file)