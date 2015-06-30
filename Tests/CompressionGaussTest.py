from RMS.Compression import Compression
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    frames = np.empty((256, 576, 720), np.uint8)
    for i in range(256):
        frames[i] = np.random.normal(128, 2, (576, 720))
    
    comp = Compression(None, None, None, None, 000)
    compressed = comp.compress(frames)
    plt.hist(compressed[1].ravel(), 256, [0,256])
    plt.show()