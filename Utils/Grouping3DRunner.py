
from RMS.VideoExtraction import Extractor
import RMS.ConfigReader as cr
import RMS.Formats.FFbin as FFbin
from RMS.Routines.Grouping3D import find3DLines, getAllPoints


import os
import sys
import time
import numpy as np
import scipy


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

if __name__ == "__main__":

    # Get bin path and name from the given argument
    bin_name = sys.argv[1]
    if os.sep in bin_name:
        bin_name = bin_name.split(os.sep)
        bin_dir = (os.sep).join(bin_name[:-1]) + os.sep
        bin_name = bin_name[-1]

    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/2015_06_20_19_33_11' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/2015_06_23_19_33_43' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/bins' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/perzeidi2015/11 12 Aug' + os.sep
    # bin_dir = "/home/anonymus/Dropbox/grouping3D/perzeidi2015/12 13 Aug" + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/2015110304 bolid' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/2015110506 bolid' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/20151111213 visnjan' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/RIB2015062627 fireball' + os.sep
    # bin_dir = '/home/anonymus/Dropbox/grouping3D/samples/sample_bins' + os.sep

    for bin_name in os.listdir(bin_dir):
        if 'FF' in bin_name:

            print bin_name

            # Load config file
            config = cr.parse(".config")

            # Load compressed file
            compressed = FFbin.read(bin_dir, bin_name, array=True).array

            # Show maxpixel
            ff = FFbin.read(bin_dir, bin_name)
            plt.imshow(ff.maxpixel, cmap='gray')
            plt.show()

            plt.clf()
            plt.close()

            # Dummy frames (empty)
            frames = np.zeros(shape=(256, compressed.shape[1], compressed.shape[2]), dtype=np.uint8)

            extract_obj = Extractor(config, bin_dir)
            extract_obj.compressed = compressed
            extract_obj.frames = frames

            event_points = extract_obj.findPoints()

            # # Produce fake event points
            # event_points = []
            # for i in range(50):
            #     j = 50 + i
            #     event_points.append([int(j/8)+1,int(j/8), i])
            #     event_points.append([int(j/8),int(j/8), i])

            #     j = 256 - i
            #     event_points.append([int(j/8),int(j/8), i+60])

            #     j = 256 - i - 50
            #     event_points.append([int(j/8),int(30-j/8), i+200])

            print event_points

            if event_points:

                # Plot lines in 3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Plot the image as a surface at frame 0
                img_y_size = ff.maxpixel.shape[0]/config.f
                img_x_size = ff.maxpixel.shape[1]/config.f
                y, x = np.mgrid[0:img_y_size, :img_x_size]
                img_resize = scipy.misc.imresize(ff.maxpixel, (img_y_size, img_x_size), interp='lanczos').astype(np.float64)
                ax.plot_surface(x, y, np.zeros_like(x), rstride=1, cstride=1, 
                    facecolors=cm.inferno(img_resize/np.max(img_resize)))

                points = np.array(event_points, dtype = np.uint8)

                xs = points[:,1]
                ys = points[:,0]
                zs = points[:,2]

                print len(xs)

                # Plot points in 3D
                ax.scatter(xs, ys, zs)

                # Set limits
                plt.xlim((0, compressed.shape[2]/config.f))
                plt.ylim((0, compressed.shape[1]/config.f))
                ax.set_zlim((0, 255))

                # Set labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Frame')

                t1 = time.clock()

                # Run line finding
                line_list = find3DLines(event_points, time.time(), config)

                elapsed_time = time.clock() - t1
                # print 'Elapsed time: ', time.clock() - t1

                if line_list:

                    print line_list

                    # Define line colors to use
                    ln_colors = ['r', 'g', 'y', 'k', 'm', 'c']

                    # Plot detected lines in 3D
                    for i, detected_line in enumerate(line_list):
                        # detected_line = detected_line[0]
                        xs = [detected_line[0][0], detected_line[1][0]]
                        ys = [detected_line[0][1], detected_line[1][1]]
                        zs = [detected_line[0][2], detected_line[1][2]]
                        ax.plot(ys, xs, zs, c = ln_colors[i%6])


                    # Plot grouped points
                    for i, detected_line in enumerate(line_list):

                        x1, x2 = detected_line[0][0], detected_line[1][0]

                        y1, y2 = detected_line[0][1], detected_line[1][1]

                        z1, z2 = detected_line[0][2], detected_line[1][2]

                        detected_points = getAllPoints(event_points, x1, y1, z1, x2, y2, z2, config)

                        if not detected_points.any():
                            continue

                        detected_points = np.array(detected_points)

                        xs = detected_points[:,1]
                        ys = detected_points[:,0]
                        zs = detected_points[:,2]

                        ax.scatter(xs, ys, zs, c = ln_colors[i%6], s = 40)


                print 'Elapsed time: ', elapsed_time
                plt.show()

                plt.clf()
                plt.close()




