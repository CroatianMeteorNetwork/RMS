""" Runs the compression on a simulated meteor. """

from __future__ import print_function, division, absolute_import

import sys
import os

import numpy as np
import matplotlib.pyplot as plt


from RMS.Formats import FieldIntensities
from RMS.ExtractStars import twoDGaussian

from RMS.Compression import Compressor
import RMS.ConfigReader as cr



def meteorIntensity(fr, frame_num):

    # Parabolic meteor intensity
    intens = frame_num - (1.0/frame_num)*(fr - frame_num/2)**2

    # Rescale intensities to 0-255 range
    intens -= np.min(intens)
    intens /= np.max(intens)
    intens *= 255.0

    return intens


def meteorSimulate(img_w, img_h, frame_num, psf_sigma, speed=1):

    frames = np.zeros((frame_num, img_h, img_w), np.float64)

    # Get meteor intensitites
    frame_range = np.arange(0, 2*frame_num)
    meteor_intens = meteorIntensity(frame_range, 2*frame_num)


    # Calculate the slope of the line the meteor will follow
    slope = float(img_h)/img_w


    x_indices, y_indices = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))

    # Simulate even-first interlaced video
    for i in range(frame_num):

        print('Frame', i)

        if i < int(frame_num/speed):
            for field in range(2):

                # Calculate the even-first half-frame
                half_frame = i + field/2.0

                # Get meteor intensity
                intens = meteor_intens[int(half_frame*2)]/4

                # The meteor is not a snapshot in time, but it is moving. Thus, simulate extra movement in 
                # every frame
                for j in range(-4, 5):

                    # Calculate meteor position
                    x = (img_w/2)*float((i + 0.125*j)*speed)/frame_num + img_w/4
                    y = slope*x

                    #print(x, y, intens)

                    # Generate a Gaussian PSF
                    gauss_values = twoDGaussian((x_indices, y_indices, 255), intens, x, y, psf_sigma, \
                        psf_sigma, 0.0, 0.0)

                    # Construct an image from the Gaussian values
                    gauss_values = gauss_values.reshape(img_h, img_w)

                    # On full frames, take only even fields
                    if field == 0:
                        frames[i, ::2, :] += gauss_values[::2, :]

                    else:

                        # Take odd fields
                        frames[i, 1::2, :] += gauss_values[1::2, :]
                        pass

        

        # Add gaussian noise to every frame
        frames[i] += np.abs(np.random.normal(0, 7, gauss_values.shape)).astype(np.uint8)

        # Add a constant level to every frame
        frames[i] += 30

        frames[i] = np.clip(frames[i], 0, 255)



    return frames.astype(np.uint8)




if __name__ == "__main__":

    import time
    import pickle

    if len(sys.argv) < 2:
        print('Usage: python -m Tests.CompressSimulatedMeteor /output/dir')

        sys.exit()


    # Read the argument as a path to the night directory
    dir_path = " ".join(sys.argv[1:])

    # Load config file
    config = cr.parse(".config")

    print('Simulating a meteor...')

    # Simulate a meteor

    # # Faster low-light fireball
    # frames = meteorSimulate(720, 576, 256, 2.0, speed=4)

    # Slow bright fireball
    frames = meteorSimulate(720, 576, 256, 5.0, speed=1)

    # # All white
    # frames = np.zeros((256, 576, 720), np.uint8) + 255



    pickle_file = 'compress_test_frames.pickle'

    ## SAVE the frames to disk
    with open(os.path.join(dir_path, pickle_file), 'wb') as f:
        pickle.dump(frames, f, protocol=2)
    ###

    # ## Load the frames from disk
    # with open(os.path.join(dir_path, pickle_file), 'r') as f:
    #     frames = pickle.load(f)
    # ###


    # # Show individual frames
    # for i in range(120, 128):
    #     plt.imshow(frames[i])
    #     plt.show()

    
    comp = Compressor(dir_path, None, None, None, None, config)

    print('Running compression...')
    t1 = time.time()

    # Run the compression
    compressed, field_intensities = comp.compress(frames)

    print('Time for compression', time.time() - t1)

    t1 = time.time()

    # Save FF file
    comp.saveFF(compressed, 0, 0)
    
    # Save the extracted intensitites per every field
    filename = FieldIntensities.saveFieldIntensitiesBin(field_intensities, dir_path, 'TEST')

    ### TEST ###
    FieldIntensities.convertFieldIntensityBinToTxt(dir_path, filename)
    ############


    # Read field intensitites from the written file
    half_frames, field_intensities = FieldIntensities.readFieldIntensitiesBin(dir_path, filename)


    # Show maxpixel
    plt.imshow(compressed[0], vmin=0, vmax=255, cmap='gray')
    plt.show()

    # Show avepixel
    plt.imshow(compressed[2], vmin=0, vmax=255, cmap='gray')
    plt.show()

    # Show field intensitites
    plt.plot(half_frames, field_intensities)
    plt.show()



