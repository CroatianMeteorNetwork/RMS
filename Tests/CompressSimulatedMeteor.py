""" Runs the compression on a simulated meteor. """


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


def meteorSimulate(img_w, img_h, frame_num, psf_sigma):

    frames = np.zeros((frame_num, img_h, img_w), np.uint8)

    # Get meteor intensitites
    frame_range = np.arange(0, 2*frame_num)
    meteor_intens = meteorIntensity(frame_range, 2*frame_num)


    # Calculate the slope of the line the meteor will follow
    slope = float(img_h)/img_w


    # Simulate even-first interlaced video
    for i in range(frame_num):

        for field in range(2):

            # Calculate the even-first half-frame
            half_frame = i + field/2.0

            # Get meteor intensity
            intens = meteor_intens[int(half_frame*2)]

            # Calculate meteor position
            x = (img_w/2)*float(i)/frame_num + img_w/4
            y = slope*x

            #print(x, y, intens)

            x_indices, y_indices = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))

            # Generate a Gaussian PSF
            gauss_values = twoDGaussian((x_indices, y_indices), intens, x, y, psf_sigma, psf_sigma, 0.0, 0.0)

            # Construct an image from the Gaussian values
            gauss_values = gauss_values.reshape(img_h, img_w)

            # On full frames, take only even fields
            if field == 0:
                frames[i, ::2, :] = gauss_values[::2, :]

            else:

                # Take odd fields
                frames[i, 1::2, :] = gauss_values[1::2, :]

        

        # Add gaussian noise to every frame
        frames[i] += np.abs(np.random.normal(0, 1, gauss_values.shape)).astype(np.uint8)
        frames[i] = np.clip(frames[i], 0, 255)



    return frames




if __name__ == "__main__":

    # Load config file
    config = cr.parse(".config")

    # Simulate a meteor
    frames = meteorSimulate(720, 480, 256, 2.0)


    # # Show individual frames
    # for i in range(20, 25):
    #     plt.imshow(frames[i])
    #     plt.show()

    
    comp = Compressor(None, None, None, None, None, config)

    # Run the compression
    compressed, field_intensities = comp.compress(frames)
    
    # Save the extracted intensitites per every field
    filename = FieldIntensities.saveFieldIntensitiesBin(field_intensities, '.', 'TEST')

    ### TEST ###
    FieldIntensities.convertFieldIntensityBinToTxt('.', filename)
    ############


    # Show compressed images
    plt.imshow(compressed[0])
    plt.show()

    # Show field intensitites
    plt.plot(np.arange(0, 2*256), field_intensities)
    plt.show()



