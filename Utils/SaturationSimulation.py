""" Simulate saturation with a moving Gaussian. """


from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize


def addGaussian(frame, x, y, amp, sigma):

    # Compute the window size as 3 sigma
    window = 3*sigma


    # Compute the limits of the window for adding the Gaussian to the frame

    x_min = np.floor(x - window)
    if x_min < 0:
        x_min = 0

    x_max = np.ceil(x + window)
    if x_max > frame.shape[1] - 1:
        x_max = frame.shape[1] - 1

    y_min = np.floor(y - window)
    if y_min < 0:
        y_min = 0

    y_max = np.ceil(y + window)
    if y_max > frame.shape[0] - 1:
        y_max = frame.shape[0] - 1


    for xp in range(int(y_min), int(y_max)):
        for yp in range(int(x_min), int(x_max)):

            # Compute the squared radius from the centre of the gaussian
            r_sq = (xp - x)**2 + (yp - y)**2

            # Compute the value of the Gaussian
            val = amp*np.exp(-0.5*r_sq/(sigma**2))

            frame[yp, xp] += val

    return frame



def simulateSaturation(app_mag, photom_offset, bg_val, fps, ang_vel, gauss_sigma, steps, saturation_point):


    # Compute the border as 3 sigma + padding
    border = 3*gauss_sigma + 10

    # Estimate the image size
    track_length = int(ang_vel/fps + 2*border)

    # Init the frame buffer
    frame = np.zeros(shape=(track_length, track_length), dtype=np.float64)


    # Init evaluation plane
    X = np.linspace(0, track_length - 1, track_length)
    Y = np.linspace(0, track_length - 1, track_length)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y


    # Compute the log sum pixel
    lsp = app_mag - photom_offset

    # Compute the intensity sum from the magnitude
    intens_sum = 10**(-lsp/2.5)

    # Compute the Gaussian amplitude
    A = intens_sum/(2*np.pi*gauss_sigma**2)

    # Normalize the ampllitude to the number of steps
    A /= steps


    # Move and evaluate the Gaussian at every step
    for n in range(steps):

        # Compute the location of the Gaussian
        x = y = n/steps*ang_vel/fps + border

        frame = addGaussian(frame, x, y, A, gauss_sigma)


    # Compute the real magnitude
    lsp_unsaturated = -2.5*np.log10(np.sum(frame))
    mag_app_unsaturated = lsp_unsaturated + photom_offset


    #print('Unsaturated magnitude:', mag_app_unsaturated)

    # Add the background
    frame += bg_val


    # Clip the image
    frame = np.clip(frame, 0, saturation_point)


    # Compute the saturated magnitude
    lsp_saturated = -2.5*np.log10(np.sum(frame - bg_val))
    mag_app_saturated = lsp_saturated + photom_offset

    #print('Modelled saturated magnitude:', mag_app_saturated)


    # plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
    # plt.show()


    return mag_app_unsaturated, mag_app_saturated

        



def findUnsaturatedMagnitude(app_mag, photom_offset, bg_val, fps, ang_vel, gauss_sigma, steps=50, \
    saturation_point=255):
    
    def _costFunc(mag_app_unsaturated, params):

        mag_app = params[0]
        params = params[1:]

        # Compute the unsatured magnitude
        _, mag_app_saturated = simulateSaturation(mag_app_unsaturated, *params)

        # Compute the residuals
        return abs(mag_app - mag_app_saturated)



    # Numerically find the position angle
    res = scipy.optimize.minimize(_costFunc, [-1], args=([app_mag, photom_offset, bg_val, fps, ang_vel, \
        gauss_sigma, steps, saturation_point]), method='Nelder-Mead')


    return res.x[0]






if __name__ == "__main__":


    photom_offset = 10.7


    background_lvl = 60
    fps = 25
    ang_vel = 250 # px/s


    #print(findUnsaturatedMagnitude(-0.5, photom_offset, 50, 25, 100, 1))
    #print(simulateSaturation(-10.0, photom_offset, 50, 25, 100, 1, 100, 255))


    # Generate the range of apparent (possibly saturated) magnitudes
    app_mag_range = np.append(np.linspace(-0.5, 2, 1000), np.linspace(2, 6, 10))


    # Generate a range of gaussian stddevs
    gauss_stddevs = [1.5, 1.3, 1.15]



    for stddev in gauss_stddevs:

        unsaturated_mags = []

        for app_mag in app_mag_range:

          # Compute the unsaturated magnitude
          mag_app_unsaturated = findUnsaturatedMagnitude(app_mag, photom_offset, background_lvl, fps, \
            ang_vel, stddev)

          print(app_mag, mag_app_unsaturated)


          unsaturated_mags.append(mag_app_unsaturated)



        plt.plot(unsaturated_mags, app_mag_range, label='$\sigma = {:.2f}$'.format(stddev))


    plt.plot(app_mag_range, app_mag_range, color='k', linestyle='--', label='No saturation')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.axes().set_aspect('equal', 'box')

    plt.xlabel('Actual magnitude')
    plt.ylabel('Apparent magnitude')

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig('saturation_sim.png', dpi=300)

    plt.show()



