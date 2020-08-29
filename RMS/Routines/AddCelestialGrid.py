""" Functions for adding equatorial grids to images, given the calibration. """

from __future__ import print_function, division, absolute_import

import numpy as np

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, computeFOVSize
from RMS.Astrometry.Conversions import jd2Date, apparentAltAz2TrueRADec, trueRaDec2ApparentAltAz
from RMS.Math import angularSeparation


def addEquatorialGrid(plt_handle, platepar, jd):
    """ Given the plot handle containing the image, the function plots an equatorial grid.
        Arguments:
            plt_handle: [pyplot instance]
            platepar: [Platepar object]
            jd: [float] Julian date of the image. 
        Return:
            plt_handle: [pyplot instance] Pyplot instance with the added grid.
    """


    # Estimate RA,dec of the centre of the FOV
    _, RA_c, dec_c, _ = xyToRaDecPP([jd2Date(jd)], [platepar.X_res/2], [platepar.Y_res/2], [1], 
        platepar, extinction_correction=False)

    RA_c = RA_c[0]
    dec_c = dec_c[0]

    # Compute FOV centre alt/az
    azim_centre, alt_centre = trueRaDec2ApparentAltAz(RA_c, dec_c, jd, platepar.lat, platepar.lon)

    # Compute FOV size
    fov_h, fov_v = computeFOVSize(platepar)
    fov_radius = np.hypot(*computeFOVSize(platepar))

    # Determine gridline frequency (double the gridlines if the number is < 4eN)
    grid_freq = 10**np.floor(np.log10(fov_radius))
    if 10**(np.log10(fov_radius) - np.floor(np.log10(fov_radius))) < 4:
        grid_freq /= 2

    # Set a maximum grid frequency of 15 deg
    if grid_freq > 15:
        grid_freq = 15


    # Grid plot density
    plot_dens = grid_freq/100

    # Compute the range of declinations to consider
    dec_min = platepar.dec_d - fov_radius/2
    if dec_min < -90:
        dec_min = -90

    dec_max = platepar.dec_d + fov_radius/2
    if dec_max > 90:
        dec_max = 90

    ra_grid_arr = np.arange(0, 360, grid_freq)
    dec_grid_arr = np.arange(-90, 90, grid_freq)

    # Filter out the dec grid for min/max declination
    dec_grid_arr = dec_grid_arr[(dec_grid_arr >= dec_min) & (dec_grid_arr <= dec_max)]

    # Plot the celestial parallel grid
    for dec_grid in dec_grid_arr:

        ra_grid_plot = np.arange(0, 360, plot_dens)
        dec_grid_plot = np.zeros_like(ra_grid_plot) + dec_grid

        # Compute alt/az
        az_grid_plot, alt_grid_plot = trueRaDec2ApparentAltAz(ra_grid_plot, dec_grid_plot, jd, platepar.lat, \
            platepar.lon)

        # Filter out points below the horizon  and outside the FOV
        filter_arr = (alt_grid_plot > 0) & (np.degrees(angularSeparation(np.radians(alt_centre), \
            np.radians(azim_centre), np.radians(alt_grid_plot), np.radians(az_grid_plot))) < fov_radius)
        ra_grid_plot = ra_grid_plot[filter_arr]
        dec_grid_plot = dec_grid_plot[filter_arr]


        # Find gaps in continuity and break up plotting individual lines
        gap_indices = np.argwhere(np.abs(ra_grid_plot[1:] - ra_grid_plot[:-1]) > fov_radius)
        if len(gap_indices):
            
            ra_grid_plot_list = []
            dec_grid_plot_list = []

            # Separate gridlines with large gaps
            prev_gap_indx = 0
            for entry in gap_indices:

                gap_indx = entry[0]

                ra_grid_plot_list.append(ra_grid_plot[prev_gap_indx:gap_indx + 1])
                dec_grid_plot_list.append(dec_grid_plot[prev_gap_indx:gap_indx + 1])

                prev_gap_indx = gap_indx

            # Add the last segment
            ra_grid_plot_list.append(ra_grid_plot[prev_gap_indx + 1:-1])
            dec_grid_plot_list.append(dec_grid_plot[prev_gap_indx + 1:-1])



        else:
            ra_grid_plot_list = [ra_grid_plot]
            dec_grid_plot_list = [dec_grid_plot]


        # Plot all grid segments
        for ra_grid_plot, dec_grid_plot in zip(ra_grid_plot_list, dec_grid_plot_list):

            # Compute image coordinates for every grid celestial parallel
            x_grid, y_grid = raDecToXYPP(ra_grid_plot, dec_grid_plot, jd, platepar)

            # Plot the grid
            plt_handle.plot(x_grid, y_grid, color='w', alpha=0.2, zorder=2, linewidth=0.5, linestyle='dotted')


    # Plot the celestial meridian grid
    for ra_grid in ra_grid_arr:

        dec_grid_plot = np.arange(-90, 90, plot_dens)
        ra_grid_plot = np.zeros_like(dec_grid_plot) + ra_grid

        # Filter out the dec grid
        filter_arr = (dec_grid_plot >= dec_min) & (dec_grid_plot <= dec_max)
        ra_grid_plot = ra_grid_plot[filter_arr]
        dec_grid_plot = dec_grid_plot[filter_arr]


        # Compute alt/az
        az_grid_plot, alt_grid_plot = trueRaDec2ApparentAltAz(ra_grid_plot, dec_grid_plot, jd, platepar.lat, \
            platepar.lon)

        # Filter out points below the horizon
        filter_arr = (alt_grid_plot > 0) & (np.degrees(angularSeparation(np.radians(alt_centre), \
            np.radians(azim_centre), np.radians(alt_grid_plot), np.radians(az_grid_plot))) < fov_radius)
        ra_grid_plot = ra_grid_plot[filter_arr]
        dec_grid_plot = dec_grid_plot[filter_arr]

        # Compute image coordinates for every grid celestial parallel
        x_grid, y_grid = raDecToXYPP(ra_grid_plot, dec_grid_plot, jd, platepar)

        # # Filter out everything outside the FOV
        # filter_arr = (x_grid >= 0) & (x_grid <= platepar.X_res) & (y_grid >= 0) & (y_grid <= platepar.Y_res)
        # x_grid = x_grid[filter_arr]
        # y_grid = y_grid[filter_arr]


        # Plot the grid
        plt_handle.plot(x_grid, y_grid, color='w', alpha=0.2, zorder=2, linewidth=0.5, linestyle='dotted')



    return plt_handle



def updateRaDecGrid(grid, platepar):
    """
    Updates the values of grid to form a right ascension and declination grid on a pyqtgraph plot.

    Arguments:
        grid: [pg.PlotCurveItem]
        platepar: [Platepar object]

    """
    # Estimate RA,dec of the centre of the FOV
    _, RA_c, dec_c, _ = xyToRaDecPP([jd2Date(platepar.JD)], [platepar.X_res/2], [platepar.Y_res/2], [1], \
                                    platepar, extinction_correction=False)
    RA_c = RA_c[0]
    dec_c = dec_c[0]

    azim_centre, alt_centre = trueRaDec2ApparentAltAz(RA_c, dec_c, platepar.JD, platepar.lat, platepar.lon)

    # Compute FOV size
    fov_radius = np.hypot(*computeFOVSize(platepar))

    # Determine gridline frequency (double the gridlines if the number is < 4eN)
    grid_freq = 10**np.floor(np.log10(fov_radius))
    if 10**(np.log10(fov_radius) - np.floor(np.log10(fov_radius))) < 4:
        grid_freq /= 2

    # Set a maximum grid frequency of 15 deg
    if grid_freq > 15:
        grid_freq = 15

    # Grid plot density
    plot_dens = grid_freq/100

    ra_grid_arr = np.arange(0, 360, grid_freq)
    dec_grid_arr = np.arange(-90, 90, grid_freq)

    x = []
    y = []
    cuts = []

    # Plot the celestial parallel grid (circles)
    for dec_grid in dec_grid_arr:
        ra_grid_plot = np.arange(0, 360, plot_dens)
        dec_grid_plot = np.zeros_like(ra_grid_plot) + dec_grid

        # Compute alt/az
        az_grid_plot, alt_grid_plot = trueRaDec2ApparentAltAz(ra_grid_plot, dec_grid_plot, platepar.JD, \
            platepar.lat, platepar.lon, platepar.refraction)

        # Filter out points below the horizon and outside the FOV
        filter_arr = (alt_grid_plot >= 0) & (np.degrees(angularSeparation(np.radians(alt_centre), \
            np.radians(azim_centre), np.radians(alt_grid_plot), np.radians(az_grid_plot))) <= fov_radius)

        ra_grid_plot = ra_grid_plot[filter_arr]
        dec_grid_plot = dec_grid_plot[filter_arr]

        # Compute image coordinates for every grid celestial parallel
        x_grid, y_grid = raDecToXYPP(ra_grid_plot, dec_grid_plot, platepar.JD, platepar)

        filter_arr = (x_grid >= 0) & (x_grid <= platepar.X_res) & (y_grid >= 0) & (y_grid <= platepar.Y_res)
        x_grid = x_grid[filter_arr]
        y_grid = y_grid[filter_arr]

        x.extend(x_grid)
        y.extend(y_grid)
        cuts.append(len(x) - 1)

    # Plot the celestial meridian grid (outward lines)
    for ra_grid in ra_grid_arr:
        dec_grid_plot = np.arange(-90, 90, plot_dens)  # how close to horizon
        ra_grid_plot = np.zeros_like(dec_grid_plot) + ra_grid

        # Compute alt/az
        az_grid_plot, alt_grid_plot = trueRaDec2ApparentAltAz(ra_grid_plot, dec_grid_plot, platepar.JD, platepar.lat,
                                                              platepar.lon, platepar.refraction)

        # Filter out points below the horizon
        filter_arr = (alt_grid_plot >= 0) & (np.degrees(angularSeparation(np.radians(alt_centre),
                                                              np.radians(azim_centre),
                                                              np.radians(alt_grid_plot),
                                                              np.radians(az_grid_plot))) <= fov_radius)
        ra_grid_plot = ra_grid_plot[filter_arr]
        dec_grid_plot = dec_grid_plot[filter_arr]

        # Compute image coordinates for every grid celestial parallel
        x_grid, y_grid = raDecToXYPP(ra_grid_plot, dec_grid_plot, platepar.JD, platepar)

        filter_arr = (x_grid >= 0) & (x_grid <= platepar.X_res) & (y_grid >= 0) & (y_grid <= platepar.Y_res)
        x_grid = x_grid[filter_arr]
        y_grid = y_grid[filter_arr]

        x.extend(x_grid)
        y.extend(y_grid)
        cuts.append(len(x) - 1)

    # horizon
    az_horiz_arr = np.arange(0, 360, plot_dens)
    alt_horiz_arr = np.zeros_like(az_horiz_arr)
    ra_horiz_plot, dec_horiz_plot = apparentAltAz2TrueRADec(az_horiz_arr, alt_horiz_arr, platepar.JD, platepar.lat,
                                                            platepar.lon, platepar.refraction)

    x_horiz, y_horiz = raDecToXYPP(ra_horiz_plot, dec_horiz_plot, platepar.JD, platepar)

    filter_arr = (x_horiz >= 0) & (x_horiz <= platepar.X_res) & (y_horiz >= 0) & (y_horiz <= platepar.Y_res)
    x_horiz = x_horiz[filter_arr]
    y_horiz = y_horiz[filter_arr]

    x.extend(x_horiz)
    y.extend(y_horiz)
    cuts.append(len(x) - 1)

    r = 15  # adjust this parameter if you see extraneous lines
    # disconnect lines that are distant (unfinished circles had straight lines completing them)
    for i in range(len(x) - 1):
        if (x[i] - x[i + 1])**2 + (y[i] - y[i + 1])**2 > r**2:
            cuts.append(i)

    # convert cuts into connect
    connect = np.full(len(x), 1)
    if len(connect) > 0:
        for i in cuts:
            connect[i] = 0

    grid.setData(x=x, y=y, connect=connect)


def updateAzAltGrid(grid, platepar):
    """
    Updates the values of grid to form an azimuth and altitude grid on a pyqtgraph plot.

    Arguments:
        grid: [pg.PlotCurveItem]
        platepar: [Platepar object]

    """

    # Compute FOV size
    fov_radius = np.hypot(*computeFOVSize(platepar))

    # Determine gridline frequency (double the gridlines if the number is < 4eN)
    grid_freq = 10**np.floor(np.log10(fov_radius))
    if 10**(np.log10(fov_radius) - np.floor(np.log10(fov_radius))) < 4:
        grid_freq /= 2

    # Set a maximum grid frequency of 15 deg
    if grid_freq > 15:
        grid_freq = 15

    # Grid plot density
    plot_dens = grid_freq/100

    az_grid_arr = np.arange(0, 90, grid_freq)
    alt_grid_arr = np.arange(0, 360, grid_freq)

    x = []
    y = []
    cuts = []

    # circles
    for az_grid in az_grid_arr:
        alt_grid_plot = np.arange(0, 360, plot_dens)  # how many degrees of circle
        az_grid_plot = np.zeros_like(alt_grid_plot) + az_grid

        # filter_arr = (np.degrees(angularSeparation(np.radians(alt_centre),
        #                                 np.radians(azim_centre),
        #                                 np.radians(alt_grid_plot),
        #                                 np.radians(az_grid_plot))) < fov_radius)
        #
        # alt_grid_plot = alt_grid_plot[filter_arr]
        # az_grid_plot = az_grid_plot[filter_arr]

        ra_grid_plot, dec_grid_plot = apparentAltAz2TrueRADec(alt_grid_plot, az_grid_plot, platepar.JD, \
            platepar.lat, platepar.lon, platepar.refraction)
        x_grid, y_grid = raDecToXYPP(ra_grid_plot, dec_grid_plot, platepar.JD, platepar)

        filter_arr = (x_grid >= 0) & (x_grid <= platepar.X_res) & (y_grid >= 0) & (y_grid <= platepar.Y_res)
        x_grid = x_grid[filter_arr]
        y_grid = y_grid[filter_arr]

        x.extend(x_grid)
        y.extend(y_grid)
        cuts.append(len(x) - 1)

    # Outward lines
    for alt_grid in alt_grid_arr:
        az_grid_plot = np.arange(0, 90, plot_dens)
        alt_grid_plot = np.zeros_like(az_grid_plot) + alt_grid

        # filter_arr = (np.degrees(angularSeparation(np.radians(alt_centre),
        #                                 np.radians(azim_centre),
        #                                 np.radians(alt_grid_plot),
        #                                 np.radians(az_grid_plot))) < fov_radius)
        #
        # alt_grid_plot = alt_grid_plot[filter_arr]
        # az_grid_plot = az_grid_plot[filter_arr]

        ra_grid_plot, dec_grid_plot = apparentAltAz2TrueRADec(alt_grid_plot, az_grid_plot, platepar.JD, 
            platepar.lat, platepar.lon, platepar.refraction)
        x_grid, y_grid = raDecToXYPP(ra_grid_plot, dec_grid_plot, platepar.JD, platepar)

        filter_arr = (x_grid >= 0) & (x_grid <= platepar.X_res) & (y_grid >= 0) & (y_grid <= platepar.Y_res)
        x_grid = x_grid[filter_arr]
        y_grid = y_grid[filter_arr]

        x.extend(x_grid)
        y.extend(y_grid)
        cuts.append(len(x) - 1)

    r = 50
    for i in range(len(x) - 1):
        if (x[i] - x[i + 1])**2 + (y[i] - y[i + 1])**2 > r**2:
            cuts.append(i)

    connect = np.full(len(x), 1)
    for i in cuts[:-1]:
        connect[i] = 0

    grid.setData(x=x, y=y, connect=connect)
