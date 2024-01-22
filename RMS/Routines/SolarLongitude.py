""" Functions for calculating solar longitude from Julian date and vice versa. """

from __future__ import absolute_import, division, print_function

import datetime

import numpy as np
import scipy.optimize
from RMS.Astrometry.Conversions import date2JD, datetime2JD, jd2Date

@np.vectorize
def jd2SolLonSteyaert(jd):
    """ Convert the given Julian date to solar longitude, J2000.0 epoch. Chris Steyaert method.

    Reference: Steyaert, C. (1991). Calculating the solar longitude 2000.0. WGN, Journal of the International 
        Meteor Organization, 19, 31-34.

    Arguments:
        jd: [float] julian date

    Return:
        [float] solar longitude in radians, J2000.0 epoch

    """

    # Define time constants
    A0 = [334166, 3489, 350, 342, 314, 268, 234, 132, 127, 120, 99, 90, 86, 78, 75, 51, 49, 36, 32, 28, 27, 
        24, 21, 21, 20, 16, 13, 13]

    B0 = [4.669257, 4.6261, 2.744, 2.829, 3.628, 4.418, 6.135, 0.742, 2.037, 1.110, 5.233, 2.045, 3.508, 
        1.179, 2.533, 4.58, 4.21, 2.92, 5.85, 1.90, 0.31, 0.34, 4.81, 1.87, 2.46, 0.83, 3.41, 1.08]

    C0 = [6283.07585, 12566.1517, 5753.385, 3.523, 77713.771, 7860.419, 3930.210, 11506.77, 529.691, 1577.344, 
        5884.927, 26.298, 398.149, 5223.694, 5507.553, 18849.23, 775.52, 0.07, 11790.63, 796.3, 10977.08, 
        5486.78, 2544.31, 5573.14, 6069.78, 213.3, 2942.46, 20.78]

    A1 = [20606, 430, 43]
    B1 = [2.67823, 2.635, 1.59]
    C1 = [6283.07585, 12566.152, 3.52]

    A2 = [872, 29]
    B2 = [1.073, 0.44]
    C2 = [6283.07585, 12566.15]

    A3 = 29
    B3 = 5.84
    C3 = 6283.07585

    # Number of millennia since 2000
    T = (jd - 2451545.0)/365250.0

    # Mean solar longitude
    L0 = 4.8950627 + 6283.07585*T - 0.0000099*T**2

    # Wrap L0 to [0, 2pi] range
    L0 = L0%(2*np.pi)

    # Periodical terms
    S0 = np.sum([A0[i]*np.cos((B0[i] + C0[i]*T)%(2*np.pi)) for i in range(28)])
    S1 = np.sum([A1[i]*np.cos((B1[i] + C1[i]*T)%(2*np.pi)) for i in range(3)])
    S2 = np.sum([A2[i]*np.cos((B2[i] + C2[i]*T)%(2*np.pi)) for i in range(2)])
    S3 = A3*np.cos((B3 + C3*T)%(2*np.pi))

    # Solar longitude of J2000.0
    L = L0 + (S0 + S1*T + S2*T**2 + S3*T**3)*1e-7

    # Bound to solar longitude to the [0, 2pi] range
    L = L%(2*np.pi)

    return L




def _solLon2jd(solFunc, year, month, L):
    """ Internal function. Numerically calculates the Julian date from the given solar longitude with the
        given method. The inverse precision is around 0.5 milliseconds.

        Because the solar longitudes around Dec 31 and Jan 1 can be ambigous, the month also has to be given.

    Arguments:
        solFunc: [function] Function which calculates solar longitudes from Julian dates.
        year: [int] Year of the event.
        month: [int] Month of the event.
        L: [float] Solar longitude (radians), J2000 epoch.

    Return:
        JD: [float] Julian date.

    """

    def _previousMonth(year, month):
        """ Internal function. Calculates the previous month. """

        dt = datetime.datetime(year, month, 1, 0, 0, 0)

        # Get some day in the next month
        next_month = dt.replace(day=1) - datetime.timedelta(days=4)

        return next_month.year, next_month.month


    def _nextMonth(year, month):
        """ Internal function. Calculates the next month. """

        dt = datetime.datetime(year, month, 1, 0, 0, 0)

        # Get some day in the next month
        next_month = dt.replace(day=28) + datetime.timedelta(days=4)

        return next_month.year, next_month.month


    # Calculate the upper and lower bounds for the Julian date using the given year
    prev_year, prev_month = _previousMonth(year, month)
    jd_min = date2JD(prev_year, prev_month, 1, 0, 0, 0)

    next_year, next_month = _nextMonth(year, month)
    jd_max = date2JD(next_year, next_month, 28, 23, 59, 59)

    # Function which returns the difference between the given JD and solar longitude that is being matched
    sol_res_func = lambda jd, sol_lon: (np.sin(sol_lon) - np.sin(solFunc(jd)))**2 + (np.cos(sol_lon) \
        - np.cos(solFunc(jd)))**2

    # Find the Julian date corresponding to the given solar longitude
    res = scipy.optimize.minimize(sol_res_func, x0=[(jd_min + jd_max)/2], args=(L), \
        bounds=[(jd_min, jd_max)], tol=1e-13)

    return res.x[0]




def solLon2jdSteyaert(*args):
    """ Convert the given solar longitude (J2000) to Julian date, J2000.0 epoch. Chris Steyaert method. 
    
    Supposing L is in the middle of the month, the year and month can be a bit over a month before or a
    bit over a month after L (year is paired with month properly) before jd will start being wrong. Within
    that it is correct to 0.5 ms
    
    Arguments:
        year: [int] Year of the event.
        month: [int] Month of the event.
        L: [float] Solar longitude (radians), J2000 epoch.

    Return:
        JD: [float] Julian date.

    """

    return _solLon2jd(jd2SolLonSteyaert, *args)

def unwrapSol(sol, first_sol, last_sol):
    """ Given a solar longitude (J2000), and the range that it must be in, the value is unwrapped so that
    any valid value of sol will keep increasing from first_sol until last_sol, even when it should otherwise
    wrap back to 0.
    
    Arguments:
        sol: [float or ndarray] Solar longitude (radians)
        first_sol: [float] Starting solar longitude (radians) which sol must be later than in time
        last_sol: [float] Ending solar longitude (radians) which sol must be be earlier in time
        
    Return:
        unwrapped_sol: [float]
    
    """
    val = (2*np.pi + first_sol + last_sol)/2
    ret = np.where(first_sol < last_sol, sol, (sol-val)%(2*np.pi) + val)
    if isinstance(sol, np.ndarray):
        return ret
    return ret.item()

if __name__ == "__main__":

    # ### Test all solar longitude functions and see the difference between the solar longitudes they return

    # year = 2012

    # for month in range(1, 13):

    #     for day in [1, 10, 20]:

    #         jd = date2JD(year, month, day, np.random.uniform(0, 24), np.random.uniform(0, 60), np.random.uniform(0, 60))

    #         #jd = date2JD(2011, 2, 4, 23, 20, 42.16)
    #         #jd = date2JD(2012, 12, 13, 8, 20, 33.07)
    #         #jd = date2JD(2012, 12, 13, 8, 21, 34.51)
    #         #jd = date2JD(2012, 12, 13, 8, 22, 20.10)
    #         #jd = date2JD(2012, 12, 13, 8, 24, 01.63)

    #         print('------------------------------------')
    #         print('JD: {:.12f}'.format(jd))

    #         print('Steyaert:', np.degrees(jd2SolLonSteyaert(jd)))


    #         # Solar longitude to Julian date

    #         jd_steyaert = solLon2jdSteyaert(year, month, jd2SolLonSteyaert(jd))
    #         print('JD inverse Steyaert: {:.12f} +/- {:.6f} s'.format(jd_steyaert, 24*60*60*abs(jd - jd_steyaert)))

    # ### ###

    print("Current UTC time is: {}".format(datetime.datetime.utcnow()))
    print("Current Julian date is: {:.12f}".format(datetime2JD(datetime.datetime.utcnow())))
    print("Current solar longitude: {:.6f} deg".format(np.degrees(jd2SolLonSteyaert(datetime2JD(datetime.datetime.utcnow())))))


    # Test inverse function
    sol_test = 140.626
    jd_test = solLon2jdSteyaert(2023, 8, np.radians(sol_test))
    dt_test = jd2Date(jd_test, dt_obj=True)
    print("Test solar longitude: {:.6f} deg".format(sol_test))
    print("Test Julian date: {:.12f}".format(jd_test))
    print("Test date: {}".format(dt_test))

    # Convert the date back to solar longitude
    jd_inv_test = datetime2JD(dt_test)
    sol_inv_test = np.degrees(jd2SolLonSteyaert(jd_inv_test))
    print("Test solar longitude (inverse): {:.6f} deg".format(sol_inv_test))
