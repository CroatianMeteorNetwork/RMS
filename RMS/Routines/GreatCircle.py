""" Fitting a great circle to points in the Cartesian coordinates system. """

# The MIT License

# Copyright (c) 2017, Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.linalg
import scipy.optimize

from RMS.Math import polarToCartesian



def greatCirclePhase(theta, phi, theta0, phi0):
    """ Find the phase angle of the point closest to the given point on the great circle. 
    
    Arguments:
        theta: [float] Inclination of the point under consideration (radians).
        phi: [float] Nodal angle of the point (radians).
        theta0: [float] Inclination of the great circle (radians).
        phi0: [float] Nodal angle of the great circle (radians).

    Return:
        [float] Phase angle on the great circle of the point under consideration (radians).
    """

    def _pointDist(x):
        """ Calculates the Cartesian distance from a point defined in polar coordinates, and a point on
            a great circle. """
        
        # Convert the pick to Cartesian coordinates
        point = polarToCartesian(phi, theta)

        # Get the point on the great circle
        circle = greatCircle(x, theta0, phi0)

        # Return the distance from the pick to the great circle
        return np.sqrt((point[0] - circle[0])**2 + (point[1] - circle[1])**2 + (point[2] - circle[2])**2)

    # Find the phase angle on the great circle which corresponds to the pick
    res = scipy.optimize.minimize(_pointDist, 0)

    return res.x



def greatCircle(t, theta0, phi0):
    """ 
    Calculates the point on a great circle defined my theta0 and phi0 in Cartesian coordinates. 
    
    Sources:
        - http://demonstrations.wolfram.com/ParametricEquationOfACircleIn3D/

    Arguments:
        t: [float or 1D ndarray] phase angle of the point in the great circle
        theta0: [float] Inclination of the great circle (radians).
        phi0: [float] Nodal angle of the great circle (radians).
    Return:
        [tuple or 2D ndarray] a tuple of (X, Y, Z) coordinates in 3D space (becomes a 2D ndarray if the input
            parameter t is also a ndarray)
    """


    # Calculate individual cartesian components of the great circle points
    x = -np.cos(t)*np.sin(phi0) + np.sin(t)*np.cos(theta0)*np.cos(phi0)
    y =  np.cos(t)*np.cos(phi0) + np.sin(t)*np.cos(theta0)*np.sin(phi0)
    z =  np.sin(t)*np.sin(theta0)

    return x, y, z




def fitGreatCircle(x, y, z):
    """ Fits a great circle to points in 3D space. 

    Arguments:
        x: [float] X coordiantes of points on the great circle.
        y: [float] Y coordiantes of points on the great circle.
        z: [float] Z coordiantes of points on the great circle.

    Return: 
        X, theta0, phi0: [tuple of floats] Great circle parameters.
    """

    # Add (0, 0, 0) to the data, as the great circle should go through the origin
    x = np.append(x, 0)
    y = np.append(y, 0)
    z = np.append(z, 0)

    # Fit a linear plane through the data points
    A = np.c_[x, y, np.ones(x.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, z)

    # Calculate the great circle parameters
    z2 = C[0]**2 + C[1]**2

    theta0 = np.arcsin(z2/np.sqrt(z2 + z2**2))
    phi0 = np.arctan2(C[1], C[0])

    return C, theta0, phi0

    



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Generate random great circle data
    t_range = np.arange(0, np.random.rand()*2*np.pi, 0.05)
    #x_data, y_data, z_data = greatCircle(t_range, np.random.rand(), np.random.rand() + 1)
    x_data, y_data, z_data = greatCircle(t_range, np.radians(30), np.radians(55))

    # Fit the great circle
    C, theta0, phi0 = fitGreatCircle(x_data, y_data, z_data)

    # Make a grid of independant variables
    X,Y = np.meshgrid(np.arange(-1.0, 1.0, 0.1), np.arange(-1.0, 1.0, 0.1))

    # Generate the Z component of the plane
    Z = C[0]*X + C[1]*Y + C[2]

    # Prepare for 3D plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the original data points
    ax.scatter(x_data, y_data, z_data)

    # Plot the fitted plane
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.1)

    # Plot the fitted great circle
    t_array = np.arange(0, 2*np.pi, 0.01)
    ax.scatter(*greatCircle(t_array, theta0, phi0), c='b', s=3)

    # Plot the zero angle point of the great circle
    ax.scatter(*greatCircle(0, theta0, phi0), c='r', s=100)

    # Define plane normal
    N = greatCircle(np.pi/2.0, theta0+np.pi/2.0, phi0)

    # Plot the plane normal point
    ax.scatter(*N, c='g', s=100)

    # Plot a line from center to plane normal
    ax.plot([0, N[0]], [0, N[1]], [0, N[2]])

    print('Normal:', N)

    # Set the limits of the plot to unit size
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_zlim(-1, 1)

    ax.set_aspect('equal')

    plt.show()