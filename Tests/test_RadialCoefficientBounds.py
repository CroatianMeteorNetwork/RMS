""" Tests that the astrometry functions read no distortion coefficient past the end of the polynomial.

cyraDecToXY and cyXYToRADec are compiled without bounds checking. Each coefficient array is put flush against
a page with no access, so a read past its last element faults instead of returning whatever follows it in
memory. The call runs in a child process, which a fault kills without taking the test run with it.
"""

from __future__ import print_function, division, absolute_import

import os
import subprocess
import sys

import pytest


CHILD = r'''
import ctypes, itertools, mmap
import numpy as np
from RMS.Astrometry import CyFunctions as cy
from RMS.Formats.Platepar import Platepar

page = mmap.PAGESIZE
libc = ctypes.CDLL(None)
ra, dec = np.array([10.0, 11.0]), np.array([-40.0, -41.0])
jd, xd, yd = np.full(2, 2461070.9), np.array([100.0, 200.0]), np.array([100.0, 200.0])

for dist_type in [d for d in Platepar().distortion_type_list if d.startswith("radial")]:
    for equal_aspect, force_centre, asymmetry_corr in itertools.product([False, True], repeat=3):

        platepar = Platepar()
        platepar.equal_aspect = equal_aspect
        platepar.force_distortion_centre = force_centre
        platepar.asymmetry_corr = asymmetry_corr
        platepar.setDistortionType(dist_type, reset_params=True)
        n = platepar.poly_length

        # Two pages, the second one with no access, and the coefficients ending exactly where it starts
        buf = mmap.mmap(-1, 2*page, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        assert libc.mprotect(ctypes.c_void_p(addr + page), page, 0) == 0
        coeffs = np.frombuffer(buf, dtype=np.float64, count=n, offset=page - 8*n)
        coeffs[:] = 0.001

        cy.cyraDecToXY(ra, dec, 2461070.9, -43.6, 172.1, 1280.0, 720.0, 0.0, 2461070.9, 10.5, -40.5, 0.0, 6.0,
            coeffs, coeffs.copy(), dist_type, True, equal_aspect, force_centre, asymmetry_corr)
        cy.cyXYToRADec(jd, xd, yd, -43.6, 172.1, 1280.0, 720.0, 0.0, 2461070.9, 10.5, -40.5, 0.0, 6.0,
            coeffs, coeffs.copy(), dist_type, True, equal_aspect, force_centre, asymmetry_corr)

        print(dist_type, equal_aspect, force_centre, asymmetry_corr, "ok", flush=True)
'''


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs mmap page protection")
def test_no_coefficient_is_read_past_the_end():
    """ radial3-odd read k2, one element past its only coefficient, k1, in both directions. """

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Run outside the repository: a directory named RMS in the working directory shadows the package
    child = subprocess.run([sys.executable, "-c", CHILD], cwd=os.path.expanduser("~"), capture_output=True,
        text=True, env=dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", "")))

    # A negative return code is the signal that killed the child; anything else is an ordinary error, an
    #   unbuilt extension for one
    assert child.returncode == 0, "{:s}\nLast combinations run:\n{:s}\n{:s}".format(
        "Killed by signal {:d}".format(-child.returncode) if child.returncode < 0 else "Failed",
        "\n".join(child.stdout.splitlines()[-3:]), "\n".join(child.stderr.splitlines()[-5:]))
