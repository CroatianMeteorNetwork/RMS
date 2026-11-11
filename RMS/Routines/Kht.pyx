# cython: language_level=3, boundscheck=False, wraparound=False
""" Cython wrapper around the Kernel-based Hough Transform (KHT) C++ library.

Previously the KHT C++ sources were compiled into a bare shared library that was
located at runtime by walking the file system (ConfigReader.findBinaryPath) and
loaded via ctypes. That approach has no ABI safety net: ctypes will happily load a
stale or wrong-version binary and only fail later with the cryptic
"undefined symbol: kht_wrapper".

Exposing KHT as a regular Cython extension module makes it behave like every other
native module in RMS: it is imported through the normal Python import machinery,
which matches the compiled binary to the running interpreter by its ABI tag. A stale
or mismatched build now raises a clear ImportError (and gets rebuilt) instead of
silently loading and crashing at call time.
"""

from libc.stddef cimport size_t


# Declare the C entry point (Native/Hough/kht.cpp) and a small C shim that casts the
# contiguous output buffer to the array-pointer type the C function expects. Doing the
# cast in C keeps the Cython side free of the awkward `double (*)[2]` pointer type.
cdef extern from * nogil:
    """
    extern "C" size_t kht_wrapper(double (*)[2], unsigned char *, const size_t,
                                  const size_t, const size_t, const size_t, const double,
                                  const double, const double, const double);

    static inline size_t kht_call(double *lines_array, unsigned char *binary_image,
                                  size_t image_width, size_t image_height, size_t lines_max,
                                  size_t cluster_min_size, double cluster_min_deviation,
                                  double delta, double kernel_min_height, double n_sigmas) {
        return kht_wrapper((double (*)[2]) lines_array, binary_image, image_width,
                           image_height, lines_max, cluster_min_size, cluster_min_deviation,
                           delta, kernel_min_height, n_sigmas);
    }
    """
    size_t kht_call(double *lines_array, unsigned char *binary_image,
                    size_t image_width, size_t image_height, size_t lines_max,
                    size_t cluster_min_size, double cluster_min_deviation, double delta,
                    double kernel_min_height, double n_sigmas) nogil


def khtLineDetection(double[:, ::1] lines_array, unsigned char[::1] binary_image,
                     size_t image_width, size_t image_height, size_t lines_max,
                     size_t cluster_min_size, double cluster_min_deviation, double delta,
                     double kernel_min_height, double n_sigmas):
    """ Run the kernel-based Hough transform line detection on a binary image.

    Arguments:
        lines_array: [ndarray] Preallocated C-contiguous (lines_max, 2) float64 output
            buffer. Filled in place with [rho, theta] for each detected line.
        binary_image: [ndarray] C-contiguous 1D uint8 flattened binary image (0 = black,
            1-255 = feature pixel).
        image_width: [int] Image width in pixels.
        image_height: [int] Image height in pixels.
        lines_max: [int] Maximum number of lines to return (size of lines_array).
        cluster_min_size: [int] Minimum number of pixels in a cluster.
        cluster_min_deviation: [float] Minimum accepted distance between a feature pixel
            and the line segment defined by its cluster's end points.
        delta: [float] Discretization step for the parameter space.
        kernel_min_height: [float] Minimum kernel height to pass culling ([0, 1] range).
        n_sigmas: [float] Number of standard deviations used by the Gaussian kernel.

    Return:
        n_lines: [int] Number of detected lines written into lines_array.
    """

    cdef size_t n_lines

    # The KHT computation does not touch Python objects, so release the GIL (this also
    # matches the previous ctypes behaviour, which released the GIL during the call).
    with nogil:
        n_lines = kht_call(&lines_array[0, 0], &binary_image[0], image_width, image_height,
                           lines_max, cluster_min_size, cluster_min_deviation, delta,
                           kernel_min_height, n_sigmas)

    return n_lines
