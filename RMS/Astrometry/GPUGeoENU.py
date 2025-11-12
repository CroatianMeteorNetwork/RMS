"""
GPU-accelerated Geo to ENU conversion using Numba CUDA.

Much simpler than ENHt conversion - no iteration, just direct coordinate transforms.
Expected speedup: 30-60x for large arrays.
"""

import numpy as np
import math

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


@cuda.jit
def GeoToENU_kernel(lat_geo_deg, lon_geo_deg, h_geo_m,
                    E_out, N_out, U_out,
                    lat_sta_rad, lon_sta_rad, h_sta_m,
                    n_points):
    """
    CUDA kernel for Geodetic to ENU conversion.

    Each GPU thread processes one point independently.
    Much simpler than ENHt - no iteration, just direct transforms.
    """
    # Get thread index
    idx = cuda.grid(1)

    if idx >= n_points:
        return

    # WGS-84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    # Station ECEF coordinates (computed once per thread, but same for all)
    latS = lat_sta_rad
    lonS = lon_sta_rad
    sS = math.sin(latS)
    cS = math.cos(latS)
    Nsta = a / math.sqrt(1.0 - e2 * sS * sS)
    Xc = (Nsta + h_sta_m) * cS * math.cos(lonS)
    Yc = (Nsta + h_sta_m) * cS * math.sin(lonS)
    Zc = (Nsta * (1.0 - e2) + h_sta_m) * sS

    # ECEF <- ENU rotation matrix columns
    RE0 = -math.sin(lonS)
    RE1 = math.cos(lonS)
    RE2 = 0.0
    RN0 = -sS * math.cos(lonS)
    RN1 = -sS * math.sin(lonS)
    RN2 = cS
    RU0 = cS * math.cos(lonS)
    RU1 = cS * math.sin(lonS)
    RU2 = sS

    # Get this point's geodetic coordinates
    latT_deg = lat_geo_deg[idx]
    lonT_deg = lon_geo_deg[idx]
    hT = h_geo_m[idx]

    # Convert target geodetic to ECEF
    latT = math.radians(latT_deg)
    lonT = math.radians(lonT_deg)
    sT = math.sin(latT)
    cT = math.cos(latT)
    NT = a / math.sqrt(1.0 - e2 * sT * sT)
    Xt = (NT + hT) * cT * math.cos(lonT)
    Yt = (NT + hT) * cT * math.sin(lonT)
    Zt = (NT * (1.0 - e2) + hT) * sT

    # ECEF difference
    dX = Xt - Xc
    dY = Yt - Yc
    dZ = Zt - Zc

    # Rotate to ENU (R^T * (ECEF - C))
    E_out[idx] = RE0 * dX + RE1 * dY + RE2 * dZ
    N_out[idx] = RN0 * dX + RN1 * dY + RN2 * dZ
    U_out[idx] = RU0 * dX + RU1 * dY + RU2 * dZ


def geoToENUPP_gpu(lat_geo_deg, lon_geo_deg, h_geo_m, platepar,
                   threads_per_block=256):
    """
    GPU-accelerated version of geoToENUPP.

    Arguments:
        lat_geo_deg: [float or ndarray] Target geodetic latitude(s) in degrees
        lon_geo_deg: [float or ndarray] Target geodetic longitude(s) in degrees
        h_geo_m: [float or ndarray] Target WGS-84 ellipsoid height(s) in meters
        platepar: Platepar object with station coordinates
        threads_per_block: CUDA threads per block (default: 256)

    Returns:
        tuple: (E, N, U) East, North, Up coordinates in meters

    Raises:
        ImportError: If Numba CUDA not available
        RuntimeError: If no CUDA GPU detected
    """
    if not CUDA_AVAILABLE:
        raise ImportError("Numba CUDA not available. Install with: pip install numba")

    # Check for GPU
    try:
        cuda.select_device(0)
    except cuda.cudadrv.error.CudaSupportError:
        raise RuntimeError("No CUDA GPU detected. GPU acceleration not available.")

    # Convert to arrays
    lat_array = np.atleast_1d(np.array(lat_geo_deg, dtype=np.float64))
    lon_array = np.atleast_1d(np.array(lon_geo_deg, dtype=np.float64))
    h_array = np.atleast_1d(np.array(h_geo_m, dtype=np.float64))

    n_points = len(lat_array)

    # Check array lengths match
    if len(lon_array) != n_points or len(h_array) != n_points:
        raise ValueError("lat, lon, and h arrays must have the same length")

    # Allocate output arrays
    E_array = np.zeros(n_points, dtype=np.float64)
    N_array = np.zeros(n_points, dtype=np.float64)
    U_array = np.zeros(n_points, dtype=np.float64)

    # Station coordinates in radians
    lat_sta_rad = math.radians(platepar.lat)
    lon_sta_rad = math.radians(platepar.lon)
    h_sta_m = platepar.height_wgs84

    # Copy data to GPU
    lat_gpu = cuda.to_device(lat_array)
    lon_gpu = cuda.to_device(lon_array)
    h_gpu = cuda.to_device(h_array)
    E_gpu = cuda.to_device(E_array)
    N_gpu = cuda.to_device(N_array)
    U_gpu = cuda.to_device(U_array)

    # Calculate grid dimensions
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

    # Launch kernel
    GeoToENU_kernel[blocks_per_grid, threads_per_block](
        lat_gpu, lon_gpu, h_gpu,
        E_gpu, N_gpu, U_gpu,
        lat_sta_rad, lon_sta_rad, h_sta_m,
        n_points
    )

    # Copy results back
    E_array = E_gpu.copy_to_host()
    N_array = N_gpu.copy_to_host()
    U_array = U_gpu.copy_to_host()

    # Return scalars if input was scalar
    if np.isscalar(lat_geo_deg) and np.isscalar(lon_geo_deg) and np.isscalar(h_geo_m):
        return E_array[0], N_array[0], U_array[0]

    return E_array, N_array, U_array


class GPUGeoENUConverter:
    """
    GPU-accelerated Geo to ENU converter with persistent GPU context.

    Reuses GPU memory allocations for multiple conversions.
    """

    def __init__(self, platepar, threads_per_block=256):
        """
        Initialize GPU converter.

        Arguments:
            platepar: Platepar object with station coordinates
            threads_per_block: CUDA threads per block
        """
        if not CUDA_AVAILABLE:
            raise ImportError("Numba CUDA not available")

        self.platepar = platepar
        self.threads_per_block = threads_per_block

        # Pre-compute station parameters in radians
        self.lat_sta_rad = math.radians(platepar.lat)
        self.lon_sta_rad = math.radians(platepar.lon)
        self.h_sta_m = platepar.height_wgs84

        # Cache for GPU arrays
        self._cached_size = None
        self._lat_gpu = None
        self._lon_gpu = None
        self._h_gpu = None
        self._E_gpu = None
        self._N_gpu = None
        self._U_gpu = None

    def convert(self, lat_geo_deg, lon_geo_deg, h_geo_m):
        """
        Convert geodetic to ENU coordinates using GPU.

        Arguments:
            lat_geo_deg, lon_geo_deg, h_geo_m: Input coordinates

        Returns:
            tuple: (E, N, U)
        """
        # Convert to arrays
        lat_array = np.atleast_1d(np.array(lat_geo_deg, dtype=np.float64))
        lon_array = np.atleast_1d(np.array(lon_geo_deg, dtype=np.float64))
        h_array = np.atleast_1d(np.array(h_geo_m, dtype=np.float64))

        n_points = len(lat_array)

        # Allocate or reuse GPU memory
        if self._cached_size != n_points:
            self._lat_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._lon_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._h_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._E_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._N_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._U_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._cached_size = n_points

        # Copy data to GPU
        self._lat_gpu.copy_to_device(lat_array)
        self._lon_gpu.copy_to_device(lon_array)
        self._h_gpu.copy_to_device(h_array)

        # Calculate grid dimensions
        blocks_per_grid = (n_points + self.threads_per_block - 1) // self.threads_per_block

        # Launch kernel
        GeoToENU_kernel[blocks_per_grid, self.threads_per_block](
            self._lat_gpu, self._lon_gpu, self._h_gpu,
            self._E_gpu, self._N_gpu, self._U_gpu,
            self.lat_sta_rad, self.lon_sta_rad, self.h_sta_m,
            n_points
        )

        # Copy results back
        E_array = self._E_gpu.copy_to_host()
        N_array = self._N_gpu.copy_to_host()
        U_array = self._U_gpu.copy_to_host()

        # Return scalars if input was scalar
        if np.isscalar(lat_geo_deg) and np.isscalar(lon_geo_deg) and np.isscalar(h_geo_m):
            return E_array[0], N_array[0], U_array[0]

        return E_array, N_array, U_array


def benchmark_gpu_vs_cpu(n_points=1000000):
    """
    Quick benchmark comparing GPU vs CPU performance for Geo to ENU.
    """
    import time
    from RMS.Astrometry.ApplyAstrometry import geoToENUPP
    from RMS.Formats.Platepar import Platepar

    if not CUDA_AVAILABLE:
        print("CUDA not available - cannot run benchmark")
        return

    print(f"Benchmarking Geo to ENU: GPU vs CPU with {n_points:,} points")
    print("=" * 80)

    # Create test data
    pp = Platepar()
    pp.lat = 43.6532
    pp.lon = -79.3832
    pp.height_wgs84 = 200.0

    np.random.seed(42)
    # Generate realistic lat/lon around station (±2 degrees)
    lat = np.random.uniform(pp.lat - 2, pp.lat + 2, n_points)
    lon = np.random.uniform(pp.lon - 2, pp.lon + 2, n_points)
    h = np.random.uniform(7500, 15000, n_points)  # Contrail heights

    # CPU benchmark
    print("\nCPU (Cython):")
    start = time.time()
    E_cpu, N_cpu, U_cpu = geoToENUPP(lat, lon, h, pp)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.3f} s ({n_points/cpu_time:,.0f} pts/sec)")

    # GPU benchmark
    print("\nGPU (with data transfer):")
    start = time.time()
    E_gpu, N_gpu, U_gpu = geoToENUPP_gpu(lat, lon, h, pp)
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.3f} s ({n_points/gpu_time:,.0f} pts/sec)")

    # Speedup
    speedup = cpu_time / gpu_time
    print(f"\nSpeedup: {speedup:.1f}x")

    # Verify results match
    E_diff = np.abs(E_gpu - E_cpu)
    N_diff = np.abs(N_gpu - N_cpu)
    U_diff = np.abs(U_gpu - U_cpu)

    print(f"\nAccuracy (max difference from CPU):")
    print(f"  E: {np.max(E_diff):.9f} m")
    print(f"  N: {np.max(N_diff):.9f} m")
    print(f"  U: {np.max(U_diff):.9f} m")

    if np.max(E_diff) < 1e-6 and np.max(N_diff) < 1e-6 and np.max(U_diff) < 1e-6:
        print("  ✓ Results match CPU within tolerance")
    else:
        print("  ⚠ Results differ from CPU!")

    # Estimate 59M
    est_59M = (gpu_time / n_points) * 59000000
    print(f"\nEstimated time for 59M points: {est_59M:.2f} s")


if __name__ == "__main__":
    if CUDA_AVAILABLE:
        print("Numba CUDA is available")
        benchmark_gpu_vs_cpu(1000000)
    else:
        print("Numba CUDA not available")
        print("Install with: pip install numba")
