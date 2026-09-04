"""
GPU-accelerated ENHt0ToENHt1 using Numba CUDA.

This implements the same double-bisection algorithm as cyENHt0ToENHt1,
but runs on GPU with thousands of parallel threads.

Expected speedup: 20-50x for large arrays (1M+ points)
"""

import numpy as np
import math

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: Numba CUDA not available. GPU acceleration disabled.")
    print("Install with: pip install numba")


@cuda.jit
def ENHt_kernel(E0_m, N0_m, Ht0_m, Ht1_m,
                E1_out, N1_out, U1_out,
                lat_sta_rad, lon_sta_rad, h_sta_m,
                n_points):
    """
    CUDA kernel for ENHt0ToENHt1 conversion.

    Each GPU thread processes one point independently.
    """
    # Get thread index
    idx = cuda.grid(1)

    # Bounds check
    if idx >= n_points:
        return

    # WGS-84 constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)

    # Station ECEF coordinates
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

    # Get this point's data
    E0 = E0_m[idx]
    N0 = N0_m[idx]
    Ht0 = Ht0_m[idx]
    Ht1 = Ht1_m[idx]

    # Build base ECEF direction for this (E0, N0)
    dxe_base = RE0 * E0 + RN0 * N0
    dye_base = RE1 * E0 + RN1 * N0
    dze_base = RE2 * E0 + RN2 * N0

    # ========================================================================
    # Step 1: Find U0 that gives the input height Ht0
    # ========================================================================

    U_lo = -50000.0
    U_hi = 400000.0

    # Evaluate at U_lo
    Xi = Xc + dxe_base + RU0 * U_lo
    Yi = Yc + dye_base + RU1 * U_lo
    Zi = Zc + dze_base + RU2 * U_lo
    pval = math.sqrt(Xi * Xi + Yi * Yi)
    theta_b = math.atan2(Zi * a, pval * b)
    st = math.sin(theta_b)
    ct = math.cos(theta_b)
    latP = math.atan2(Zi + ep2 * b * st * st * st, pval - e2 * a * ct * ct * ct)
    Ncur = a / math.sqrt(1.0 - e2 * math.sin(latP) * math.sin(latP))
    hP = pval / math.cos(latP) - Ncur
    f_lo = hP - Ht0

    # Evaluate at U_hi
    Xi = Xc + dxe_base + RU0 * U_hi
    Yi = Yc + dye_base + RU1 * U_hi
    Zi = Zc + dze_base + RU2 * U_hi
    pval = math.sqrt(Xi * Xi + Yi * Yi)
    theta_b = math.atan2(Zi * a, pval * b)
    st = math.sin(theta_b)
    ct = math.cos(theta_b)
    latP = math.atan2(Zi + ep2 * b * st * st * st, pval - e2 * a * ct * ct * ct)
    Ncur = a / math.sqrt(1.0 - e2 * math.sin(latP) * math.sin(latP))
    hP = pval / math.cos(latP) - Ncur
    f_hi = hP - Ht0

    # Ensure bracketing (expand bounds if needed)
    for _ in range(8):
        if f_lo * f_hi <= 0.0:
            break

        if math.fabs(f_lo) < math.fabs(f_hi):
            U_lo -= 0.5 * (U_hi - U_lo)
        else:
            U_hi += 0.5 * (U_hi - U_lo)

        Xi = Xc + dxe_base + RU0 * U_hi
        Yi = Yc + dye_base + RU1 * U_hi
        Zi = Zc + dze_base + RU2 * U_hi
        pval = math.sqrt(Xi * Xi + Yi * Yi)
        theta_b = math.atan2(Zi * a, pval * b)
        st = math.sin(theta_b)
        ct = math.cos(theta_b)
        latP = math.atan2(Zi + ep2 * b * st * st * st, pval - e2 * a * ct * ct * ct)
        Ncur = a / math.sqrt(1.0 - e2 * math.sin(latP) * math.sin(latP))
        hP = pval / math.cos(latP) - Ncur
        f_hi = hP - Ht0

    # Bisection iterations to find U0
    for _ in range(20):
        U_mid = 0.5 * (U_lo + U_hi)
        Xi = Xc + dxe_base + RU0 * U_mid
        Yi = Yc + dye_base + RU1 * U_mid
        Zi = Zc + dze_base + RU2 * U_mid
        pval = math.sqrt(Xi * Xi + Yi * Yi)
        theta_b = math.atan2(Zi * a, pval * b)
        st = math.sin(theta_b)
        ct = math.cos(theta_b)
        latP = math.atan2(Zi + ep2 * b * st * st * st, pval - e2 * a * ct * ct * ct)
        Ncur = a / math.sqrt(1.0 - e2 * math.sin(latP) * math.sin(latP))
        hP = pval / math.cos(latP) - Ncur
        f_mid = hP - Ht0

        if f_lo * f_mid <= 0.0:
            U_hi = U_mid
            f_hi = f_mid
        else:
            U_lo = U_mid
            f_lo = f_mid

        if math.fabs(f_mid) < 1e-3:  # 1mm tolerance
            break

    U0 = 0.5 * (U_lo + U_hi)

    # ========================================================================
    # Step 2: Find position at height Ht1 along same line of sight
    # ========================================================================

    distance0 = math.sqrt(E0 * E0 + N0 * N0)

    if distance0 > 1e-10:
        # Direction cosines in ENU
        ray_length = math.sqrt(E0 * E0 + N0 * N0 + U0 * U0)
        dir_e = E0 / ray_length
        dir_n = N0 / ray_length
        dir_u = U0 / ray_length

        # Binary search along ray for point at height Ht1
        t_lo = 10.0
        t_hi = 1000000.0

        for _ in range(30):
            t_mid = 0.5 * (t_lo + t_hi)

            # ENU coordinates at distance t_mid along ray
            E1 = t_mid * dir_e
            N1 = t_mid * dir_n
            U1 = t_mid * dir_u

            # Convert to ECEF and find geodetic height
            Xi = Xc + RE0 * E1 + RN0 * N1 + RU0 * U1
            Yi = Yc + RE1 * E1 + RN1 * N1 + RU1 * U1
            Zi = Zc + RE2 * E1 + RN2 * N1 + RU2 * U1

            pval = math.sqrt(Xi * Xi + Yi * Yi)
            theta_b = math.atan2(Zi * a, pval * b)
            st = math.sin(theta_b)
            ct = math.cos(theta_b)
            latP = math.atan2(Zi + ep2 * b * st * st * st, pval - e2 * a * ct * ct * ct)
            Ncur = a / math.sqrt(1.0 - e2 * math.sin(latP) * math.sin(latP))
            hP = pval / math.cos(latP) - Ncur

            if hP < Ht1:
                t_lo = t_mid
            else:
                t_hi = t_mid

            if math.fabs(hP - Ht1) < 1e-3:  # 1mm tolerance
                break

        # Final position at height Ht1
        E1_out[idx] = t_mid * dir_e
        N1_out[idx] = t_mid * dir_n
        U1_out[idx] = t_mid * dir_u
    else:
        # Point directly above/below station
        E1_out[idx] = 0.0
        N1_out[idx] = 0.0
        U1_out[idx] = Ht1 - h_sta_m  # Approximate for vertical case


def ENHt0ToENHt1_gpu(E0_data, N0_data, Ht0_data, Ht1_data, platepar,
                     threads_per_block=256):
    """
    GPU-accelerated version of ENHt0ToENHt1.

    Arguments:
        E0_data, N0_data, Ht0_data, Ht1_data: Input coordinates (numpy arrays)
        platepar: Platepar object with station coordinates
        threads_per_block: CUDA threads per block (default: 256)

    Returns:
        tuple: (E1, N1, U1) as numpy arrays

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
    E0_array = np.array(E0_data, dtype=np.float64).ravel()
    N0_array = np.array(N0_data, dtype=np.float64).ravel()
    Ht0_array = np.array(Ht0_data, dtype=np.float64).ravel()
    Ht1_array = np.array(Ht1_data, dtype=np.float64).ravel()

    n_points = len(E0_array)

    # Check array lengths match
    if len(N0_array) != n_points or len(Ht0_array) != n_points or len(Ht1_array) != n_points:
        raise ValueError("E0, N0, Ht0, and Ht1 arrays must have the same length")

    # Allocate output arrays
    E1_array = np.zeros(n_points, dtype=np.float64)
    N1_array = np.zeros(n_points, dtype=np.float64)
    U1_array = np.zeros(n_points, dtype=np.float64)

    # Station coordinates in radians
    lat_sta_rad = math.radians(platepar.lat)
    lon_sta_rad = math.radians(platepar.lon)
    h_sta_m = platepar.height_wgs84

    # Copy data to GPU
    E0_gpu = cuda.to_device(E0_array)
    N0_gpu = cuda.to_device(N0_array)
    Ht0_gpu = cuda.to_device(Ht0_array)
    Ht1_gpu = cuda.to_device(Ht1_array)
    E1_gpu = cuda.to_device(E1_array)
    N1_gpu = cuda.to_device(N1_array)
    U1_gpu = cuda.to_device(U1_array)

    # Calculate grid dimensions
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

    # Launch kernel
    ENHt_kernel[blocks_per_grid, threads_per_block](
        E0_gpu, N0_gpu, Ht0_gpu, Ht1_gpu,
        E1_gpu, N1_gpu, U1_gpu,
        lat_sta_rad, lon_sta_rad, h_sta_m,
        n_points
    )

    # Copy results back from GPU
    E1_array = E1_gpu.copy_to_host()
    N1_array = N1_gpu.copy_to_host()
    U1_array = U1_gpu.copy_to_host()

    return E1_array, N1_array, U1_array


class GPUENHtConverter:
    """
    GPU-accelerated converter with persistent GPU context for batch processing.

    Reuses GPU memory allocations for multiple conversions with the same array size.
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
        self._E0_gpu = None
        self._N0_gpu = None
        self._Ht0_gpu = None
        self._Ht1_gpu = None
        self._E1_gpu = None
        self._N1_gpu = None
        self._U1_gpu = None

    def convert(self, E0_data, N0_data, Ht0_data, Ht1_data):
        """
        Convert ENHt coordinates using GPU.

        Arguments:
            E0_data, N0_data, Ht0_data, Ht1_data: Input coordinates

        Returns:
            tuple: (E1, N1, U1)
        """
        # Convert to arrays
        E0_array = np.array(E0_data, dtype=np.float64).ravel()
        N0_array = np.array(N0_data, dtype=np.float64).ravel()
        Ht0_array = np.array(Ht0_data, dtype=np.float64).ravel()
        Ht1_array = np.array(Ht1_data, dtype=np.float64).ravel()

        n_points = len(E0_array)

        # Allocate or reuse GPU memory
        if self._cached_size != n_points:
            # Allocate new GPU arrays
            self._E0_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._N0_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._Ht0_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._Ht1_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._E1_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._N1_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._U1_gpu = cuda.device_array(n_points, dtype=np.float64)
            self._cached_size = n_points

        # Copy data to GPU
        self._E0_gpu.copy_to_device(E0_array)
        self._N0_gpu.copy_to_device(N0_array)
        self._Ht0_gpu.copy_to_device(Ht0_array)
        self._Ht1_gpu.copy_to_device(Ht1_array)

        # Calculate grid dimensions
        blocks_per_grid = (n_points + self.threads_per_block - 1) // self.threads_per_block

        # Launch kernel
        ENHt_kernel[blocks_per_grid, self.threads_per_block](
            self._E0_gpu, self._N0_gpu, self._Ht0_gpu, self._Ht1_gpu,
            self._E1_gpu, self._N1_gpu, self._U1_gpu,
            self.lat_sta_rad, self.lon_sta_rad, self.h_sta_m,
            n_points
        )

        # Copy results back
        E1_array = self._E1_gpu.copy_to_host()
        N1_array = self._N1_gpu.copy_to_host()
        U1_array = self._U1_gpu.copy_to_host()

        return E1_array, N1_array, U1_array

    def convert_to_reference_height(self, E0_data, N0_data, Ht0_data, reference_height=10000.0):
        """
        Convert all points to a single reference height.

        Arguments:
            E0_data, N0_data, Ht0_data: Input coordinates
            reference_height: Target height in meters (default: 10km)

        Returns:
            tuple: (E1, N1, U1)
        """
        E0_array = np.array(E0_data, dtype=np.float64).ravel()
        Ht1_data = np.full_like(E0_array, reference_height)

        return self.convert(E0_data, N0_data, Ht0_data, Ht1_data)

    def __del__(self):
        """Clean up GPU memory."""
        # GPU memory will be freed automatically when objects are deleted
        pass


def benchmark_gpu_vs_cpu(n_points=1000000):
    """
    Quick benchmark comparing GPU vs CPU performance.

    Arguments:
        n_points: Number of points to test (default: 1M)
    """
    import time
    from RMS.Astrometry.ApplyAstrometry import ENHt0ToENHt1
    from RMS.Formats.Platepar import Platepar

    if not CUDA_AVAILABLE:
        print("CUDA not available - cannot run benchmark")
        return

    print(f"Benchmarking GPU vs CPU with {n_points:,} points")
    print("=" * 80)

    # Create test data
    pp = Platepar()
    pp.lat = 43.6532
    pp.lon = -79.3832
    pp.height_wgs84 = 200.0

    np.random.seed(42)
    E0 = np.random.uniform(-150000, 150000, n_points)
    N0 = np.random.uniform(-150000, 150000, n_points)
    Ht0 = np.random.uniform(7500, 15000, n_points)
    Ht1 = np.full(n_points, 10000.0)

    # CPU benchmark
    print("\nCPU (serial):")
    start = time.time()
    E1_cpu, N1_cpu, U1_cpu = ENHt0ToENHt1(E0, N0, Ht0, Ht1, pp)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.3f} s ({n_points/cpu_time:,.0f} pts/sec)")

    # GPU benchmark (includes data transfer)
    print("\nGPU (with data transfer):")
    start = time.time()
    E1_gpu, N1_gpu, U1_gpu = ENHt0ToENHt1_gpu(E0, N0, Ht0, Ht1, pp)
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.3f} s ({n_points/gpu_time:,.0f} pts/sec)")

    # Speedup
    speedup = cpu_time / gpu_time
    print(f"\nSpeedup: {speedup:.1f}x")

    # Verify results match
    E_diff = np.abs(E1_gpu - E1_cpu)
    N_diff = np.abs(N1_gpu - N1_cpu)
    U_diff = np.abs(U1_gpu - U1_cpu)

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
    print(f"\nEstimated time for 59M points: {est_59M:.1f} s")
    if est_59M < 10:
        print("  ✓ Real-time target achieved!")
    else:
        print(f"  ⚠ Needs {10/est_59M:.1f}x faster GPU or optimization")


if __name__ == "__main__":
    if CUDA_AVAILABLE:
        print("Numba CUDA is available")
        print(f"GPU devices: {cuda.gpus}")
        benchmark_gpu_vs_cpu(1000000)
    else:
        print("Numba CUDA not available")
        print("Install with: pip install numba")
