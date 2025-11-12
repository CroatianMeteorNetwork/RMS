"""
Parallel processing wrapper for ENHt0ToENHt1 to handle massive point clouds.

For real-time processing of 59M+ points in contrail analysis.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import warnings


def ENHt0ToENHt1_parallel(E0_data, N0_data, Ht0_data, Ht1_data, platepar, n_workers=None, chunk_size=None):
    """
    Parallel version of ENHt0ToENHt1 for processing large point clouds.

    Arguments:
        E0_data: [ndarray or float] ENU east coordinate(s) at height Ht0 (meters).
        N0_data: [ndarray or float] ENU north coordinate(s) at height Ht0 (meters).
        Ht0_data: [ndarray or float] WGS-84 ellipsoidal height(s) of input points (meters).
        Ht1_data: [ndarray or float] WGS-84 ellipsoidal height(s) of output points (meters).
        platepar: [Platepar object] Platepar object with station coordinates.
        n_workers: [int] Number of parallel workers. Default: cpu_count() - 2
        chunk_size: [int] Size of chunks to process. Default: auto-calculated

    Returns:
        tuple: (E1, N1, U1) ENU coordinates at height Ht1
    """
    from RMS.Astrometry.ApplyAstrometry import ENHt0ToENHt1

    # Convert to arrays
    E0_array = np.array(E0_data, dtype=np.float64).ravel()
    N0_array = np.array(N0_data, dtype=np.float64).ravel()
    Ht0_array = np.array(Ht0_data, dtype=np.float64).ravel()
    Ht1_array = np.array(Ht1_data, dtype=np.float64).ravel()

    n_points = len(E0_array)

    # For small arrays, don't bother with parallel processing
    if n_points < 50000:
        return ENHt0ToENHt1(E0_array, N0_array, Ht0_array, Ht1_array, platepar)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)  # Leave 2 cores free

    # Determine chunk size
    if chunk_size is None:
        # Aim for ~100-500k points per chunk for good balance
        chunk_size = max(50000, min(500000, n_points // (n_workers * 2)))

    # Calculate actual number of chunks
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    # If very few chunks, limit workers
    n_workers = min(n_workers, n_chunks)

    # Create platepar dict for pickling (can't pickle Platepar object directly)
    pp_dict = {
        'lat': float(platepar.lat),
        'lon': float(platepar.lon),
        'height_wgs84': float(platepar.height_wgs84)
    }

    # Create chunks
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)

        chunks.append((
            E0_array[start_idx:end_idx],
            N0_array[start_idx:end_idx],
            Ht0_array[start_idx:end_idx],
            Ht1_array[start_idx:end_idx],
            pp_dict,
            i  # chunk index for reconstruction
        ))

    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(_process_chunk_worker, chunks)

    # Reconstruct results in order
    E1_list = []
    N1_list = []
    U1_list = []

    for E1_chunk, N1_chunk, U1_chunk, chunk_idx in results:
        E1_list.append(E1_chunk)
        N1_list.append(N1_chunk)
        U1_list.append(U1_chunk)

    # Concatenate
    E1 = np.concatenate(E1_list)
    N1 = np.concatenate(N1_list)
    U1 = np.concatenate(U1_list)

    return E1, N1, U1


def _process_chunk_worker(args):
    """
    Worker function for processing a chunk of data.
    Must be at module level for multiprocessing to pickle it.
    """
    from RMS.Astrometry.ApplyAstrometry import ENHt0ToENHt1
    from RMS.Formats.Platepar import Platepar

    E0_chunk, N0_chunk, Ht0_chunk, Ht1_chunk, pp_dict, chunk_idx = args

    # Reconstruct platepar
    pp = Platepar()
    pp.lat = pp_dict['lat']
    pp.lon = pp_dict['lon']
    pp.height_wgs84 = pp_dict['height_wgs84']

    # Process chunk
    E1, N1, U1 = ENHt0ToENHt1(E0_chunk, N0_chunk, Ht0_chunk, Ht1_chunk, pp)

    return E1, N1, U1, chunk_idx


class ENHtConverter:
    """
    Reusable converter object for processing multiple batches with the same platepar.
    Maintains worker pool for efficient batch processing.
    """

    def __init__(self, platepar, n_workers=None):
        """
        Initialize converter with platepar.

        Arguments:
            platepar: [Platepar object] Station coordinates
            n_workers: [int] Number of parallel workers
        """
        self.platepar = platepar
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 2)

        self.pp_dict = {
            'lat': float(platepar.lat),
            'lon': float(platepar.lon),
            'height_wgs84': float(platepar.height_wgs84)
        }

    def convert(self, E0_data, N0_data, Ht0_data, Ht1_data, parallel=True, chunk_size=None):
        """
        Convert ENHt coordinates.

        Arguments:
            E0_data, N0_data, Ht0_data, Ht1_data: Input coordinates
            parallel: [bool] Use parallel processing (default: True for large arrays)
            chunk_size: [int] Chunk size for parallel processing

        Returns:
            tuple: (E1, N1, U1)
        """
        from RMS.Astrometry.ApplyAstrometry import ENHt0ToENHt1

        E0_array = np.array(E0_data, dtype=np.float64).ravel()
        n_points = len(E0_array)

        # Auto-decide on parallel
        if parallel and n_points >= 50000:
            return ENHt0ToENHt1_parallel(
                E0_data, N0_data, Ht0_data, Ht1_data,
                self.platepar,
                n_workers=self.n_workers,
                chunk_size=chunk_size
            )
        else:
            return ENHt0ToENHt1(E0_data, N0_data, Ht0_data, Ht1_data, self.platepar)

    def convert_to_reference_height(self, E0_data, N0_data, Ht0_data, reference_height=10000.0, **kwargs):
        """
        Convenience method to convert all points to a single reference height.

        Arguments:
            E0_data, N0_data, Ht0_data: Input coordinates
            reference_height: [float] Target height in meters (default: 10km)

        Returns:
            tuple: (E1, N1, U1)
        """
        E0_array = np.array(E0_data, dtype=np.float64).ravel()
        Ht1_data = np.full_like(E0_array, reference_height)

        return self.convert(E0_data, N0_data, Ht0_data, Ht1_data, **kwargs)


# GPU-accelerated version (placeholder for future implementation)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def ENHt0ToENHt1_gpu(E0_data, N0_data, Ht0_data, Ht1_data, platepar):
    """
    GPU-accelerated version of ENHt0ToENHt1 using CuPy.

    WARNING: Not yet implemented! Returns NotImplementedError.

    For implementation, this would:
    1. Transfer data to GPU memory
    2. Run bisection iterations in parallel on GPU cores
    3. Transfer results back to CPU

    Expected speedup: 10-50x for large arrays (1M+ points)
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x (or appropriate CUDA version)")

    raise NotImplementedError("""
    GPU acceleration not yet implemented.

    To implement:
    1. Port cyENHt0ToENHt1 bisection algorithm to CuPy kernel
    2. Handle WGS-84 ellipsoid math on GPU
    3. Manage GPU memory for large point clouds

    Alternative: Use Numba CUDA for GPU kernels without requiring CuPy.

    For now, use ENHt0ToENHt1_parallel() for multi-core CPU acceleration.
    """)
