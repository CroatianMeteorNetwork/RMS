""" Go through all directories and recompute the .json collection areas files."""

from __future__ import print_function, division, absolute_import

import os
import multiprocessing

from RMS.Routines.MaskImage import getMaskFile
from RMS.ConfigReader import loadConfigFromDirectory
from RMS.Formats import Platepar
from Utils.Flux import getCollectingArea, FluxConfig


def updateCollectionAreaNight(dir_path, flux_config):

    try:
        # Load the config file
        config = loadConfigFromDirectory(['.'], dir_path)

        # Load the platepar
        platepar = Platepar.Platepar()
        platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)

        # Load the mask
        mask = getMaskFile(dir_path, config)

        if mask is not None:
            
            # Check that the mask has the correct resolution
            mask.checkMask(platepar.X_res, platepar.Y_res)

        # Recompute the collecting area file
        getCollectingArea(dir_path, config, flux_config, platepar, mask, overwrite=True)

    except Exception as e:
        print("Error in", dir_path)
        print(e)



def recomputeCollectionAreas(root_dir_path, ncores=1):
    """ Go through all directories and recompute the .json collection areas files. """


    # Init the flux configuration
    flux_config = FluxConfig()


    dir_list = []

    print()
    print("Building the list of directories...")

    # Go through all directories in the STATIONID/NIGHT/ directory structure
    for station_id in sorted(os.listdir(root_dir_path)):
        
        # Check that the station ID is 6 characters long and the first two characters are letters
        if len(station_id) != 6 or not station_id[:2].isalpha():
            continue
        
        # Path to the station directory
        station_dir_path = os.path.join(root_dir_path, station_id)

        print("Station:", station_id)

        # Go through all nights
        for night in sorted(os.listdir(station_dir_path)):

            # Path to the night directory
            dir_path = os.path.join(station_dir_path, night)

            # Check that the night is a directory
            if not os.path.isdir(dir_path):
                continue


            dir_list.append(dir_path)
    

    print()
    print("Found", len(dir_list), "directories.")
    print("Running in parallel with", ncores, "cores.")


    if ncores == 1:
        
        for dir_path in dir_list:
                
            print()
            print("Processing:")
            print(dir_path)

            # Update the collection area
            updateCollectionAreaNight(dir_path, flux_config)

    else:

        # Run in parallel
        pool = multiprocessing.Pool(ncores)

        # Use tqdm to show a progress bar if available
        try:
            from tqdm import tqdm
            pool.imap_unordered(
                tqdm(updateCollectionAreaNight, total=len(dir_list)), 
                    [(dir_path, flux_config) for dir_path in dir_list]
            )
            
        except ImportError:
            pool.imap_unordered(updateCollectionAreaNight, [(dir_path, flux_config) for dir_path in dir_list])

        pool.close()
        pool.join()




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Go through all directories and recompute the .json collection areas files.")

    parser.add_argument("root_dir_path", type=str, help="Path to the root directory.")

    parser.add_argument("--ncores", type=int, default=1, help="Number of cores to use.")

    args = parser.parse_args()

    recomputeCollectionAreas(args.root_dir_path, ncores=args.ncores)



