import os

import RMS.ConfigReader as cr
from RMS.Formats import CALSTARS
from RMS.ExtractStars import extractStarsImgHandle
from RMS.Formats.FrameInterface import detectInputType


if __name__ == "__main__":
    
    import argparse

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="""Extract the stars from the image.""", formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("data_path", type=str, help="Path to the image data. Either a directory or a file.""")

    arg_parser.add_argument("--config", type=str, default=None, help="Path to the configuration file.")


    cml_args = arg_parser.parse_args()

    #########################

    # Extract the directory path even if it's a file
    if os.path.isdir(cml_args.data_path):
        dir_path = cml_args.data_path
    else:
        dir_path = os.path.dirname(cml_args.data_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Number of frames to chunk
    chunk_frames = 128

    # Detect the input type 
    #   (take 128 frames for making the FF files, applicable for videos and image sequences)
    img_handle = detectInputType(cml_args.data_path, config, use_fr_files=False, detection=True, 
                                 chunk_frames=chunk_frames)


    # Extract the stars on the image handle
    star_list = extractStarsImgHandle(img_handle, config=config)

    
    dir_name = os.path.basename(os.path.abspath(dir_path))
    if dir_name.startswith(config.stationID):
        prefix = dir_name
    else:
        prefix = "{:s}_{:s}".format(config.stationID, dir_name)

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS_' + prefix + '.txt'


    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, dir_path, calstars_name, 
                           config.stationID, config.height, config.width, chunk_frames=chunk_frames)
    
    print("Stars extracted and written to {:s}".format(calstars_name))


