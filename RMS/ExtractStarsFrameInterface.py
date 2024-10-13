import os

import RMS.ConfigReader as cr
from RMS.Formats import CALSTARS
from RMS.ExtractStars import extractStarsImgHandle
from RMS.Formats.FrameInterface import detectInputType, detectInputTypeFile, checkIfVideoFile
from RMS.DetectionTools import loadImageCalibration


def extractStarsFrameInterface(img_handle, config, 
                               chunk_frames=128, 
                               flat_struct=None, dark=None, mask=None, 
                               save_calstars=True):
    """ Given an image handle, extract the stars from the image data.

    Arguments:
        img_handle: [ImageHandle] Image handle object.
        config: [Config] Configuration object.

    Keyword arguments:
        chunk_frames: [int] Number of frames to stacked image on which the stars will be extracted.
        flat_struct: [np.array] Flat field structure.
        dark: [np.array] Dark field structure.
        mask: [np.array] Mask structure.
        save_calstars: [bool] Flag to indicate if the CALSTARS file should be saved.

    Return:
        star_list: [list] List of stars detected in the image.
    """

    # Extract the stars on the image handle
    star_list = extractStarsImgHandle(img_handle, config=config, 
                                      flat_struct=flat_struct, dark=dark, mask=mask)
    
    if save_calstars:

        # Construct the name of the CALSTARS file by using the camera code and the time of the first frame
        timestamp = img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_%f")
        prefix = "{:s}_{:s}".format(config.stationID, timestamp)

        # Generate the name for the CALSTARS file
        calstars_name = 'CALSTARS_' + prefix + '.txt'

        # Write detected stars to the CALSTARS file
        CALSTARS.writeCALSTARS(star_list, img_handle.dir_path, calstars_name, 
                            config.stationID, config.height, config.width, chunk_frames=chunk_frames)
        
        print("Stars extracted and written to {:s}".format(calstars_name))

    return star_list

def extractStarsDetectFrameInterface(data_path, config, 
                                     chunk_frames=128, multivids=False, save_calstars=True):
    """ Extract the stars from the image data.
    
    Arguments:
        data_path: [str] Path to the image data. Either a directory or a file.
        config: [Config] Configuration object.

    Keyword arguments:
        chunk_frames: [int] Number of frames to stacked image on which the stars will be extracted.
        multivids: [bool] Flag to indicate that the data path is a directory containing multiple video files.
        save_calstars: [bool] Flag to indicate if the CALSTARS file should be saved.
    
    Return:
        star_list: [list] List of stars detected in the image.
    """

    # If the data path contrains multiple video files, find them and add them to the list for extraction
    if multivids:
        
        videos_to_process = []

        # Find all video files in the directory
        print()
        for file_name in sorted(os.listdir(data_path)):
            if checkIfVideoFile(file_name):
                print("Found video file: {:s}".format(file_name))
                videos_to_process.append(os.path.join(data_path, file_name))

        print()
        print("Total number of video files found: {:d}".format(len(videos_to_process)))
        print()

        print("Processing the video files...")

        # Extract the stars from each video file
        for video_file in videos_to_process:

            print("Processing video file: {:s}".format(video_file))

            # Load the video file
            img_handle = detectInputTypeFile(video_file, config, detection=True, chunk_frames=chunk_frames,
                                             preload_video=True)
            
            # Load mask, dark, flat
            mask, dark, flat_struct = loadImageCalibration(img_handle.dir_path, config, 
                dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap)
            
            # Extract the stars from the image handle
            star_list = extractStarsFrameInterface(img_handle, config, chunk_frames=chunk_frames,
                flat_struct=flat_struct, dark=dark, mask=mask, save_calstars=save_calstars)


    else:

        # Detect the input type 
        #   (take 128 frames for making the FF files, applicable for videos and image sequences)
        img_handle = detectInputType(data_path, config, use_fr_files=False, detection=True, 
                                    chunk_frames=chunk_frames)
        
        # Load mask, dark, flat
        mask, dark, flat_struct = loadImageCalibration(img_handle.dir_path, config, 
            dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap)
        
        # Extract the stars from the image handle
        star_list = extractStarsFrameInterface(img_handle, config, chunk_frames=chunk_frames,
            flat_struct=flat_struct, dark=dark, mask=mask, save_calstars=save_calstars)

    return star_list


if __name__ == "__main__":
    
    import argparse

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(
        description="""Extract stars from the given video or a directory with images.""", 
        formatter_class=argparse.RawTextHelpFormatter
        )

    arg_parser.add_argument("data_path", type=str, 
        help="Path to the image data. Either a directory or a file."""
        )

    arg_parser.add_argument("--config", type=str, default=None, 
        help="Path to the configuration file."
        )

    arg_parser.add_argument("--chunk_frames", type=int, default=128, 
        help="Number of frames to stacked image on which the stars will be extracted."
        )
    
    arg_parser.add_argument("--multivids", action="store_true",
        help="Flag to indicate that the data path is a directory containing multiple video files."
        )


    cml_args = arg_parser.parse_args()

    #########################

    # Extract the directory path even if it's a file
    if os.path.isdir(cml_args.data_path):
        dir_path = cml_args.data_path
    else:
        dir_path = os.path.dirname(cml_args.data_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Extract the stars
    extractStarsDetectFrameInterface(
        cml_args.data_path, config, 
        chunk_frames=cml_args.chunk_frames, multivids=cml_args.multivids
        )


