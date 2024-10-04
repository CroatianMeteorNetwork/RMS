from RMS.ExtractStars import extractStarsImgHandle
import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import detectInputType


if __name__ == "__main__":
    
    import argparse

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="""Extract the stars from the image.""", formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument("dir_path", type=str, help="Path to the image data. Either a directory or a file.""")

    arg_parser.add_argument("--config", type=str, default=None, help="Path to the configuration file.")


    cml_args = arg_parser.parse_args()

    #########################


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


    # Detect the input type
    img_handle = detectInputType(cml_args.dir_path, config, use_fr_files=False, detection=True)


    # Extract the stars on the image handle
    extractStarsImgHandle(img_handle, config=config)


