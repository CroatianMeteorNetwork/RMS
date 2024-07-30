import re
import configparser
import argparse
import os


def extractConfigOptions(file_path):
    """Extract configuration options from ConfigReader.py
    
    Arguments:
        file_path: [str] Path to the ConfigReader.py file
    
    """

    options = set()

    # Ignore DFNS specific options
    excluded_words = {'lat', 'lon', 'location', 'altitude'}

    with open(file_path, 'r') as file:
        content = file.read()
        # Look for patterns like parser.has_option(section, "option_name")
        matches = re.findall(r'parser\.has_option\([^,]+,\s*["\'](\w+)["\']', content)
        options.update(match.lower() for match in matches if match.lower() not in excluded_words)
    return options


def parseConfigFile(config_path):
    """Parse the .config file, excluding specific words
    
    Arguments:
        config_path: [str] Path to the .config file

    """

    config = configparser.ConfigParser()
    config.read(config_path)
    options = set()

    # Ignore stationID which is a special case
    excluded_words = {'stationid'}

    for section in config.sections():
        for option in config.options(section):
            if option.lower() not in excluded_words:
                options.add(option.lower())

    return options

def check_commented_options(config_path, options):
    commented_out_options = set()
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} does not exist.")
        return commented_out_options

    with open(config_path, 'r') as file:
        lines = file.readlines()
        
    for option in options:
        commented_option_1 = f";{option}:"
        commented_option_2 = f"; {option}:"
        for line in lines:
            if commented_option_1 in line or commented_option_2 in line:
                commented_out_options.add(option.lower())
                break
    
    return commented_out_options


def compareConfigs(reference_path, config_path, template=True):
    """Compare ConfigReader.py options with .config file options

    Arguments:
        reference_path: [str] Path to the ConfigReader.py or the .configTemplate file
        config_path: [str] Path to the .config file

    """
    
    config_file_options = parseConfigFile(config_path)

    if template:
        reference_file_options = parseConfigFile(reference_path)

    else:
        reference_file_options = extractConfigOptions(reference_path)

    missing_in_config = reference_file_options - config_file_options
    extra_in_config = config_file_options - reference_file_options

    commented_options = check_commented_options(config_path, missing_in_config)
    missing_in_config -= commented_options

    print("\n" + "="*80)
    print("CONFIG COMPARISON RESULTS".center(80))
    print("="*80 + "\n")

    if missing_in_config:
        print("OPTIONS NOT IN .CONFIG FILE (default values will be used):".center(80))
        print("-"*80)
        for option in sorted(missing_in_config):
            print(f"  • {option}")
        print()

    if commented_options:
        print("OPTIONS COMMENTED OUT IN .CONFIG FILE (default values will be used):".center(80))
        print("-"*80)
        for option in sorted(commented_options):
            print(f"  • {option}")
        print()

    if extra_in_config:
        print("OPTIONS IN .CONFIG FILE NOT IMPLEMENTED IN CONFIGREADER (will be ignored):".center(80))
        print("-"*80)
        for option in sorted(extra_in_config):
            print(f"  • {option}")
        print()

    if not missing_in_config and not extra_in_config:
        print("All options match between ConfigReader.py and .config file.".center(80))
        print()

    print("="*80)
    print(f"Total options in reference: {len(reference_file_options)}".center(80))
    print(f"Total options in .config file: {len(config_file_options)}".center(80))
    print(f"Common options: {len(reference_file_options.intersection(config_file_options))}".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compare either .configTemplate or ConfigReader.py "
                                         "options with .config file options.")

    arg_parser.add_argument("config_path", default="./.config", help="Path to the .config file")

    arg_parser.add_argument('-r', '--cr', action="store_true", \
        help="""Use ConfigReader options as reference. """)

    arg_parser.add_argument("--template", default="./.configTemplate",
                        help="Path to .configTemplate (default: ./.configTemplate)")

    arg_parser.add_argument("--configreader", default="./RMS/ConfigReader.py",
                        help="Path to ConfigReader.py (default: ./RMS/ConfigReader.py)")
    
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    config_path = cml_args.config_path
    configreader_path = cml_args.configreader
    configTemplate_path = cml_args.template

    if not os.path.exists(configreader_path) and cml_args.cr:
        print(f"Error: ConfigReader.py not found at {configreader_path}")
        exit(1)

    if not os.path.exists(configTemplate_path) and not cml_args.cr:
        print(f"Error: .configTemplate not found at {configTemplate_path}")
        exit(1)

    if not os.path.exists(config_path):
        print(f"Error: .config file not found at {config_path}")
        exit(1)

    if cml_args.cr:
        compareConfigs(configreader_path, config_path,template=False)
    else:
        compareConfigs(configTemplate_path, config_path,template=True)