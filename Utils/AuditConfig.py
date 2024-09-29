from __future__ import print_function, division, absolute_import

import re
import argparse
import os
import sys


# Map FileNotFoundError to IOError in Python 2 as it does not exist
if sys.version_info[0] < 3:
    FileNotFoundError = IOError



try:
    # Python 3
    import configparser

except:
    # Python 2
    import ConfigParser as configparser

OMIT_FROM_CONFIG = {'lat',  # Deprecated DFNS Station
                    'lon',  # Deprecated DFNS Station
                    'location',  # Deprecated DFNS Station
                    'altitude',  # Deprecated DFNS Station
                    'event_monitor_db_name',  # Should not be exposed
                    'force_v4l2',  # Deprecated
                    'mask',  # Should not be exposed
                    'platepar_name',  # Should not be exposed
                    'brightness',  # Deprecated
                    'contrast',  # Deprecated
                    'dark_file',  # Deprecated
                    'use_dark',  # Deprecated
                    'mask_remote_name',  # Should not be exposed
                    'remote_mask_dir',  # Should not be exposed
                    'platepar_template_dir'  # Should not be exposed
                    }


def extractConfigOptions(file_path):
    """Extract configuration options from ConfigReader.py

    Arguments:
        file_path: [str] Path to the ConfigReader.py file

    """

    options = set()
    options.add('stationid')

    with open(file_path, 'r') as file:
        content = file.read()
        # Look for patterns like parser.has_option(section, "option_name")
        matches = re.findall(r'parser\.has_option\([^,]+,\s*["\'](\w+)["\']', content)
        options.update(match.lower() for match in matches)
    return options


def parseConfigFile(config_path):
    """Parse the .config file, excluding specific words

    Arguments:
        config_path: [str] Path to the .config file

    """

    config = configparser.ConfigParser()
    config.read(config_path)
    options = set()

    for section in config.sections():
        for option in config.options(section):
            if option.lower():
                options.add(option.lower())

    return options


def checkCommentedOptions(config_path, options):
    commented_out_options = set()

    if not os.path.exists(config_path):
        print("Config file {} does not exist.".format(config_path))
        return commented_out_options

    with open(config_path, 'r') as file:
        lines = file.readlines()

    for option in options:
        commented_option_1 = ";{}:".format(option)
        commented_option_2 = "; {}:".format(option)
        for line in lines:
            if commented_option_1 in line or commented_option_2 in line:
                commented_out_options.add(option.lower())
                break

    return commented_out_options


def validatePath(path, file_name):
    """Validate if the given path includes a filename and exists."""
    if not os.path.basename(path):
        raise ValueError("'{}' does not include a filename.".format(path))
    
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError("{} not found. Tried to find it at absolute path: '{}'".format(file_name, abs_path))
    
    return abs_path   


def compareConfigs(config_path, template_path, configreader_path, dev_report=False):
    """Audit .config, and optionally .configTemplate, by comparing options between '.config',
    '.configTemplate', and 'ConfigReader.py'

    Arguments:
        config_path: [str] Path to the .config file
        template_path: [str] Path to the .configTemplate file
        configreader_path: [str] Path to the ConfigReader.py file
        dev_report: [bool] enable auditing template file for developers

    Returns:
        str: A formatted string containing the comparison results
    """
    
    # Initialize flags and empty sets
    found_config = found_template = found_configreader = False
    config_file_options = template_file_options = configreader_file_options = set()

    # Try to load and parse each file
    try:
        validatePath(config_path, ".config file")
        config_file_options = parseConfigFile(config_path)
        found_config = True
    except (ValueError, FileNotFoundError) as e:
        dev_report = True
        print("Error loading .config file: {}".format(e))

    try:
        validatePath(template_path, ".configTemplate")
        template_file_options = parseConfigFile(template_path)
        found_template = True
    except (ValueError, FileNotFoundError) as e:
        dev_report = True
        print("Error loading .configTemplate file: {}".format(e))

    try:
        validatePath(configreader_path, "ConfigReader.py")
        configreader_file_options = extractConfigOptions(configreader_path)
        found_configreader = True
    except (ValueError, FileNotFoundError) as e:
        dev_report = True
        print("Error loading ConfigReader.py file: {}".format(e))

    # Find missing and extra options
    missing_in_config_wrt_template = template_file_options - config_file_options if found_template and found_config else set()
    missing_in_config_wrt_cr = configreader_file_options - config_file_options if found_configreader and found_config else set()
    missing_in_template_wrt_cr = configreader_file_options - template_file_options if found_configreader and found_template else set()
    extra_in_config = config_file_options - configreader_file_options if found_config and found_configreader else set()
    extra_in_template = template_file_options - configreader_file_options if found_template and found_configreader else set()

    # Find commented out options (only if respective files are found)
    commented_options_in_config = checkCommentedOptions(config_path, missing_in_config_wrt_cr) if found_config else set()
    commented_options_in_template = checkCommentedOptions(template_path, missing_in_template_wrt_cr) if found_template else set()

    # Remove commented out options from missing
    missing_in_config_wrt_template -= commented_options_in_config
    missing_in_config_wrt_cr -= (commented_options_in_config | OMIT_FROM_CONFIG)
    missing_in_template_wrt_cr -= (commented_options_in_template | OMIT_FROM_CONFIG)

    # Generate report
    output = []
    output.append("")
    output.append("=" * 80)
    title = "CONFIG COMPARISON RESULTS"
    if dev_report:
        title = "DEV " + title
    output.append(title.center(80))
    output.append("=" * 80 + "\n")

    if not found_config:
        output.append("WARNING: .config file not found. Some comparisons will be skipped.".center(80))
        output.append("")

    if not found_template:
        output.append("NOTE: .configTemplate file not found. Some comparisons will be skipped.".center(80))
        output.append("")

    if not found_configreader:
        output.append("WARNING: ConfigReader.py file not found. Some comparisons will be skipped.".center(80))
        output.append("")

    if missing_in_config_wrt_template and found_template and found_config:
        output.append("OPTIONS MISSING IN .CONFIG FILE (PRESENT IN TEMPLATE)".center(80))
        output.append("****** CONSIDER MANUALLY UPDATING .CONFIG FILE ******".center(80))
        output.append("Default values will be used:".center(80))
        output.append("-" * 80)
        for option in sorted(missing_in_config_wrt_template):
            output.append(" - {}".format(option))
        output.append("")

    if missing_in_template_wrt_cr and dev_report and found_template and found_configreader:
        output.append("OPTIONS NOT IN TEMPLATE FILE BUT IMPLEMENTED IN RMS:".center(80))
        output.append("-" * 80)
        for option in sorted(missing_in_template_wrt_cr):
            output.append(" - {}".format(option))
        output.append("")

    if missing_in_config_wrt_cr and dev_report and found_config and found_configreader:
        output.append("OPTIONS NOT IN CONFIG FILE BUT IMPLEMENTED IN RMS:".center(80))
        output.append("-" * 80)
        for option in sorted(missing_in_config_wrt_cr):
            output.append(" - {}".format(option))
        output.append("")

    if commented_options_in_config and found_config:
        output.append("OPTIONS COMMENTED OUT IN .CONFIG FILE".center(80))
        output.append("Default values will be used:".center(80))
        output.append("-" * 80)
        for option in sorted(commented_options_in_config):
            output.append(" - {}".format(option))
        output.append("")

    if commented_options_in_template and dev_report and found_template:
        output.append("OPTIONS COMMENTED OUT IN TEMPLATE FILE:".center(80))
        output.append("-" * 80)
        for option in sorted(commented_options_in_template):
            output.append(" - {}".format(option))
        output.append("")

    if extra_in_config and found_config and found_configreader:
        output.append("OPTIONS IN .CONFIG FILE NOT IMPLEMENTED IN RMS (will be ignored):".center(80))
        output.append("-" * 80)
        for option in sorted(extra_in_config):
            output.append(" - {}".format(option))
        output.append("")

    if extra_in_template and dev_report and found_template and found_configreader:
        output.append("OPTIONS IN TEMPLATE FILE NOT IMPLEMENTED IN RMS (will be ignored):".center(80))
        output.append("-" * 80)
        for option in sorted(extra_in_template):
            output.append(" - {}".format(option))
        output.append("")

    if found_config and found_template and found_configreader:
        if not missing_in_config_wrt_template and not extra_in_config:
            output.append("There are no missing or extraneous options in .config file.".center(80))
            output.append("")

        output.append("=" * 80)
        output.append("Total options in template: {}".format(len(template_file_options)).center(80))
        output.append("Total options in .config file: {}".format(len(config_file_options)).center(80))
        output.append("Common options: {}".format(len(template_file_options.intersection(config_file_options))).center(80))
        output.append("=" * 80 + "\n")
    else:
        output.append("=" * 80)
        output.append("INCOMPLETE COMPARISON".center(80))
        output.append("Some files were not found, resulting in an incomplete comparison.".center(80))
        output.append("=" * 80 + "\n")

    return "\n".join(output)


if __name__ == "__main__":
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Audit .config and optionally .configTemplate files.")

    arg_parser.add_argument("config_path", nargs='?', default="./.config", help="Path to the .config file")

    arg_parser.add_argument("--template", default="./.configTemplate",
                        help="Path to .configTemplate (default: ./.configTemplate)")

    arg_parser.add_argument("--configreader", default="./RMS/ConfigReader.py",
                        help="Path to ConfigReader.py (default: ./RMS/ConfigReader.py)")

    arg_parser.add_argument('-d', '--dev', action="store_true", help="""Audit template file. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    try:
        print(compareConfigs(cml_args.config_path, cml_args.template, cml_args.configreader,
                             dev_report=cml_args.dev))
        
    except (ValueError, FileNotFoundError) as e:
        print("Error: {}".format(str(e)))
        exit(1)