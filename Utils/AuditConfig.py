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


def checkCommentedOptions(config_path, options):
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

    # Gather options from all three files
    config_file_options = parseConfigFile(config_path)
    template_file_options = parseConfigFile(template_path)
    configreader_file_options = extractConfigOptions(configreader_path)

    # Find missing options relative to each other
    missing_in_config_wrt_template = template_file_options - config_file_options
    missing_in_config_wrt_cr = configreader_file_options - config_file_options
    missing_in_template_wrt_cr = configreader_file_options - template_file_options

    # Find extraneous options
    extra_in_config = config_file_options - configreader_file_options
    extra_in_template = template_file_options - configreader_file_options

    # Find commented out options
    commented_options_in_config = checkCommentedOptions(config_path, missing_in_config_wrt_cr)
    commented_options_in_template = checkCommentedOptions(template_path, missing_in_template_wrt_cr)

    # Remove commented out options from missing
    missing_in_config_wrt_template -= commented_options_in_config
    missing_in_config_wrt_cr -= commented_options_in_config
    missing_in_template_wrt_cr -= commented_options_in_template

    # Generate report
    output = []
    output.append("=" * 80)
    output.append("CONFIG COMPARISON RESULTS".center(80))
    output.append("=" * 80 + "\n")

    if missing_in_config_wrt_template:
        output.append("OPTIONS MISSING IN .CONFIG FILE (PRESENT IN TEMPLATE)".center(80))
        output.append("****** CONSIDER MANUALLY UPDATING .CONFIG FILE ******".center(80))
        output.append("Default values will be used:".center(80))
        output.append("-" * 80)
        for option in sorted(missing_in_config_wrt_template):
            output.append(f" • {option}")
        output.append("")

    if missing_in_template_wrt_cr and dev_report:
        output.append("OPTIONS MISSING IN TEMPLATE FILE BUT IMPLEMENTED IN RMS:".center(80))
        output.append("-" * 80)
        for option in sorted(missing_in_template_wrt_cr):
            output.append(f" • {option}")
        output.append("")

    if commented_options_in_config:
        output.append("OPTIONS COMMENTED OUT IN .CONFIG FILE".center(80))
        output.append("Default values will be used:".center(80))
        output.append("-" * 80)
        for option in sorted(commented_options_in_config):
            output.append(f" • {option}")
        output.append("")

    if commented_options_in_template and dev_report:
        output.append("OPTIONS COMMENTED OUT IN TEMPLATE FILE:".center(80))
        output.append("-" * 80)
        for option in sorted(commented_options_in_template):
            output.append(f" • {option}")
        output.append("")

    if extra_in_config:
        output.append("OPTIONS IN .CONFIG FILE NOT IMPLEMENTED IN RMS (will be ignored):".center(80))
        output.append("-" * 80)
        for option in sorted(extra_in_config):
            output.append(f" • {option}")
        output.append("")

    if extra_in_template and dev_report:
        output.append("OPTIONS IN TEMPLATE FILE NOT IMPLEMENTED IN RMS (will be ignored):".center(80))
        output.append("-" * 80)
        for option in sorted(extra_in_template):
            output.append(f" • {option}")
        output.append("")

    if not missing_in_config_wrt_template and not extra_in_config:
        output.append("There are no missing or extraneous options in .config file.".center(80))
        output.append("")

    output.append("=" * 80)
    output.append(f"Total options in template: {len(template_file_options)}".center(80))
    output.append(f"Total options in .config file: {len(config_file_options)}".center(80))
    output.append(f"Common options: {len(template_file_options.intersection(config_file_options))}".center(80))
    output.append("=" * 80 + "\n")

    return "\n".join(output)


def checkFileExists(file_path, file_name):
    """Exit if file path does not exist"""

    if not os.path.exists(file_path):
        print(f"Error: {file_name} not found at {file_path}")
        exit(1)


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

    config_path = cml_args.config_path
    configTemplate_path = cml_args.template
    configreader_path = cml_args.configreader

    checkFileExists(config_path, ".config file")
    checkFileExists(configTemplate_path, ".configTemplate")
    checkFileExists(configreader_path, "ConfigReader.py")

    print(compareConfigs(config_path, configTemplate_path, configreader_path, dev_report=cml_args.dev))
