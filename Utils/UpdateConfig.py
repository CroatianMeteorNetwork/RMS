# RPi Meteor Station
# Copyright (C) 2024
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import argparse
import os
import sys

from Utils.AuditConfig import validatePath
import glob
import shutil
import tempfile
from RMS.Misc import mkdirP
from datetime import datetime

try:
    # Python 3
    import configparser


except:
    # Python 2
    import ConfigParser as configparser


def printWarning(config_path, template):

    print("This utility will copy all the options from the config file {} ".format(config_path))
    print("into a new file based on the template {} \n".format(template))
    print("Any options in the config file which are not in the template")
    print("will be copied to the new config, however any commented options,")
    print("or bespoke comments will be lost. \n\n")

    response = input("Would you like to proceed?")
    if response.lower() != "y":
        exit()
    else:
        return

def parseConfigFileBySection(config_path):
    """Parse the .config file, into one list and two lists of lists


    Arguments:
        config_path: [str] Path to the .config file

    Returns:
        sections_list: A list of all the sections
        options_list_of_lists: A list of lists of all the options
        values_list_of_lists: A list of lists of all the values

    """

    config = configparser.ConfigParser()
    config.read(os.path.expanduser(config_path))

    sections_list = []
    options_list_of_lists = []
    values_list_of_lists = []

    for section in config.sections():
        sections_list.append(section)
        options_list = []
        values_list = []
        for option in config.options(section):
            if option:
                options_list.append(option)
                value = config.get(section,option)
                if ";" in value:
                    value = value[:value.index(";")].strip()
                values_list.append(value)
        options_list_of_lists.append(options_list)
        values_list_of_lists.append(values_list)

    return sections_list, options_list_of_lists, values_list_of_lists

def backupConfig(config_file_path, number_to_keep=5):

    """Create numbered backups of the .config file


        Arguments:
            config_path: [str] Path to the .config file

        Returns:
            sections_list: A list of all the sections
            options_list_of_lists: A list of lists of all the options
            values_list_of_lists: A list of lists of all the values

        """

    config_file_backups = sorted(glob.glob(config_file_path + "_*"), reverse=True)
    for config_backup in config_file_backups:
        name, number = config_backup.split("_")[0], config_backup.split("_")[1]
        try:
            number = int(number)
        except:
            continue
        if number > number_to_keep:
            continue
        if int(number) < number_to_keep:
            number += 1
            os.rename(config_backup,"{}_{}".format(name,number))
        else:
            os.rename(config_backup,"{}_{}".format(name,number))

    config_backup = config_file_path +"_1"
    shutil.copy(config_file_path, config_file_path +"_1")

    return config_backup



def sectionHeaderLine(line, section_list):

    """
    Detect if this is a section header line
    Qualification is an opening [
    A word which is the the list of sections
    Following by a closing ]

    Arguments:
        line: line to be checked
        section_list: list of all the expected sections

    Returns:
        bool

    """


    opening_square_bracket, closing_square_bracket = False, False
    opening_square_bracket_position, closing_square_bracket_position = 0, 0

    n = 0
    for c in line:
        if c == "[" and opening_square_bracket_position == False:
            opening_square_bracket = True
            opening_square_bracket_position = n
        if c == "]" and opening_square_bracket:
            closing_square_bracket = True
            closing_square_bracket_position = n
        n += 1
    if opening_square_bracket and closing_square_bracket:
        header = line[opening_square_bracket_position + 1:closing_square_bracket_position]
        if header in section_list:
            return True
        else:
            return False
    else:
        return False

def getOption(line, delimiter =":"):

    """
    Get the option, if any from a line

    Arguments:
        line: line to be checked
        delimiter: optional default :

    Returns:
        str: option name

    """

    if delimiter in line:
        return line[0:line.index(delimiter)].strip().lower()
    else:
        return None

def getValue(line, delimiter=":", comment=";"):


    """
    Get the value, if any from a line

    Arguments:
        line: line to be checked
        delimiter: optional default :

    Returns:
        str: value

    """


    if delimiter in line:
        value = line[line.index(delimiter) + 1:].strip()
        if comment in value:
            value = value[:value.index(comment)]
        return value
    else:
        return None



def insert(current_section, insert_options_list, interactive=False, newline_after_last = True):


    """
    Create a string containing options which can be inserted into the .config file

    Arguments:
        line: line to be checked
        delimiter: optional default :

    Returns:
        str: value

    """
    output_lines = ""
    insert_made = False



    for section, option, value in insert_options_list:
        if section == current_section:
            insert_made = True
            insert = "{}: {}\n".format(option, value)
            if interactive:
                print("Working in section {}".format(section))
                response = input("Would you like to insert {}".format(insert))
                if response.lower() == "y":
                    output_lines += insert
            else:
                output_lines += insert

    if insert_made:
        comment_lines = ""
        comment_lines += "; These options added automatically by {:s} ".format(os.path.basename(sys.argv[0]))
        comment_lines += "; on {:s}\n\n".format(datetime.utcnow().strftime('%Y-%m-%dT%H:%M'))
        output_lines = comment_lines + output_lines
        if newline_after_last:
            output_lines += "\n"

    return output_lines

def populateTemplateConfig(config_template_file_path, config_file_path,
                            sections_list, options_list_of_lists, values_list_of_lists,
                                 temporary_suffix="new", insert_options_list = [], interactive = False):


    """
    Reads in a .config template file, and populates it with the data collected from another .config file
    If an option is commented out in the template, but is required in the station config file, then it
    will be uncommented, provided it does not exist elsewhere.

    Arguments:
        config_template_file_path: path to the template file
        config_file_path: path to the config file t be upgradedd
        sections_list: list of the sections in the original config file
        options_list_of_lists: list of lists of options
        values_list_of_lists: list of lists of values
        temporary_suffix: optional suffix to use for the temporary config file
        insert_options_list: list of [section, option, value] to insert, used if the template is missing required
                                options
        interactive: optional, default False, if True, asks for confirmations

    Returns:
        str: value

    """


    # Create paths for reuse later
    config_template_file_path = os.path.expanduser(config_template_file_path)
    config_file_path = os.path.expanduser(config_file_path)
    config_template_file_name = os.path.basename(config_template_file_path)
    config_file_name = os.path.basename(config_file_path)

    # Create a temporary directory
    temp_directory = tempfile.mkdtemp()
    if os.path.isfile(temp_directory):
        os.unlink(temp_directory)
    mkdirP(temp_directory)

    # Copy the .config file to the temporary directory
    shutil.copy(config_file_path, os.path.join(temp_directory, config_template_file_name))
    # Copy the template file to the temporary directory
    shutil.copy(config_template_file_path, os.path.join(temp_directory, config_template_file_name))

    # Create paths for reuse later
    temporary_template = os.path.join(temp_directory, os.path.basename(config_template_file_path))
    temporary_output_file = os.path.join(temp_directory, "{}_{}".format(config_file_name,temporary_suffix))

    # Create file handles
    input_file_handle = open(temporary_template, "r")
    output_file_handle = open(temporary_output_file,"w")

    # Read the template file in as a list of lines
    lines_list = []
    for line in input_file_handle:
        lines_list.append(line)

    # Work through the template file
    current_section = ""
    for line in lines_list:
        output_line = ""

        # Set up booleans to track the type of each line
        blank_line, comment_line, section_header_line, uncomment_line = False, False, False, False

        # Remove newlines
        line = line.replace("\n","")

        if not len(line):
            blank_line = True

        # Detect a comment line, or a comment line that should be uncommented
        elif line.strip()[0] == ";" or line.strip()[0] == "#":
            comment = line.strip()[1:].strip()
            comment_line = True
            # Could this be an option that is commented out
            # Check to see if it is an option in the current section
            if ":" in comment:
                if current_section in sections_list:
                    option = comment[:comment.index(":")]
                    if option in options_list_of_lists[sections_list.index(current_section)]:
                        uncomment_line = True
                        # This option is required in the station configuration file.
                        # Does it exist elsewhere uncommented - even in the wrong section
                        # This will be overly cautious when the same option name occurs in
                        # multiple sections.
                        for line2 in lines_list:
                            if option in line2:
                                if option == line2[:len(option)]:
                                    # It exists elsewhere uncommented, do not uncomment this line
                                    uncomment_line = False
            else:
                comment_line = True

        # Detect a section header line
        elif sectionHeaderLine(line, sections_list):
            section_header_line = True

        # If this is a section header line, add any additional options before this line
        if section_header_line:
            if current_section != "":
                # This is the last line of this section, so insert any missing options
                # that were passed in as parameters
                output_line += insert(current_section, insert_options_list, interactive=interactive)
            #Get the current section name from the line
            current_section = line.strip()[1:-1]

        # If this was a line that should be uncommented
        if uncomment_line:
            # Retrieve the value
            option_number = options_list_of_lists[sections_list.index(current_section)].index(option)
            value = values_list_of_lists[sections_list.index(current_section)][option_number]
            # Generate the line, without the comment mark
            line = "{}: {}".format(option, value)

        # If not any of these, then it must be a line with an option and value
        if not blank_line and not comment_line and not section_header_line and not uncomment_line:
            option = getOption(line)
            section_number = sections_list.index(current_section)
            if option.lower() in options_list_of_lists[section_number]:
                option_number = options_list_of_lists[section_number].index(option.lower())
                station_value = values_list_of_lists[section_number][option_number]
                # Generate the line
                output_line = "{}: {}\n".format(option, station_value)
            else:
                if interactive:
                    print("Option:{} from Section:{} was not found in station .config, set to {}"
                                                .format(option, current_section, getValue(line)))
                output_line = line

        else:
            output_line += "{}\n".format(line)

        output_file_handle.write(output_line)

    # Close the files
    input_file_handle.close()
    output_file_handle.close()

    return temporary_output_file

def compare(new_config_path, sections_list, options_list_of_lists, values_list_of_lists):

    """
    Compares a configuration file on disc against data held in lists

    Arguments:
        new_config_path: path to the config file to be checked
        sections_list: list of sections
        options_list_of_lists: list of lists of options
        values_list_of_lists: list of lists of values

    Returns:
        errors: human-readable list of errors
        missing_options: list of lists of [section, option, value], intended to be passed to insert

    """


    fh = open(new_config_path,"r")
    config_file_as_list, errors, missing_options = [], [], []


    for line in fh:
        config_file_as_list.append(line.replace("\n",""))

    correct_section = False
    for section in sections_list:
        for option, value in zip(options_list_of_lists[sections_list.index(section)],
                                                                values_list_of_lists[sections_list.index(section)]):
            correct_section, option_value_mismatch, found_option_value = False, False, False
            wrong_value = ""

            for line in config_file_as_list:
                if sectionHeaderLine(line, sections_list):
                    correct_section = True if line.strip() == "[{}]".format(section) else False

                if correct_section:

                    if getOption(line) == option and getValue(line) != value:
                        option_value_mismatch = True
                        found_option_value = True
                        wrong_value = getValue(line)
                        break

                    if getOption(line) == option and getValue(line) == value:
                        found_option_value = True

            if option_value_mismatch and found_option_value:
                errors.append("Found {} in section {} with wrong value {} instead of {}".
                                    format(option, section, wrong_value, value))

            if not found_option_value:
                errors.append("Did not find option {} in section {} with expected value {}"
                                                                            .format(option, section, value))
                missing_options.append([section,option,value])

    return errors, missing_options


def updateConfig(config_file_path, config_template_file_path, interactive=False):

    """
    Reads an existing station config, and writes out in the same format as the template file.


    Arguments:
        new_config_path: path to the config file to be checked
        sections_list: list of sections
        options_list_of_lists: list of lists of options
        values_list_of_lists: list of lists of values

    Returns:
        errors: human-readable list of errors
        missing_options: list of lists of [section, option, value], intended to be passed to insert

    """


    config_file_path = os.path.expanduser(config_file_path)
    config_template_file_path = os.path.expanduser(config_template_file_path)

    if validatePath(config_file_path, ".config"):
        # Make backup of the .config file
        config_backup = backupConfig(config_file_path)
        # Parse the .config file
        sections_list, options_list_of_lists, values_list_of_lists = parseConfigFileBySection(config_file_path)
        # Create a new config file in a temporary directory
        new_config_path = populateTemplateConfig(config_template_file_path, config_file_path, sections_list,
                                                                options_list_of_lists, values_list_of_lists,
                                                                interactive=interactive)

        # Check that every value that was in the station .config file has been migrated in the new .config file
        error_list, missing_option_list = compare(new_config_path, sections_list, options_list_of_lists,
                                                  values_list_of_lists)

        if interactive:
            abort = False
            while len(error_list) or abort:
                print("Unsuccessful compare - the following missing options were noted")
                for error in error_list:
                    print("    {:s}".format(error))
                response = input("Would you like to insert the missing options? (y/n)")
                if response.lower() == "y":
                    new_config_path = populateTemplateConfig(config_template_file_path, config_file_path, sections_list,
                                                         options_list_of_lists, values_list_of_lists,
                                                         insert_options_list=missing_option_list,
                                                             interactive=interactive)

                    error_list, missing_option_list = compare(new_config_path, sections_list, options_list_of_lists,
                                                      values_list_of_lists)
                else:
                    break


            print("The new configuration is stored at {}".format(new_config_path))
            print("A backup of the .config file is stored at {}".format(config_backup))
            response = input("Copy new config file at {} over old config file at {} (y/n)"
                                            .format(new_config_path, config_file_path))
            if response.lower() == "y":
                shutil.copy(new_config_path, config_file_path)

            response = input("Would you like remove the temporary working directory (y/n)")
            if response.lower() == "y":
                files = os.listdir(os.path.dirname(new_config_path))
                for file in files:
                    os.unlink(os.path.join(os.path.dirname(new_config_path), file))
                os.rmdir(os.path.dirname(new_config_path))
                exit(0)
            else:
                exit(0)

        else:
            new_config_path = populateTemplateConfig(config_template_file_path, config_file_path, sections_list,
                                                     options_list_of_lists, values_list_of_lists,
                                                     insert_options_list=missing_option_list,
                                                     interactive=interactive)

            compare(new_config_path, sections_list, options_list_of_lists, values_list_of_lists)
            shutil.copy(new_config_path, config_file_path)
            files = os.listdir(os.path.dirname(new_config_path))
            for file in files:
                os.unlink(os.path.join(os.path.dirname(new_config_path), file))
            os.rmdir(os.path.dirname(new_config_path))


if __name__ == "__main__":
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Upgrade a .config file to match the .configTemplate format")

    arg_parser.add_argument("config_file_path", nargs='?', default="./.config",
                            help="Path to the .config file")

    arg_parser.add_argument("-t","--template", default="./.configTemplate",
                        help="Path to .configTemplate (default: ./.configTemplate)")

    arg_parser.add_argument("-q", "--quiet", default=False, action="store_true",
                            help="Run silently, always the case in python < 3")



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    interactive = not cml_args.quiet

    # input is not safe in python < 3
    if sys.version_info.major < 3:
        interactive = False

    if interactive:
        printWarning(cml_args.config_file_path, cml_args.template)
    updateConfig(cml_args.config_file_path, cml_args.template, interactive)