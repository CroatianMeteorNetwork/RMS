#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Acknowledgement
---------------
This script builds on a script graciously shared by Steve Matheson.

Quick usage example
-------------------
Upgrade single or multi stations config (with backups) to latest configTemplate format preserving attributes:

python -m Utils.MigrateConfig -u

To first test the upgrade without applying it, run:

python -m Utils.MigrateConfig

# To incorporate recent default values (e.g. star catalog file) use the -r option:

python -m Utils.MigrateConfig -r
'''
import sys
import os
import re
import argparse
from datetime import datetime
import shutil
import platform
from io import StringIO

from Utils.AuditConfig import extractConfigOptions
from RMS.Misc import getRmsRootDir

# Get ConfigReader.py path dynamically
CONFIGREADER_PATH = os.path.join(getRmsRootDir(), "RMS", "ConfigReader.py")

# Extract valid options from ConfigReader.py
VALID_OPTIONS = extractConfigOptions(CONFIGREADER_PATH)


def logMessage(message):
    """Print and write log messages to buffer"""
    print(message)
    log_buffer.write(message + "\n")


def updateConfig(original_config_file, template_config_file, args, backup=True):
    global log_buffer
    station_id = None

    # Read the original config and extract station ID
    with open(original_config_file, "r") as file:
        for line in file:
            match = re.match(r"^\s*stationID\s*:\s*([\w\d-]+)", line)
            if match:
                station_id = match.group(1)
                break

    # Ensure station_id is valid
    if not station_id:
        logMessage("ERROR: No valid stationID found in {}".format(original_config_file))
        sys.exit(1)
    
    # build output path next to the input .config ---
    output_dir = os.path.dirname(os.path.abspath(original_config_file))
    new_config_file = os.path.join(output_dir, "configNew_{}".format(station_id))        

    if backup:
        original_backup = os.path.join(os.path.dirname(original_config_file), "{}.config.original.bak".format(station_id))
        latest_backup = os.path.join(os.path.dirname(original_config_file), "{}.config.bak".format(station_id))

        try:
            if not os.path.exists(original_backup):
                shutil.copy(original_config_file, original_backup)
                logMessage("\nOriginal backup created: {}".format(original_backup))

            shutil.copy(original_config_file, latest_backup)
            logMessage("\nBackup created: {}".format(latest_backup))
        except Exception as e:
            logMessage("ERROR: Backup failed: {}".format(e))
            sys.exit(1)

    # list of attributes with recently updated defaults
    recent_defaults_list = ["[Calibration]star_catalog_file"]

    if not os.path.exists(original_config_file):
        logMessage("ERROR: Can't find existing .config file {}".format(original_config_file))
        sys.exit(1)

    if not os.path.exists(template_config_file):
        logMessage(
                "ERROR: Can't find template file {}\n"
            "       No '.configTemplate' detected. Either:\n"
            "       - run ./Scripts/RMS_Update.sh to create/update it, or\n"
            "       - pass the template path explicitly with -t\n"
            "         e.g.  python -m Utils.MigrateConfig -t /path/to/.configTemplate"
            .format(template_config_file)
        )
        sys.exit(1)

    logMessage("\nInput: {}".format(original_config_file))

    # read the original config and build a dictionary of attributes
    # and values

    with open(original_config_file, "r") as file:
        config_lines = file.readlines()
        logMessage("{} lines read".format(len(config_lines)))

    attributes_dict = {}

    section = "unknown"
    for line in config_lines:
        l = line.strip()

        # skip blank
        if l == "":
            continue

        # skip comment lines
        if l[:1] == ";":
            continue

        # save current section
        m = re.match(r'\s*(\[[^\]]+\])', l)
        if m:
            # keep only the bracketed section name (e.g. "[System]")
            section = m.group(1)
            continue

        bits = re.split(": ", l)
        cnt = len(bits)
        if cnt > 1:
            if (section + bits[0]) in attributes_dict:
                logMessage("WARNING - duplicate value for {}, assuming last value".format(bits[0]))
            i = l.index(": ") + 2 
            attributes_dict[section + bits[0]] = l[i:].strip()

    # --- SPECIAL-CASE: legacy 'quota_management_disabled' > 'quota_management_enabled' ---
    for legacy_key in list(attributes_dict.keys()):
        if legacy_key.endswith('quota_management_disabled'):
            raw_val = attributes_dict.pop(legacy_key).split(';', 1)[0].strip().lower()
            disabled = raw_val in ('1', 'true', 'yes', 'on')
            new_key = legacy_key.replace('quota_management_disabled',
                                         'quota_management_enabled')
            if new_key not in attributes_dict:       # don't override if already present
                attributes_dict[new_key] = 'false' if disabled else 'true'

    logMessage("{} attributes identified".format(len(attributes_dict)))

    if not "[System]stationID" in attributes_dict:
        logMessage("\nERROR: No stationID found in the [System] section of .config !!!\n")
        raise SystemExit("Exiting the program\n")

    logMessage("\n[System]stationID: {}".format(attributes_dict['[System]stationID']))

    new_config_file = os.path.join(
        output_dir,
        "configNew_{}".format(attributes_dict['[System]stationID'])
    )
    if args.output:
        # absolute > use verbatim, relative > still next to the original
        new_config_file = (
            args.output if os.path.isabs(args.output)
            else os.path.join(output_dir, args.output)
        )
        logMessage("Output: {}\n".format(new_config_file))

    with open(template_config_file, "r") as templatefile:
        template_lines = templatefile.readlines()
        logMessage("{} lines read from {}".format(len(template_lines),template_config_file))

    logMessage("\nMerging attributes...")

    custom_cnt = 0
    section = "unknown"
    tlcnt = 0

    with open(new_config_file, "w") as newfile:
        for line in template_lines:

            # first write out anything that doesn't look like an attribute

            l = line.strip()
            tlcnt += 1

            if not l:  # blank
                newfile.write("\n")
                newfile.flush()
                continue

            # save current section
            m = re.match(r'\s*(\[[^\]]+\])', l)
            if m:
                section = m.group(1)

            if l[:1] == ";":  # comment
                newfile.write("{}\n".format(l))
                newfile.flush()
                continue

            bits = re.split(": ", l)  # is it an attribute?
            cnt = len(bits)

            if cnt != 2:  # no it isn't
                newfile.write("{}\n".format(l))
                newfile.flush()
                continue

            # insert our original .config value if we have one
            # and its not the default value

            # perhaps we need an exclude list so some old values aren't carried forward
            # and new defaults (from template) apply ???

            if (section + bits[0]) in attributes_dict:
                original_value = attributes_dict[section + bits[0]]
                bits2 = original_value.split(";")
                if len(bits2) == 2:
                    original_value = bits2[
                        0
                    ].strip()  # trim comments from attribute line

                del attributes_dict[
                    section + bits[0]
                ]  # remove so we can list unrecognised later
                new_default_value = bits[1].strip()
                if original_value != new_default_value:
                    if not (
                        (section + bits[0]) in recent_defaults_list and args.recent
                    ):
                        newfile.write(
                            "{}: {}\n".format(bits[0],original_value)
                        )  # write original attribute/value pair
                        newfile.flush()
                        logMessage(
                            "  {}{}: template default '{}' => kept '{}'"
                                    .format(section, bits[0], new_default_value, original_value)
                        )
                        custom_cnt += 1
                        continue

                    logMessage(
                        "  {}{}: '{}' => RECENT template default '{}'"
                                .format(section, bits[0], original_value, new_default_value)
                    )

            # if we don't need to update the template line write it out as is
            newfile.write("{}\n".format(l))
            newfile.flush()

        newfile.write(
            "\n; Reformated by {} on {}\n".format(sys.argv[0],datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        newfile.flush()
        logMessage("{} customized attributes written".format(custom_cnt))

        if len(attributes_dict) > 0:
            logMessage("\n{} unrecognized attributes preserved in new config".format(len(attributes_dict)))

        # Read the new config into memory to insert attributes at correct positions
        with open(new_config_file, "r") as newfile:
            new_config_lines = newfile.readlines()

        # Track section positions (last non-blank, non-comment option in each section)
        section_positions = {}
        current_section = None

        for idx, line in enumerate(new_config_lines):
            stripped = line.strip()

            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped
                section_positions[current_section] = {"last_option": None}  # Track last valid option

            elif stripped and not stripped.startswith(";"):  # Ignore comments and blank lines
                if current_section:
                    section_positions[current_section]["last_option"] = idx  # Update last valid option

        # Dictionary to track whether a comment was added per section
        added_comment = {}

        # Insert preserved options **after the last valid option in their section**
        for key, value in attributes_dict.items():
            section, attr_name = key.split("]", 1)
            section = section + "]"

            # **Only preserve attributes that are valid (found in ConfigReader.py)**
            if attr_name.lower() not in VALID_OPTIONS:
                logMessage("  IGNORING: {} {} => {} (Not supported by RMS)".format(section, attr_name.strip(), value))
                continue  # Skip this attribute

            # Find where to insert the attribute
            if section in section_positions and section_positions[section]["last_option"] is not None:
                insert_position = section_positions[section]["last_option"] + 1  # Insert after last valid option
            else:
                # If section is missing or empty, add at the end
                new_config_lines.append("\n{}\n".format(section))
                insert_position = len(new_config_lines)

            # Ensure a blank line before inserting the first preserved option (if needed)
            if not new_config_lines[insert_position - 1].strip():
                new_config_lines.insert(insert_position, "\n")

            # Ensure we only add the comment **once** per section
            if section not in added_comment:
                new_config_lines.insert(insert_position + 1, "; The following options were preserved but are not in the template\n")
                added_comment[section] = True  # Prevent duplicate comments

            # Insert attribute at the correct position **after the last attribute in the section**
            new_config_lines.insert(insert_position + 2, "{}: {}\n".format(attr_name.strip(), value))
            logMessage("  PRESERVING: {} {} => {} (Supported by RMS)".format(section, attr_name.strip(), value))

        # Write the modified config back to disk
        with open(new_config_file, "w") as newfile:
            newfile.writelines(new_config_lines)
            newfile.flush()
    return new_config_file, station_id


def getSystemInfo():
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    processor = platform.processor()

    return {
        "System": system,
        "Release": release,
        "Version": version,
        "Machine": machine,
        "Processor": processor
    }


if __name__ == "__main__":
    
    print("\n{} - Migrate RMS .config to latest template standard carrying forward customizations\n".format(sys.argv[0]))

    parser = argparse.ArgumentParser(
        description="Migrate .config to latest format carrying forward attribute customizations"
    )
    parser.add_argument(
        "-i", "--input", help='input .config filename, overrides default ".config"'
    )
    parser.add_argument(
        "-t",
        "--template",
        help='reference template filename, overrides default ".configTemplate"',
    )
    parser.add_argument(
        "-o",
        "--output",
        help='reformated .config filename, overrides default "configNew_<CamId>"',
    )
    parser.add_argument(
        "-u",
        "--update",
        help="automatically UPDATE the active .config",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--recent",
        help="update recently changed default values",
        action="store_true",
    )
    args = parser.parse_args()
    
    curr_path = os.getcwd()
    print("Current path is {}".format(curr_path))

    
    # Get the path to the RMS root directory
    rms_root_dir = getRmsRootDir()
    template_config_file = os.path.join(rms_root_dir, ".configTemplate")

    if args.template:
        template_config_file = args.template
    
    print("\nTemplate: {}".format(template_config_file))

    # assume default input
    original_config_files = [os.path.join(rms_root_dir, ".config")]

    # if multi-cam find and assume those
    stations_dir = os.path.expanduser("~/source/Stations")
    if os.path.isdir(stations_dir):
        for d in os.listdir(stations_dir):
            f = os.path.join(stations_dir, d, ".config")
            if os.path.isfile(f):                 # skip broken/missing configs
                original_config_files.append(f)
    print("Multi-cam count: {}".format(len(original_config_files) - 1))

    # if specified assume
    if args.input:
        original_config_files = [args.input]

    # process each input config
    for orig_config in original_config_files:
        log_buffer = StringIO()
        out_file_name, station_id = updateConfig(orig_config, template_config_file, args, backup=args.update)

        if args.update:
            try:
                shutil.copy(out_file_name, orig_config)
                os.remove(out_file_name)
                print("Updated\n")

                log_file = os.path.join(os.path.dirname(orig_config), "{}_MigrateConfig.log".format(station_id))
                with open(log_file, "a") as log:
                    log.write("\n=== Migration Log: {} ===\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    log.write(log_buffer.getvalue())
                    log.write("\nMigration applied successfully.\n")

                log_buffer.close()

                print("Updated\n")

            except Exception as e:
                print("ERROR: Update failed: {}".format(e))
                sys.exit(1)
        else:
            print("\nSaved new config to: {}".format(out_file_name))

    if not args.update:
        print(
            "\nAfter saving a copy of the existing .config, copy/rename new version(s) into production."
        )
        print("    e.g. cp ConfigNew_XXxxxx .config \n")

    print("Done.\n")
