import sys
import os
import re
import argparse
from datetime import datetime
import shutil
from io import StringIO

if sys.version_info < (2,7):
    print("Python versions < 2.7 not supported")
    sys.exit(1)

from Utils.AuditConfig import extractConfigOptions
from RMS.Misc import getRmsRootDir
from RMS.ConfigReader import Config as DefaultConfig

# Get ConfigReader.py path dynamically
CONFIGREADER_PATH = os.path.join(getRmsRootDir(), "RMS", "ConfigReader.py")

# Extract valid options from ConfigReader.py
VALID_OPTIONS = extractConfigOptions(CONFIGREADER_PATH)

def log_message(message, log_buffer=None):
    """Print and optionally write log messages to buffer"""
    print(message)
    if log_buffer is not None:
        log_buffer.write(message + "\n")

def get_clean_default_config():
    """Get a Config instance with true default values, preventing file parsing."""
    default_config = DefaultConfig()
    # Set a non-existent config file to prevent parsing
    default_config.config_file_name = "_no_such_file_"
    default_config.__init__()
    return default_config

def write_compact_config(config_dict, output_file):
    """Write a compact config file with only non-default values."""
    sections = {}
    # Get clean default config
    default_config = get_clean_default_config()
    
    # Debug print all default values
    print("\nDefault values:")
    for attr in sorted(dir(default_config)):
        if not attr.startswith('_') and not callable(getattr(default_config, attr)):
            print(f"{attr}: {getattr(default_config, attr)}")
    
    # Organize options by section
    for section, options in config_dict.items():
        non_default_options = {}
        for option, value in options.items():
            # Skip if option is not supported
            if option.lower() not in VALID_OPTIONS:
                continue
                
            # Get corresponding attribute name in Config class
            attr_name = option.lower()
            if hasattr(default_config, attr_name):
                default_value = getattr(default_config, attr_name)
                # Convert both values to string and strip for comparison
                value_str = str(value).strip().lower()
                default_str = str(default_value).strip().lower()
                
                # Debug logging
                print(f"\nComparing {option}:")
                print(f"  Value:   '{value_str}'")
                print(f"  Default: '{default_str}'")
                
                if value_str != default_str:
                    non_default_options[option] = value
        
        if non_default_options:
            sections[section] = non_default_options
    
    # Write the compact config
    with open(output_file, 'w') as f:
        for section, options in sorted(sections.items()):
            f.write(f"[{section}]\n")
            for option, value in sorted(options.items()):
                f.write(f"{option}: {value}\n")
            f.write("\n")

def extract_sections_from_config(config_file):
    """Extract sections and their options from a config file."""
    sections = {}
    current_section = None
    
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith(';'):
                continue
                
            # Section header
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                sections[current_section] = {}
                continue
                
            # Option line
            if current_section and ':' in line:
                option, value = line.split(':', 1)
                option = option.strip()
                value = value.strip()
                sections[current_section][option] = value
                
    return sections

def revert_config(config_file):
    """Revert config file to its original backup."""
    # Extract station ID from the config
    station_id = None
    with open(config_file, "r") as file:
        for line in file:
            match = re.match(r"^\s*stationID\s*:\s*([\w\d-]+)", line)
            if match:
                station_id = match.group(1)
                break
    
    if not station_id:
        print(f"ERROR: No valid stationID found in {config_file}")
        return False
        
    original_backup = os.path.join(os.path.dirname(config_file), f"{station_id}.config.original")
    
    if not os.path.exists(original_backup):
        print(f"ERROR: No original backup found at {original_backup}")
        return False
        
    try:
        shutil.copy(original_backup, config_file)
        print(f"Successfully reverted {config_file} to original backup")
        return True
    except Exception as e:
        print(f"ERROR: Revert failed: {e}")
        return False

def update_config(original_config_file, template_config_file, args):
    # If revert option is used, handle it and return
    if args.revert:
        success = revert_config(original_config_file)
        if success:
            return None, None
        else:
            sys.exit(1)

    # Initialize log buffer if logging is enabled
    log_buffer = StringIO() if args.log else None

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
        log_message(f"ERROR: No valid stationID found in {original_config_file}", log_buffer)
        sys.exit(1)
        
    new_config_file = f"configNew_{station_id}"
    if args.output:
        new_config_file = args.output

    # Create backup of original config if it doesn't exist
    original_backup = os.path.join(os.path.dirname(original_config_file), f"{station_id}.config.original")
    if not os.path.exists(original_backup):
        try:
            shutil.copy(original_config_file, original_backup)
            log_message(f"\nOriginal backup created: {original_backup}", log_buffer)
        except Exception as e:
            log_message(f"ERROR: Backup failed: {e}", log_buffer)
            sys.exit(1)

    # list of attributes with recently updated defaults
    recent_defaults_list = ["[Calibration]star_catalog_file"]

    if not os.path.exists(original_config_file):
        log_message(f"ERROR: Can't find existing .config file {original_config_file}", log_buffer)
        sys.exit(1)

    if not os.path.exists(template_config_file):
        log_message(f"ERROR: Can't find template file {template_config_file}", log_buffer)
        sys.exit(1)

    log_message(f"\nInput: {original_config_file}", log_buffer)

    # Process the config based on mode
    if args.compact:
        # Extract sections and options from original config
        config_dict = extract_sections_from_config(original_config_file)
        # Write compact version
        write_compact_config(config_dict, new_config_file)
        log_message(f"\nCompact config written to: {new_config_file}", log_buffer)
    else:
        # read the original config and build a dictionary of attributes and values
        with open(original_config_file, "r") as file:
            config_lines = file.readlines()
            log_message(f"{len(config_lines)} lines read", log_buffer)

        attributes_dict = {}
        section = "unknown"
        
        for line in config_lines:
            l = line.strip()

            if not l or l[:1] == ";":  # skip blank and comment lines
                continue

            if l[0] == "[" and l[-1] == "]":  # save current section
                section = l
                continue

            bits = re.split(": ", l)
            if len(bits) > 1:
                if (section + bits[0]) in attributes_dict:
                    log_message(f"WARNING - duplicate value for {bits[0]}, assuming last value", log_buffer)
                i = l.index(": ") + 2 
                attributes_dict[section + bits[0]] = l[i:].strip()

        log_message(f"{len(attributes_dict)} attributes identified", log_buffer)

        if not "[System]stationID" in attributes_dict:
            log_message("\nERROR: No stationID found in the [System] section of .config !!!\n", log_buffer)
            raise SystemExit("Exiting the program\n")

        log_message(f"\n[System]stationID: {attributes_dict['[System]stationID']}", log_buffer)
        log_message(f"Output: {new_config_file}\n", log_buffer)

        with open(template_config_file, "r") as templatefile:
            template_lines = templatefile.readlines()
            log_message(f"{len(template_lines)} lines read from {template_config_file}", log_buffer)

        log_message("\nMerging attributes...", log_buffer)

        custom_cnt = 0
        section = "unknown"

        with open(new_config_file, "w") as newfile:
            for line in template_lines:
                l = line.strip()

                if not l:  # blank
                    newfile.write("\n")
                    continue

                if l[0] == "[" and l[-1] == "]":  # save current section
                    section = l

                if l[:1] == ";":  # comment
                    newfile.write(f"{l}\n")
                    continue

                bits = re.split(": ", l)
                if len(bits) != 2:  # not an attribute
                    newfile.write(f"{l}\n")
                    continue

                # handle attributes
                if (section + bits[0]) in attributes_dict:
                    original_value = attributes_dict[section + bits[0]]
                    bits2 = original_value.split(";")
                    if len(bits2) == 2:
                        original_value = bits2[0].strip()  # trim comments

                    del attributes_dict[section + bits[0]]
                    new_default_value = bits[1].strip()
                    
                    if original_value != new_default_value:
                        if not ((section + bits[0]) in recent_defaults_list and args.recent):
                            newfile.write(f"{bits[0]}: {original_value}\n")
                            log_message(
                                f"  {section}{bits[0]}: {original_value} carried forward over => {new_default_value}",
                                log_buffer
                            )
                            custom_cnt += 1
                            continue

                        log_message(
                            f"  {section}{bits[0]}: {original_value} RECENT template default applied => {new_default_value}",
                            log_buffer
                        )

                newfile.write(f"{l}\n")

            newfile.write(
                f"\n; Reformated by {sys.argv[0]} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

            log_message(f"{custom_cnt} customized attributes written", log_buffer)

            if len(attributes_dict) > 0:
                log_message(f"\n{len(attributes_dict)} unrecognized attributes preserved in new config", log_buffer)

            # Read the new config into memory to insert attributes
            with open(new_config_file, "r") as newfile:
                new_config_lines = newfile.readlines()

            # Track section positions
            section_positions = {}
            current_section = None

            for idx, line in enumerate(new_config_lines):
                stripped = line.strip()

                if stripped.startswith("[") and stripped.endswith("]"):
                    current_section = stripped
                    section_positions[current_section] = {"last_option": None}

                elif stripped and not stripped.startswith(";"):
                    if current_section:
                        section_positions[current_section]["last_option"] = idx

            added_comment = {}

            # Insert preserved options
            for key, value in attributes_dict.items():
                section, attr_name = key.split("]", 1)
                section = section + "]"

                if attr_name.lower() not in VALID_OPTIONS:
                    log_message(f"  IGNORING: {section} {attr_name.strip()} => {value} (Not supported by RMS)", log_buffer)
                    continue

                if section in section_positions and section_positions[section]["last_option"] is not None:
                    insert_position = section_positions[section]["last_option"] + 1
                else:
                    new_config_lines.append(f"\n{section}\n")
                    insert_position = len(new_config_lines)

                if not new_config_lines[insert_position - 1].strip():
                    new_config_lines.insert(insert_position, "\n")

                if section not in added_comment:
                    new_config_lines.insert(
                        insert_position + 1,
                        "; The following options were preserved but are not in the template\n"
                    )
                    added_comment[section] = True

                new_config_lines.insert(insert_position + 2, f"{attr_name.strip()}: {value}\n")
                log_message(f"  PRESERVING: {section} {attr_name.strip()} => {value} (Supported by RMS)", log_buffer)

            # Write the modified config
            with open(new_config_file, "w") as newfile:
                newfile.writelines(new_config_lines)
    
    # Write log file if logging is enabled
    if log_buffer:
        log_file = os.path.join(os.path.dirname(original_config_file), f"{station_id}_MigrateConfig.log")
        with open(log_file, "w") as log:
            log.write(f"=== Migration Log: {datetime.now():%Y-%m-%d %H:%M:%S} ===\n\n")
            log.write(log_buffer.getvalue())
            log.write("\nMigration completed.\n")
        log_buffer.close()
                    
    return new_config_file, station_id

if __name__ == "__main__":
    print(f"\n{sys.argv[0]} - Migrate RMS .config to latest template standard carrying forward customizations\n")

    parser = argparse.ArgumentParser(
        description="Migrate .config to latest format carrying forward attribute customizations"
    )
    parser.add_argument(
        "-i", "--input",
        help='input .config filename, overrides default ".config"'
    )
    parser.add_argument(
        "-t", "--template",
        help='reference template filename, overrides default ".configTemplate"'
    )
    parser.add_argument(
        "-o", "--output",
        help='reformatted .config filename, overrides default "configNew_<CamId>"'
    )
    parser.add_argument(
        "-update", "--update",
        help="automatically UPDATE the active .config",
        action="store_true"
    )
    parser.add_argument(
        "-recent", "--recent",
        help="update recently changed default values",
        action="store_true"
    )
    parser.add_argument(
        "-log", "--log",
        help="enable logging to file",
        action="store_true"
    )
    parser.add_argument(
        "-compact", "--compact",
        help="create a compact config with only non-default values",
        action="store_true"
    )
    parser.add_argument(
        "-revert", "--revert",
        help="revert to original backup configuration",
        action="store_true"
    )
    args = parser.parse_args()
    
    # If reverting, we don't need the template file
    if args.revert:
        template_config_file = None
    else:
        template_config_file = os.path.expanduser("~/source/RMS/.configTemplate")
        if args.template:
            template_config_file = args.template
        print(f"\nTemplate: {template_config_file}")

    curr_path = os.getcwd()
    print(f"Current path is {curr_path}")

    template_config_file = os.path.expanduser("~/source/RMS/.configTemplate")
    if args.template:
        template_config_file = args.template
    
    print(f"\nTemplate: {template_config_file}")

    # assume default input
    original_config_files = [os.path.expanduser("~/source/RMS/.config")]

    # if multi-cam find and assume those
    if os.path.exists(os.path.expanduser("~/source/Stations")):
        original_config_files = []
        for d in os.listdir(os.path.expanduser("~/source/Stations")):
            original_config_files.append(
                os.path.join(os.path.expanduser("~/source/Stations"), d, ".config")
            )
        print(f"Multi-cam count: {len(original_config_files)}")

    # if specified assume
    if args.input:
        original_config_files = [args.input]

    # process each input config
    for orig_config in original_config_files:
        out_file_name, station_id = update_config(orig_config, template_config_file, args)

        if args.update:
            try:
                shutil.copy(out_file_name, orig_config)
                os.remove(out_file_name)
                print("Updated\n")
            except Exception as e:
                print(f"ERROR: Update failed: {e}")
                sys.exit(1)

    if not args.update:
        print("\nAfter saving a copy of the existing .config, copy/rename new version(s) into production.")
        print("    e.g. cp ConfigNew_XXxxxx .config \n")

    print("Done.\n")