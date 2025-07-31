#!/usr/bin/env python3
"""
Script to manage camera brightness settings for day/night modes in camera_settings.json.

This script:
1. Reads the current camera_settings.json
2. Copies the SetColor line from 'init' section to preserve user's custom brightness
3. Adds a day-specific SetColor line with brightness=50 if not already present
4. Ensures the night section uses the original user brightness value
5. Inserts SetColor lines below the DayNightColor parameter in each section

Usage: python3 manage_camera_brightness.py [path_to_camera_settings.json]
"""

import json
import sys
import os
import re
from typing import Dict, List, Any, Optional, Tuple

def find_setcolor_command(commands: List[List[str]]) -> Optional[Tuple[int, str]]:
    """Find SetColor command in a list of commands.
    
    Returns:
        Tuple of (index, brightness_value) if found, None otherwise
    """
    for i, cmd in enumerate(commands):
        if len(cmd) >= 2 and cmd[0] == "SetColor":
            # Extract brightness (first value in the comma-separated string)
            brightness = cmd[1].split(',')[0]
            return i, brightness
    return None

def find_daynight_color_index(commands: List[List[str]]) -> Optional[int]:
    """Find the index of DayNightColor parameter in commands list.
    
    Returns:
        Index after DayNightColor command, or None if not found
    """
    for i, cmd in enumerate(commands):
        if (len(cmd) >= 4 and 
            cmd[0] == "SetParam" and 
            cmd[1] == "Camera" and 
            cmd[2] == "DayNightColor"):
            return i + 1  # Return index after DayNightColor
    return None

def create_setcolor_command(brightness: str, original_values: str = "50,50,50,0,0") -> List[str]:
    """Create a SetColor command with specified brightness.
    
    Args:
        brightness: Brightness value (0-100)
        original_values: The remaining color values from original command
        
    Returns:
        SetColor command as [command, parameters]
    """
    return ["SetColor", f"{brightness},{original_values}"]

def format_camera_settings_json(settings: Dict[str, Any]) -> str:
    """Format camera settings JSON in the original compact style with preserved blank lines.
    
    Args:
        settings: Dictionary containing camera settings
        
    Returns:
        Formatted JSON string with arrays on single lines and blank lines preserved
    """
    lines = ['{']
    
    sections = list(settings.keys())
    for i, section in enumerate(sections):
        lines.append(f'  "{section}": [')
        
        commands = settings[section]
        for j, command in enumerate(commands):
            # Format each command as a compact array on one line (spaces after commas)
            formatted_cmd = json.dumps(command, separators=(', ', ': '))
            if j < len(commands) - 1:
                lines.append(f'    {formatted_cmd},')
                
                # Add blank lines in init section at logical breaks
                if section == 'init' and j < len(commands) - 1:
                    next_cmd = commands[j + 1]
                    current_cmd = command
                    
                    # Add blank line after VideoFormat (before Encode section)
                    if (len(current_cmd) >= 4 and 
                        current_cmd[0] == "SetParam" and 
                        current_cmd[1] == "General" and 
                        current_cmd[2] == "VideoFormat"):
                        lines.append('')
                    
                    # Add blank line after SecondStream (before Camera section)  
                    elif (len(current_cmd) >= 3 and
                          current_cmd[0] == "SetParam" and
                          current_cmd[1] == "Encode" and
                          current_cmd[2] == "SecondStream"):
                        lines.append('')
            else:
                lines.append(f'    {formatted_cmd}')
        
        if i < len(sections) - 1:
            lines.append('  ],')
            lines.append('')  # Empty line between sections
        else:
            lines.append('  ]')
    
    lines.append('}')
    return '\n'.join(lines)

def compare_settings(init_commands: List[List[str]], night_commands: List[List[str]]) -> List[str]:
    """Compare night settings against init settings to find discrepancies.
    
    Args:
        init_commands: List of commands from init section
        night_commands: List of commands from night section
        
    Returns:
        List of warning messages for settings that don't match
    """
    warnings = []
    
    # Build a lookup of init settings (excluding certain commands that shouldn't match)
    init_settings = {}
    skip_commands = {"CameraTime", "reboot"}  # Commands that are expected to differ
    
    for cmd in init_commands:
        if len(cmd) >= 1 and cmd[0] not in skip_commands:
            if cmd[0] == "SetParam" and len(cmd) >= 4:
                # For SetParam commands, use a composite key
                key = f"{cmd[0]}|{cmd[1]}|{cmd[2]}"
                if len(cmd) == 4:
                    init_settings[key] = cmd[3]
                elif len(cmd) == 5:
                    init_settings[f"{key}|{cmd[3]}"] = cmd[4]
            elif cmd[0] == "SetColor":
                init_settings["SetColor"] = cmd[1]
            else:
                # For other commands, use the full command as key
                init_settings[cmd[0]] = cmd[1] if len(cmd) > 1 else ""
    
    # Check night settings against init
    for cmd in night_commands:
        if len(cmd) >= 1 and cmd[0] not in skip_commands:
            if cmd[0] == "SetParam" and len(cmd) >= 4:
                # For SetParam commands, check against init
                key = f"{cmd[0]}|{cmd[1]}|{cmd[2]}"
                if len(cmd) == 4:
                    init_value = init_settings.get(key)
                    if init_value is not None and init_value != cmd[3]:
                        warnings.append(f"Night setting {cmd[1]}.{cmd[2]} = '{cmd[3]}' differs from init value '{init_value}'")
                elif len(cmd) == 5:
                    full_key = f"{key}|{cmd[3]}"
                    init_value = init_settings.get(full_key)
                    if init_value is not None and init_value != cmd[4]:
                        warnings.append(f"Night setting {cmd[1]}.{cmd[2]}.{cmd[3]} = '{cmd[4]}' differs from init value '{init_value}'")
            elif cmd[0] == "SetColor":
                init_value = init_settings.get("SetColor")
                if init_value is not None and init_value != cmd[1]:
                    warnings.append(f"Night SetColor = '{cmd[1]}' differs from init value '{init_value}'")
    
    return warnings

def manage_camera_brightness(camera_settings_path: str) -> bool:
    """Manage brightness settings in camera_settings.json.
    
    Args:
        camera_settings_path: Path to camera_settings.json file
        
    Returns:
        True if modifications were made, False otherwise
    """
    
    if not os.path.exists(camera_settings_path):
        print(f"Error: Camera settings file not found: {camera_settings_path}")
        return False
    
    # Read the current camera settings
    try:
        with open(camera_settings_path, 'r') as f:
            settings = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading camera settings file: {e}")
        return False
    
    # Check if required sections exist
    if 'init' not in settings:
        print("Error: 'init' section not found in camera_settings.json")
        return False
    
    # Find the original SetColor command in init section
    init_setcolor = find_setcolor_command(settings['init'])
    if not init_setcolor:
        print("Warning: No SetColor command found in 'init' section")
        print("Cannot determine user's custom brightness value")
        return False
    
    init_setcolor_index, original_brightness = init_setcolor
    original_command = settings['init'][init_setcolor_index][1]  # Full parameter string
    remaining_values = ','.join(original_command.split(',')[1:])  # Everything after brightness
    
    print(f"Found original brightness setting: {original_brightness}")
    print(f"Original SetColor parameters: {original_command}")
    
    modifications_made = False
    
    # Process day section
    if 'day' in settings:
        day_commands = settings['day']
        day_setcolor = find_setcolor_command(day_commands)
        daynight_index = find_daynight_color_index(day_commands)
        
        if not day_setcolor and daynight_index is not None:
            # Add SetColor with brightness=50 for day mode
            day_setcolor_cmd = create_setcolor_command("50", remaining_values)
            day_commands.insert(daynight_index, day_setcolor_cmd)
            print("Added day brightness setting (brightness=50)")
            modifications_made = True
        elif day_setcolor:
            day_brightness = day_setcolor[1]
            print(f"Day section already has SetColor with brightness={day_brightness}")
        else:
            print("Warning: Could not find DayNightColor in day section to insert SetColor")
    else:
        print("Warning: 'day' section not found in camera_settings.json")
    
    # Process night section
    if 'night' in settings:
        night_commands = settings['night']
        night_setcolor = find_setcolor_command(night_commands)
        daynight_index = find_daynight_color_index(night_commands)
        
        if not night_setcolor and daynight_index is not None:
            # Add SetColor with original brightness for night mode
            night_setcolor_cmd = create_setcolor_command(original_brightness, remaining_values)
            night_commands.insert(daynight_index, night_setcolor_cmd)
            print(f"Added night brightness setting (brightness={original_brightness})")
            modifications_made = True
        elif night_setcolor:
            night_brightness = night_setcolor[1]
            if night_brightness != original_brightness:
                print(f"Note: Night brightness ({night_brightness}) differs from init brightness ({original_brightness})")
            else:
                print(f"Night section already has correct SetColor with brightness={night_brightness}")
        else:
            print("Warning: Could not find DayNightColor in night section to insert SetColor")
    else:
        print("Warning: 'night' section not found in camera_settings.json")
    
    # Check for night/init setting discrepancies
    if 'night' in settings:
        setting_warnings = compare_settings(settings['init'], settings['night'])
        if setting_warnings:
            print("\n⚠️  Configuration Warnings:")
            print("Night settings that don't match init conditions:")
            for warning in setting_warnings:
                print(f"  • {warning}")
            print("\nThese settings may prevent the camera from returning to init state during night mode.")
            print("Consider reviewing these discrepancies to ensure proper day/night transitions.")
        else:
            print("Night settings properly restore init conditions (excluding expected differences)")
    
    # Write back the modified settings if changes were made
    if modifications_made:
        try:
            # Create backup
            backup_path = camera_settings_path + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Created backup: {backup_path}")
            
            # Write modified settings with original compact format
            with open(camera_settings_path, 'w') as f:
                f.write(format_camera_settings_json(settings))
            print(f"Updated camera settings: {camera_settings_path}")
            
        except IOError as e:
            print(f"Error writing camera settings file: {e}")
            return False
    else:
        print("No modifications needed - brightness settings already configured correctly")
    
    return modifications_made

def main():
    """Main function to handle command line arguments and run the brightness management."""
    
    # Default path to camera_settings.json
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'camera_settings.json')
    
    # Use provided path or default
    if len(sys.argv) > 1:
        camera_settings_path = sys.argv[1]
    else:
        camera_settings_path = default_path
    
    print(f"Managing camera brightness settings in: {camera_settings_path}")
    print("=" * 60)
    
    success = manage_camera_brightness(camera_settings_path)
    
    if success:
        print("=" * 60)
        print("Camera brightness management completed successfully!")
        print()
        print("Summary:")
        print("- Day mode: Uses brightness=50 for better daytime visibility")
        print("- Night mode: Uses your original brightness setting for optimal night capture")
        print("- Original settings preserved in .backup file")
    else:
        print("=" * 60)
        print("Camera brightness management completed with warnings or errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()