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
import shutil
import re
from typing import Dict, List, Any, Optional, Tuple

SKIP_COMPARE_PREFIXES = {
    "CameraTime",
    "reboot",
    "SetParam|Camera|DayNightColor",
}

def getCommandKey(cmd: List[str]) -> str:
    """Generate a unique key for a command for comparison purposes."""
    if not cmd:
        return ""
    if cmd[0] == "SetParam":
        if len(cmd) >= 5:   # SetParam, Section, Group, Name, Value
            return f"SetParam|{cmd[1]}|{cmd[2]}|{cmd[3]}"
        elif len(cmd) == 4: # SetParam, Section, Name, Value
            return f"SetParam|{cmd[1]}|{cmd[2]}"
        return "SetParam"
    if cmd[0] == "SetColor":
        return "SetColor"
    return cmd[0]

def shouldSkipComparison(cmd: List[str]) -> bool:
    """Check if a command should be skipped during comparison."""
    k = getCommandKey(cmd)
    return k in SKIP_COMPARE_PREFIXES or any(k.startswith(pref) for pref in SKIP_COMPARE_PREFIXES)

def findSetColorCommand(commands: List[List[str]]) -> Optional[Tuple[int, str]]:
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

def findDayNightColorIndex(commands: List[List[str]]) -> Optional[int]:
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

def createSetColorCommand(brightness: str, original_values: str = "50,50,50,0,0") -> List[str]:
    """Create a SetColor command with specified brightness.
    
    Args:
        brightness: Brightness value (0-100)
        original_values: The remaining color values from original command
        
    Returns:
        SetColor command as [command, parameters]
    """
    return ["SetColor", f"{brightness},{original_values}"]

def formatCameraSettingsJson(settings: Dict[str, Any]) -> str:
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

def compareSettings(init_commands: List[List[str]], day_commands: List[List[str]], night_commands: List[List[str]]) -> List[str]:
    """Check that night properly resets parameters that day modified from init."""
    warnings: List[str] = []

    def buildMap(cmds: List[List[str]]) -> Dict[str, List[str]]:
        m: Dict[str, List[str]] = {}
        for c in cmds:
            if not shouldSkipComparison(c):
                m[getCommandKey(c)] = c
        return m

    init_map = buildMap(init_commands)
    day_map = buildMap(day_commands)
    night_map = buildMap(night_commands)

    def desc(c: Optional[List[str]]) -> str:
        if not c:
            return "not set"
        if c[0] == "SetParam":
            if len(c) >= 5:
                return f"{c[1]}.{c[2]}.{c[3]} = '{c[4]}'"
            elif len(c) == 4:
                return f"{c[1]}.{c[2]} = '{c[3]}'"
        if c[0] == "SetColor":
            return f"SetColor = '{c[1]}'"
        return " ".join(c)

    # Find parameters that day changes from init
    day_changes = {}
    for k in day_map:
        if any(k.startswith(pref) for pref in SKIP_COMPARE_PREFIXES):
            continue
        init_val = init_map.get(k)
        day_val = day_map.get(k)
        if init_val != day_val:
            day_changes[k] = (init_val, day_val)
    
    # Check if night properly resets those day-changed parameters back to init
    for k, (init_val, day_val) in day_changes.items():
        night_val = night_map.get(k)
        
        # Night should either reset to init value OR not set it at all (letting init value persist)
        if night_val is not None and night_val != init_val:
            warnings.append(f"Day changes {desc(init_val)} to {desc(day_val)}, but night sets {desc(night_val)} instead of restoring init value")
    
    # Also warn about night settings that differ from init but weren't changed by day
    for k in night_map:
        if any(k.startswith(pref) for pref in SKIP_COMPARE_PREFIXES):
            continue
        if k not in day_changes:  # This parameter wasn't changed by day
            init_val = init_map.get(k)
            night_val = night_map.get(k)
            if init_val != night_val:
                warnings.append(f"Night unnecessarily sets {desc(night_val)} (init: {desc(init_val)}, day doesn't change it)")

    return warnings

def manageCameraBrightness(camera_settings_path: str) -> bool:
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
    init_setcolor = findSetColorCommand(settings['init'])
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
        day_setcolor = findSetColorCommand(day_commands)
        daynight_index = findDayNightColorIndex(day_commands)
        
        if not day_setcolor and daynight_index is not None:
            # Add SetColor with brightness=50 for day mode
            day_setcolor_cmd = createSetColorCommand("50", remaining_values)
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
        night_setcolor = findSetColorCommand(night_commands)
        daynight_index = findDayNightColorIndex(night_commands)
        
        if not night_setcolor and daynight_index is not None:
            # Add SetColor with original brightness for night mode
            night_setcolor_cmd = createSetColorCommand(original_brightness, remaining_values)
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
    
    # Check for proper day/night transitions
    if 'day' in settings and 'night' in settings:
        setting_warnings = compareSettings(settings['init'], settings['day'], settings['night'])
        if setting_warnings:
            print("\n⚠️  Configuration Warnings:")
            print("Day/Night transition issues detected:")
            for warning in setting_warnings:
                print(f"  • {warning}")
            print("\nThese issues may prevent proper day/night transitions.")
            print("Night should reset any parameters that day modifies back to their init values.")
        else:
            print("✓ Day/night transitions look correct - night properly resets day's changes")
    
    # Write back the modified settings if changes were made
    if modifications_made:
        try:
            # Create backup
            backup_path = camera_settings_path + '.backup'
            shutil.copy2(camera_settings_path, backup_path)
            print(f"Created backup: {backup_path}")
            
            # Write modified settings with original compact format
            with open(camera_settings_path, 'w') as f:
                f.write(formatCameraSettingsJson(settings))
            print(f"Updated camera settings: {camera_settings_path}")
            
        except IOError as e:
            print(f"Error writing camera settings file: {e}")
            return False
    else:
        print("No modifications needed - brightness settings already configured correctly")
    
    return True  # Return True for success (whether modified or not)

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
    
    success = manageCameraBrightness(camera_settings_path)
    
    print("=" * 60)
    if success:
        print("Camera brightness management completed successfully!")
        sys.exit(0)
    else:
        print("Camera brightness management failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()