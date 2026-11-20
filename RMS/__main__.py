""" Umbrella launcher for RMS tools.

Run "python -m RMS" to list the available tools, and "python -m RMS <command> [args...]" to run one.
Every command is a thin alias for a module which can also be run directly, e.g.
"python -m RMS skyfit ." is the same as "python -m Utils.SkyFit2 .".
"""

from __future__ import print_function, division, absolute_import

import runpy
import sys


# Registry of subcommands: (command, module, one-line description), grouped by topic.
# Keep the descriptions short - they are printed as a single line each.
COMMANDS = [
    ("Station operation", [
        ("start", "RMS.StartCapture", "Start the capture pipeline (recording, detection, upload)"),
        ("reprocess", "RMS.Reprocess", "Reprocess a night directory from raw FF files"),
        ("status", "Utils.StationStatus", "Show the station health report from the last night"),
        ("capture-duration", "RMS.CaptureDuration", "Print the capture start time and duration for tonight"),
        ("delete-old", "RMS.DeleteOldObservations", "Free up disk space by deleting old observations"),
    ]),
    ("Calibration & astrometry", [
        ("skyfit", "Utils.SkyFit2", "Astrometric calibration and manual reduction GUI"),
        ("calibration-report", "Utils.CalibrationReport", "Generate a calibration quality report for a night"),
        ("flat", "Utils.MakeFlat", "Make a flat field image from a night of data"),
        ("shower-association", "Utils.ShowerAssociation", "Associate detected meteors with showers"),
    ]),
    ("Camera setup", [
        ("camera-control", "Utils.CameraControl", "Get/set IP camera parameters"),
        ("camera-manager", "Utils.CamManager", "Find and configure cameras on the network (GUI)"),
        ("camera-address", "Utils.SetCameraAddress", "Change the camera IP address"),
        ("camera-params", "Utils.setAllCameraParams", "Apply the standard RMS camera settings"),
        ("livestream", "Utils.ShowLiveStream", "Show the live video stream from the camera"),
    ]),
    ("Viewing & media", [
        ("liveview", "Utils.LiveViewer", "Slideshow of FF files as they are created"),
        ("frbin", "Utils.FRbinViewer", "View fireball detections from FR bin files"),
        ("checknight", "Utils.CheckNight", "Visually inspect the images of a night"),
        ("stack", "Utils.StackFFs", "Stack FF files into a single image"),
        ("trackstack", "Utils.TrackStack", "Star-aligned stack of a whole night"),
        ("timelapse", "Utils.GenerateTimelapse", "Make a timelapse video of a night"),
        ("mp4s", "Utils.GenerateMP4s", "Make MP4 clips of individual detections"),
        ("thumbnails", "Utils.GenerateThumbnails", "Make a thumbnail grid of a night"),
    ]),
    ("Configuration", [
        ("audit-config", "Utils.AuditConfig", "Compare the station .config against the current template"),
        ("migrate-config", "Utils.MigrateConfig", "Upgrade the .config to the latest template format"),
    ]),
    ("Analysis", [
        ("flux", "Utils.Flux", "Compute single-station meteor shower flux"),
        ("fov-kml", "Utils.FOVKML", "Make a Google Earth KML of the camera field of view"),
        ("fov-skymap", "Utils.FOVSkyMap", "Plot the camera field of view on a sky map"),
    ]),
]


def printToolList():

    print(__doc__)

    for group_name, entries in COMMANDS:

        print("{:s}:".format(group_name))

        for command, module, description in entries:
            print("  {:<20s} {:s}".format(command, description))

        print()

    print("Run \"python -m RMS <command> -h\" for the arguments of an individual tool.")
    print("Tools not listed here can be run directly as modules - see Utils/README.md for a full catalog.")


def main():

    args = sys.argv[1:]

    # No command, an explicit list request, or a help flag - print the tool list
    if (not args) or (args[0] in ("tools", "list", "-h", "--help", "help")):
        printToolList()
        return

    command = args[0]

    # Find the command in the registry
    module = None
    for _, entries in COMMANDS:
        for cmd_name, cmd_module, _ in entries:
            if command == cmd_name:
                module = cmd_module
                break

    if module is None:
        print("Unknown command: {:s}".format(command))
        print("Run \"python -m RMS\" to list the available commands.")
        sys.exit(2)

    # Hand over to the target module as if it was run with python -m
    sys.argv = [module] + args[1:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
