Helper scripts for running a meteor station. All of them assume that RMS is installed in `~/source/RMS`.

- Most scripts here (`RMS_FirstRun.sh`, `RMS_StartCapture.sh`, `RMS_Update.sh`, ...) are linked on the desktop of a station and run from there.
- `RMS_Installer.sh` installs RMS and everything it needs on a generic Linux machine (Ubuntu LTS or Debian).
- `add_Station.sh` creates camera stations under `~/source/Stations/` — interactively, or in bulk from a CSV file (see `stations_template.csv` in the repository root). It also converts a legacy single-camera setup to that layout (by copying — nothing in the RMS root is modified). Works on any Linux platform, Raspberry Pi included.
- `RMS_StartCapture.sh` starts capture on either data structure: every station under `~/source/Stations/` in its own terminal, or the single legacy camera if there are none.
- `MultiCamLinux/` holds the per-station runtime launchers (`StartCapture.sh`, `LiveStream.sh`) plus compatibility wrappers for the older `add_GStation.sh`, `add_Pi_Station.sh` and `Pi/RMS_StartCapture_MCP.sh` entry points.
