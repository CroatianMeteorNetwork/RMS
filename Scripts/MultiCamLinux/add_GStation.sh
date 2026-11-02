#!/bin/bash
# Kept for compatibility with existing documentation.
# The station tooling is now universal - see add_Station.sh.
exec "$(dirname "$(readlink -f "$0")")/../add_Station.sh" "$@"
