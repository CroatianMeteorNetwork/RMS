#!/bin/bash
# Kept for compatibility with existing documentation and older images.
# The station tooling is now universal - see add_Station.sh.
exec "$(dirname "$0")/../../add_Station.sh" "$@"
