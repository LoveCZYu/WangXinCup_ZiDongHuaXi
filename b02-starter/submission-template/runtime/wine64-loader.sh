#!/bin/sh
set -eu
# The official serve.exe is AMD64. Do not let Debian prefer wine32 when both exist.
export WINELOADER=/usr/lib/wine/wine64
exec "$WINELOADER" "$@"
