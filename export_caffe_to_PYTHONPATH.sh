#!/bin/bash

THISSCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ":$PYTHONPATH:" == *":$THISSCRIPTPATH/python:"* ]]; then
	echo "$THISSCRIPTPATH/python is already in \$PYTHONPATH"
else
	echo "Adding $THISSCRIPTPATH/python to \$PYTHONPATH"
	export PYTHONPATH=$THISSCRIPTPATH/python:$PYTHONPATH
fi
