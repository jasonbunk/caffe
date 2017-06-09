#!/bin/bash

THISSCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ":$PYTHONPATH:" == *":$THISSCRIPTPATH/python:"* ]]; then
	echo "$THISSCRIPTPATH/python is already in \$PYTHONPATH"
else
	echo "Adding $THISSCRIPTPATH/python to \$PYTHONPATH"
	export PYTHONPATH=$THISSCRIPTPATH/python:$PYTHONPATH
fi

if ! [[ :$LIBRARY_PATH: == *":$THISSCRIPTPATH/build/lib:"* ]] ; then
  export LIBRARY_PATH=$THISSCRIPTPATH/build/lib:$LIBRARY_PATH
fi
if ! [[ :$LD_LIBRARY_PATH: == *":$THISSCRIPTPATH/build/lib:"* ]] ; then
  export LD_LIBRARY_PATH=$THISSCRIPTPATH/build/lib:$LD_LIBRARY_PATH
fi

