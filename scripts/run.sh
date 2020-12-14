#!/bin/bash

helpFunction() {
  echo ""
  echo "Usage: $0 -c script -l log"
  echo -e "\t-c the command to run."
  echo -e "\t-l the log file."
  exit 1
}

while getopts "c:l:" opt; do
  case "$opt" in
  c) script="$OPTARG" ;;
  l) log="$OPTARG" ;;
  ?) helpFunction ;;
  esac
done

if [ -z "$script" ] || [ -z "$log" ]; then
  echo "Some or all of the parameters are empty"
  helpFunction
fi

nohup $script >$log 2>&1 &
