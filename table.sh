#!/usr/bin/env bash

./eval-params.py $1 $2 # exp_code boost_mode

# reverse sort by last column
awk -F, '{print $NF,$0}' $1/gain-pro-report.txt | sort -nr -t',' | cut -f2- -d',' | column -s',' -t | less

