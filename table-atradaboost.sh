#!/usr/bin/env bash

./eval-atradaboost.py $1 $2 # generator seed
sort -n -t',' -k7 $1/gain-pro-report.txt | column -s',' -t | less
