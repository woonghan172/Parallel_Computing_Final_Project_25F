#!/bin/bash

#gcc -O3 serial_version.c -o serial_version -lm
nvcc cu_files/serial.cu -o serial