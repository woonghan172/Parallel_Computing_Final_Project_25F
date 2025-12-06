#!/bin/bash

gcc -O3 serial_version.c -o serial_version -lm

nvcc -O3 -o shared_memory shared_memory.cu