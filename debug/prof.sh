#!/bin/sh

mkdir -p out
echo "pmc: L2CacheHit" > input.txt
rocprof -i input.txt --timestamp on -o out/res.csv python layer_norm_inference.py
