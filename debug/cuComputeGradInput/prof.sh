#!/bin/sh

mkdir -p out
echo "pmc: L2CacheHit" > input.txt
rocprof -i input.txt --obj-tracking on --timestamp on --stats -o out/res.csv $1
echo "$1:"
grep $1 out/res.stats.csv | awk -F, '{ print "avgDurationNs: ", $(NF-1) }'
