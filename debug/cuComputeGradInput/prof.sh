#!/bin/sh

OUT_DIR=out

mkdir -p $OUT_DIR
echo "pmc: L2CacheHit" > input.txt
rocprof -i input.txt --obj-tracking on --timestamp on --stats -o out/res.csv $1

# MEMORY CONFLICT
MEM_PMC="SQ_LDS_BANK_CONFLICT LDSBankConflict"
echo "pmc: $MEM_PMC" > /tmp/input.txt
rocprof -i /tmp/input.txt --timestamp on -o $OUT_DIR/mem.csv $1

echo "$1:"
grep $1 out/res.stats.csv | awk -F, '{ print "avgDurationNs: ", $(NF-1) }'
