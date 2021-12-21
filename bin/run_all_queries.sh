#!/usr/bin/env bash

tpch_tbl_files=/home/jigao/Desktop/tpch-dbgen
#tpch_tbl_files=/home/jigao/Desktop/tpch-dbgen-sf10
# echo $tpch_tbl_files
python3 ../query/tpch.py $tpch_tbl_files all