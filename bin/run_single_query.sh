#!/usr/bin/env bash

tpch_tbl_files=/home/jigao/Desktop/tpch-dbgen

function run_single_query {
  # echo $tpch_tbl_files
  # echo $1
  python3 ../query/tpch.py $tpch_tbl_files $1
}

run_single_query $1