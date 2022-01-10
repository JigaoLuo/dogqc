# select l_suppkey, l_orderkey, count(l_partkey), count(l_linenumber), count(*)
# from lineitem
# group by l_suppkey, l_orderkey
# order by l_suppkey, l_orderkey

alg.projection (
    [ "l_suppkey", "l_orderkey", "count_l_partkey", "count_l_linenumber", "count_l_suppkey_orderkey"],
    alg.aggregation (
        [ "l_suppkey", "l_orderkey" ], # group by ...
        [( Reduction.COUNT, "l_partkey", "count_l_partkey" ),
         ( Reduction.COUNT, "l_linenumber", "count_l_linenumber" ),
         ( Reduction.COUNT, "", "count_l_suppkey_orderkey" )], # aggragation function: sum, count...
        alg.scan ( "lineitem" )
    )
)
