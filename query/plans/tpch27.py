# select l_orderkey, count(*)
# from lineitem
# group by l_orderkey
# order by l_orderkey
alg.projection (
    [ "l_orderkey", "count_l_orderkey"],
    alg.aggregation (
        [ "l_orderkey" ], # group by ...
        [( Reduction.COUNT, "", "count_l_orderkey" )], # aggragation function: sum, count...
        alg.scan ( "lineitem" )
    )
)