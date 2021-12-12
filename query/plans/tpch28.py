# select l_suppkey, count(*)
# from lineitem
# group by l_suppkey
# order by l_suppkey

alg.projection (
    [ "l_suppkey", "count_l_suppkey"],
    alg.aggregation (
        [ "l_suppkey" ], # group by ...
        [( Reduction.COUNT, "", "count_l_suppkey" )], # aggragation function: sum, count...
        alg.scan ( "lineitem" )
    )
)