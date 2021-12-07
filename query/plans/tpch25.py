# select l_linenumber, sum(CAST(l_suppkey AS bigint))
# from lineitem
# group by l_linenumber
# order by l_linenumber
alg.projection (
    [ "l_linenumber", "sum_l_suppkey"],
    alg.aggregation (
        [ "l_linenumber" ], # group by ...
        [( Reduction.SUM, "l_suppkey", "sum_l_suppkey" )], # aggragation function: sum, count...
        alg.scan ( "lineitem" )
    )
)

# alg.aggregation (
#     [ "l_linenumber", "sum_l_quantity" ], # group by ...
#     [( Reduction.SUM, "l_quantity", "sum_l_quantity" )], # aggragation function: sum, count...
#     alg.scan ( "lineitem" )
# )