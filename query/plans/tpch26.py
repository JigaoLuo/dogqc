# select l_linenumber, sum(l_quantity) from lineitem group by l_linenumber
alg.projection (
    [ "l_linenumber", "sum_l_quantity"],
    alg.aggregation (
        [ "l_linenumber" ], # group by ...
        [( Reduction.SUM, "l_quantity", "sum_l_quantity" )], # aggragation function: sum, count...
        alg.scan ( "lineitem" )
    )
)

# alg.aggregation (
#     [ "l_linenumber", "sum_l_quantity" ], # group by ...
#     [( Reduction.SUM, "l_quantity", "sum_l_quantity" )], # aggragation function: sum, count...
#     alg.scan ( "lineitem" )
# )