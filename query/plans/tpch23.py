# select l_linenumber, count(*) from lineitem group by l_linenumber
alg.projection (
    [ "l_linenumber, count_l_linenumber"],
    alg.aggregation (
        [ "l_linenumber" ],
        [( Reduction.COUNT, "", "count_l_linenumber" )],
        alg.scan ( "lineitem" )
    )
)

# alg.aggregation (
#     [ "l_linenumber" ], # group by ...
#     [],                 # aggragation function: sum, count...
#     alg.scan ( "lineitem" )
# )