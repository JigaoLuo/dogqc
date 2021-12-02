# select l_linenumber from lineitem group by l_linenumber
alg.projection (
    [ "l_linenumber"],
    [],
    alg.aggregation (
        [ "l_linenumber" ],
        alg.scan ( "lineitem" )
    )
)

# alg.aggregation (
#     [ "l_linenumber" ], # group by ...
#     [],                 # aggragation function: sum, count...
#     alg.scan ( "lineitem" )
# )