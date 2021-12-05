# Query Plans

## Number of queries

The number of queries is hardcoded in  `query/tpch.py`.

```python
    if sys.argv[2] == "all":
        for i in range(1,23 + 1):
            print ( "-----------------------Executing TPCH-H query " + str(i) + "-----------------------" )
            execTpch ( acc, i )
```

## Parser

Query plans are parsed by `query/tpch.py`

```python
qPath = "../query/plans/tpch" + str(num) + ".py"
# read query plan
plan = eval ( loadScript ( qPath ) )
```

## How to create a new plan?

### SELECT

```sql
select a, b
from t
```

```python
alg.projection (
    [ "a", "b" ],
    alg.scan ( "t" )
)
```

### Aggregation

```sql
alg.aggregation ( 
    [],
    [ ( Reduction.SUM, "rev", "revenue" ),
      ( Reduction.COUNT, "rev", "count" ) ],
    alg.map (
        "rev",
        scal.MulExpr ( 
            scal.AttrExpr ( "l_extendedprice" ),
            scal.AttrExpr ( "l_discount" )
        ),
        alg.selection (
            scal.AndExpr (
                scal.LargerEqualExpr (
                    scal.AttrExpr ( "l_shipdate" ),
                    scal.ConstExpr ( "19940101", Type.DATE )
                ),
                scal.AndExpr (
                    scal.SmallerExpr (
                        scal.AttrExpr ( "l_shipdate" ),
                        scal.ConstExpr ( "19950101", Type.DATE )
                    ),
                    scal.AndExpr (
                        scal.LargerEqualExpr (
                            scal.AttrExpr ( "l_discount" ),
                            scal.ConstExpr ( "0.05", Type.DOUBLE )
                        ),
                        scal.AndExpr (
                            scal.SmallerEqualExpr (
                                scal.AttrExpr ( "l_discount" ),
                                scal.ConstExpr ( "0.07", Type.DOUBLE )
                            ),
                            scal.SmallerExpr (
                                scal.AttrExpr ( "l_quantity" ),
                                scal.ConstExpr ( "24", Type.DOUBLE )
                            )
                        )
                    )
                )
            ),
            alg.scan ( "lineitem" )
        )
    )
)
```