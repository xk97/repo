from pyspark import SparkContext
# Bazic reduceByKey example in python
# creating PairRDD x with key value pairs

arr = sc.parallelize(range(10))
arr.map(lambda x: x%2).zip(arr).reduceByKey(lambda x, y: x + y).collect()
# [(0, 20), (1, 25)]

x = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("a", 1),
                    ("b", 1), ("b", 1), ("b", 1), ("b", 1)], 3)

# Applying reduceByKey operation on x
y = x.reduceByKey(lambda accum, n: accum + n)
print(y.collect())
# [('b', 5), ('a', 3)]

# Define associative function separately
def sumFunc(accum, n):
    return accum + n

y = x.reduceByKey(sumFunc)
print(y.collect())
# [('b', 5), ('a', 3)]
