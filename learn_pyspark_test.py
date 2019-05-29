import pyspark
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('data_processing').getOrCreate()
df=spark.read.csv('../data/sales-of-shampoo-over-a-three-ye.csv',
                  inferSchema=True, header=True)
df.columns
df.printSchema()
df.describe().show()
df.withColumn('total', df[df.columns[-1]]).show(3)
df = df.withColumn('time', df[df.columns[0]]).withColumn('total', df[df.columns[-1]])
df.filter(df['total'] > 200).select(['time', 'TIME', 'total']).show(5)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,DoubleType

split_col = pyspark.sql.functions.split(df['time'], '-')
df = df.withColumn('year', split_col.getItem(0))
df = df.withColumn('month', split_col.getItem(1))
df.withColumn('year_double',df['year'].cast(DoubleType())).show(10,False)
df.select('month').show(5)
df.select('year').distinct().show(5)
df.select(['year', 'total']).groupBy('year').mean().show(5,False)
df.show(5)
df_new = df.drop('Sales of shampoo over a three year period').dropna()
df_new = df_new.withColumn('year_double', df_new['year'].cast(DoubleType()))#.show(5)
df_new = df_new.withColumn('month_double', df_new['month'].cast(DoubleType()))#.show(5)

if 0:
    print('do pandas udf\n')
    from pyspark.sql.functions import pandas_udf
    def prod(month, year):
        return 12 * (year - 1.0) + month
    prod_udf = pandas_udf(prod, DoubleType())
    df_new.withColumn('prod', prod_udf(df_new['month_double'], df_new['year_double'])).show(5)
    
    df.coalesce(1).write.format('csv').option('header', 'true').save('../data/sample_csv')
    df_new.dropna().write.format('parquet').save('../data/parquet_uri')
df_new.show(5)

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
vec_assembler = VectorAssembler(inputCols=['month_double', 'year_double'], outputCol='features')
df_new.printSchema()
print(df_new.count(), len(df_new.columns))
df_feature = vec_assembler.transform(df_new)
df_feature.show(5)
df_train, df_test = df_feature.randomSplit([0.7,0.3], seed=42)

from pyspark.ml.regression import LinearRegression
lin_reg = LinearRegression(labelCol='total')
lr_model = lin_reg.fit(df_train)
print(lr_model.coefficients, '\n', lr_model.intercept)
train_prediction = lr_model.evaluate(df_train)
print(train_prediction.r2, train_prediction.meanAbsoluteError)

test_prediction = lr_model.evaluate(df_test)
print(test_prediction.r2, test_prediction.meanAbsoluteError)
