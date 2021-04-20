"""
https://www.kdnuggets.com/2021/01/model-experiments-tracking-registration-mlflow-databricks.html
"""
# Import required libraries
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.types import FloatType

import mlflow
import mlflow.spark
import mlflow.tracking

mlflow.set_experiment('/Users/dash@streamsets.com/transformer-experiments')
mlflow_client = mlflow.tracking.MlflowClient()

# Setup variables for convenience and readability 
trainSplit = ${trainSplit}
testSplit = ${testSplit}
maxIter = ${maxIter}
numberOfCVFolds = ${numberOfCVFolds}
r2 = 0
rmse = 0
stage = "Staging"

# The input dataframe is accessbile via inputs[0]
df = inputs[0]

features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']

# MUST for Spark features
vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
df = vectorAssembler.transform(df)

# Split dataset into "train" and "test" sets
(train, test) = df.randomSplit([trainSplit, testSplit], 42) 

# Setup evaluator -- default is F1 score
classEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")

with mlflow.start_run(): 
  # Gradient-boosted tree regression
  gbt = GBTRegressor(maxIter=maxIter)

  # Setup pipeline
  pipeline = Pipeline(stages=[gbt])

  # Setup hyperparams grid
  paramGrid = ParamGridBuilder().build()

  # Setup model evaluators
  rmseevaluator = RegressionEvaluator() #Note: By default, it will show how many units off in the same scale as the target -- RMSE
  r2evaluator = RegressionEvaluator(metricName="r2") #Select R2 as our main scoring metric

  # Setup cross validator
  cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=r2evaluator, numFolds=numberOfCVFolds) 

  # Fit model on "train" set
  cvModel = cv.fit(train)

  # Get the best model based on CrossValidator
  model = cvModel.bestModel

  # Run inference on "test" set
  predictions = model.transform(test)

  rmse = rmseevaluator.evaluate(predictions)
  r2 = r2evaluator.evaluate(predictions)

  mlflow.log_param("transformer-pipeline-id","${pipeline:id()}")
  
  mlflow.log_param("features", features)
  mlflow.log_param("maxIter_hyperparam", maxIter)
  mlflow.log_param("numberOfCVFolds_hyperparam", numberOfCVFolds)
  mlflow.log_metric("rmse_metric_param", rmse)
  mlflow.log_metric("r2_metric_param", r2)
  
  # Log and register the model
  mlflow.spark.log_model(spark_model=model, artifact_path="SparkML-GBTRegressor-model", registered_model_name="SparkML-GBTRegressor-model")

mlflow.end_run()

# Transition the current model to 'Staging' or 'Production'
current_version = mlflow_client.search_model_versions('name="SparkML-GBTRegressor-model"')[0].version
while mlflow_client.search_model_versions('name="SparkML-GBTRegressor-model"')[0].status != 'READY':
  current_version = current_version

if (r2 >= ${r2Threshold} or rmse <= ${rmseThreshold}):
  stage = "Production"

mlflow_client.transition_model_version_stage(name="SparkML-GBTRegressor-model",stage=stage,version=current_version)

output = inputs[0]