# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:06:32 2018

@author: ccai
"""
import numpy as np
hour  =  ["%02d:00"  %  i  for  i  in  range(0,  24,  3)]
day  =  ["Mon",  "Tue",  "Wed",  "Thu",  "Fri",  "Sat",  "Sun"]
features  =    day  +  hour

"{Mon}, {Tue}".format(**{_: i+1 for i, _ in enumerate(day)})

x = list(range(10))
print(x)
y = [x, x]
x = np.power(x, 2)
x = []
#%%
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]
#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data1[Target]

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict
#%%
import matplotlib.pyplot as plt
from numpy.random import random

colors = ['b', 'c', 'y', 'm', 'r']

lo = plt.scatter(random(10), random(10), marker='x', color=colors[0])
ll = plt.scatter(random(10), random(10), marker='o', color=colors[0])
l  = plt.scatter(random(10), random(10), marker='o', color=colors[1])
a  = plt.scatter(random(10), random(10), marker='o', color=colors[2])
h  = plt.scatter(random(10), random(10), marker='o', color=colors[3])
hh = plt.scatter(random(10), random(10), marker='o', color=colors[4])
ho = plt.scatter(random(10), random(10), marker='x', color=colors[4])

plt.legend((lo, ll, l, a, h, hh, ho),
           ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)

plt.show()
#%%
import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."
#
#    if x.size < window_len:
#        raise ValueError, "Input vector needs to be bigger than window size."
#
#
#    if window_len<3:
#        return x


#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y




from numpy import *
from pylab import *

def smooth_demo():

    t=linspace(-4,4,100)
    x=sin(t)
    xn=x+randn(len(t))*0.1
    y=smooth(x)

    ws=31

    subplot(211)
    plot(ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    hold(True)
    for w in windows[1:]:
        eval('plot('+w+'(ws) )')

    axis([0,30,0,1.1])

    legend(windows)
    title("The smoothing windows")
    subplot(212)
    plot(x)
    plot(xn)
    for w in windows:
        plot(smooth(xn,10,w))
    l=['original signal', 'signal with noise']
    l.extend(windows)

    legend(l)
    title("Smoothing a noisy signal")
    show()


if __name__=='__main__':
    smooth_demo()
    
#%%
x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.8

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plot(x, y,'o')
plot(x, smooth(y,3), 'r-', lw=2)
plot(x, smooth(y,19), 'g-', lw=2)
#%%
from scipy import signal
sig = np.repeat([0., 1., 0.], 100)
win = signal.hann(50)
filtered = signal.convolve(sig, win, mode='same') / sum(win)
plt.plot(win)
plt.plot(sig)
plt.plot(filtered)

#%%
from pyspark.sql import SparkSession
ss = SparkSession.builder.appName('abc').getOrCreate()
from pyspark.conf import SparkConf
SparkSession.builder.config(conf=SparkConf())



#%% https://github.com/vishwajeet97/Cocktail-Party-Problem
from scipy.io import wavfile
rate1, data1 = wavfile.read('../data/X_rsm2.wav')

plt.plot(range(data1.shape[0]), data1[:, 0])
plt.plot(range(data1.shape[0]), data1[:, 1])
plt.title((rate1, data1.shape))
x1 = pd.DataFrame(data1[:200]).melt()
plt.scatter(x1.index, x1.value, c=x1.variable, cmap=plt.cm.jet)
print(data1[-5:])

#%%
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
fpath = r'C:\Users\cyret\Documents\Python Scripts\data'
series = read_csv(os.path.join(fpath, 'sales-of-shampoo-over-a-three-ye.csv'),
                  header=0, parse_dates=[0], index_col=0, squeeze=True, 
                  date_parser=None).dropna()

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)

# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()

#%%

import unittest


class Testing(unittest.TestCase):
    def test_string(self):
        a = 'some'
        b = 'some '
        self.assertEqual(a, b)

    def test_boolean(self):
        a = True
        b = True
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()
#%% Linear programming A*x <= b, A is matrix coef, b is number of bound
#    https://www.jianshu.com/p/9be417cbfebb
import numpy as np

z = np.array([2, 3, 1])
a = np.array([[1, 4, 2], [3, 2, 0]])
b = np.array([8, 6])
x1_bound = x2_bound = x3_bound =(0, None)

from scipy import optimize
res = optimize.linprog(z, A_ub=-a, b_ub=-b,bounds=(x1_bound, x2_bound, x3_bound))
# a_ub, b_ub -> bound, a_eq, b_eq -> equal

print(res)

#%%

import unittest

# This is the class we want to test. So, we need to import it
import Person as PersonClass

class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    person = PersonClass.Person()  # instantiate the Person Class
    user_id = []  # variable that stores obtained user_id
    user_name = []  # variable that stores person name

    # test case function to check the Person.set_name function
    def test_0_set_name(self):
        print("Start set_name test\n")
        """
        Any method which starts with ``test_`` will considered as a test case.
        """
        for i in range(4):
            # initialize a name
            name = 'name' + str(i)
            # store the name into the list variable
            self.user_name.append(name)
            # get the user id obtained from the function
            user_id = self.person.set_name(name)
            # check if the obtained user id is null or not
            self.assertIsNotNone(user_id)  # null user id will fail the test
            # store the user id to the list
            self.user_id.append(user_id)
        print("user_id length = ", len(self.user_id))
        print(self.user_id)
        print("user_name length = ", len(self.user_name))
        print(self.user_name)
        print("\nFinish set_name test\n")

    # test case function to check the Person.get_name function
    def test_1_get_name(self):
        print("\nStart get_name test\n")
        """
        Any method that starts with ``test_`` will be considered as a test case.
        """
        length = len(self.user_id)  # total number of stored user information
        print("user_id length = ", length)
        print("user_name length = ", len(self.user_name))
        for i in range(6):
            # if i not exceed total length then verify the returned name
            if i < length:
                # if the two name not matches it will fail the test case
                self.assertEqual(self.user_name[i], self.person.get_name(self.user_id[i]))
            else:
                print("Testing for get_name no user test")
                # if length exceeds then check the 'no such user' type message
                self.assertEqual('There is no such user', self.person.get_name(i))
        print("\nFinish get_name test\n")


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()


#%%
import random
import nltk
from nltk.corpus import movie_reviews
print(nltk.pos_tag(nltk.word_tokenize('Albert Einstein was born in Ulm, Germany in 1879.')))
print(movie_reviews.categories(), len(movie_reviews.fileids()), len(movie_reviews.words()))
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) 
#{'contains(waste)': False, 'contains(lot)': False, ...}
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

#%%
import pyspark
spark = pyspark.sql.SparkSession.builder.appName('test').getOrCreate()

print(spark.range(10).collect())

from pyspark import SQLContext
sqlContext = SQLContext(spark)
dataset = sqlContext.createDataFrame([
            (10, 10.0),
            (50, 50.0),
            (100, 100.0),
            (500, 500.0)] * 10,
            ["feature", "label"])
dataset.show()


#%%
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
import random
NUM_SAMPLES = 10000
def inside(p):
 x, y = random.random(), random.random()
 return x*x + y*y < 1
count = sc.parallelize(list(range(0, NUM_SAMPLES)), 1).filter(inside).count()
pi = 4 * count / NUM_SAMPLES
print('Pi is roughly', pi)

sc.close()

#%%
import pyspark
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('data_processing').getOrCreate()
fpath = os.getcwd()
df=spark.read.csv('C:\\Users\\cyret\\Documents\\Python Scripts\\data/sales-of-shampoo-over-a-three-ye.csv',inferSchema=True, header=True)
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

print('do pandas udf\n')
from pyspark.sql.functions import pandas_udf
def prod(month, year):
    return 12 * (year - 1.0) + month
prod_udf = pandas_udf(prod, DoubleType())
df_new.withColumn('prod', prod_udf(df_new['month_double'], df_new['year_double'])).show(5)

print(pwd)
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

test_prediction.predictions.show(3)

from pyspark.ml.regression import RandomForestRegressor
rf_model = RandomForestRegressor(featuresCol='features', 
                                 labelCol='total', numTrees=100).fit(df_train)
predictions = rf_model.transform(df_test)
predictions.show()
rf_model.featureImportances

from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="total", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rf_model.stages[1]
print(rf_model)  # summary only


from pyspark.ml.feature import StandardScaler
 
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler.fit(df_train).transform(df_train).show()

spark.stop()


#%%
class Solution:
    def twoSum1(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        if len(nums) == 0: return []
        for i, valuei in enumerate(nums[:-1]):
            value = target - valuei
            for j, valuej in enumerate(nums[i+1:]):
                if valuej == value:
                    return [i, i + 1 + j]
                else:
                    pass

    def twoSum(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        if len(nums) < 2: return []
        dic = {}
        for i, v in enumerate(nums):
            print(i, v, target - v )
            if target - v in dic:
                print(dic)
                return [dic[target - v], i]
            else:
                print(v, dic)
                dic[v] =  i
                            
            
    
Solution().twoSum(nums = [2, 7, 11, 15], target = 9)
        


#%%
class Solution:
    def threeSum(self, nums: 'List[int]') -> 'List[List[int]]':
        if len(nums) < 3: return []
        s = sorted(nums)
        ans = []
#        right = len(s) - 1
#        for i, v in enumerate(s[:-2]):
        for i in range(len(s) - 2):
            if i > 0 and s[i] == s[i-1]: continue
            left, right = i + 1, len(s) - 1
            while left < right:
                print(i, s[i], s[left], s[right],  s[i]+ s[left] + s[right])
                su = s[i] + s[left] + s[right]
                if su < 0:
                    left += 1
                elif su > 0:
                    right -= 1
                else:
                    ans.append([s[i], s[left], s[right]])
                    while (left < right) and (s[left] == s[left+1]): left += 1
                    while (left < right) and (s[right] == s[right-1]): right -= 1
                    right -= 1
                    left += 1
        return ans
    
    def threeSum1(self, nums):
        res = []
        nums.sort()
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l +=1 
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1; r -= 1
        return res
#Solution().threeSum([-4, -1, -1, 0, 1, 2])
Solution().threeSum([-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6])

#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model

# return training data
def get_train():
    seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape((len(X), 1, 1))
    return X, y

# define model
model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1, activation='linear'))
# compile model
model.compile(loss='mse', optimizer='adam')
# fit model
X,y = get_train()
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
# save model to single file
model.save('lstm_model.h5')

# snip...
# later, perhaps run from another script

# load model from single file
model = load_model('lstm_model.h5')
# make predictions
yhat = model.predict(X, verbose=0)
print(yhat)

#%%
import pandas as pd
from sklearn.datasets import load_boston
import ggplot
from ggplot import *  # aes, geom_density, scale_color_brewer, facet_wrap
data = load_boston(return_X_y=False)
df = pd.DataFrame(data.data, columns=data.feature_names)

(ggplot(df, aes(x='CRIM', y='AGE')) + \
    geom_point() +\
    facet_wrap('RAD')
    + ggtitle("Area vs Population"))

(ggplot(df, aes(x='CRIM', y='AGE')) 
    + geom_point() 
    + geom_step(method = 'loess')
    + ggtitle("Area vs Population"))

(ggplot(aes(x='ZN'), data=df) 
    + geom_bar()
    + ggtitle("Area vs Population"))


