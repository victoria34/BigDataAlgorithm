from __future__ import print_function
import sys
from pyspark import SparkContext
import re
import time
import math
from pyspark.mllib.recommendation import ALS

def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user 
		where each entry is in the form (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form 
		(UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda (UserID, MovieID, Rating): 
	((UserID, MovieID), Rating))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda (UserID, MovieID, Rating): 
	((UserID, MovieID), Rating))

    # Compute the squared error for each matching entry (i.e., the same (User ID, 
	# Movie ID) in each RDD) in the reformatted RDDs using RDD transformtions 
	# do not use collect()
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                       .map(lambda ((UserID, MovieID), (PredRating, ActRating)): 
					   (PredRating - ActRating)**2))
    
    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.reduce(lambda a,b: a+b)

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(1.0*totalError/numRatings)


if __name__ == "__main__":	
	#Print error message when the command is wrong.
	if len(sys.argv) != 3:
		print("Usage: BaseLine.py HW5train.txt  HW5test.txt", file=sys.stderr)
		exit(-1)
    
	sc = SparkContext(appName="Model-Based")
    
	#Create a SparkContext to store txt files.   
	rawTraining = sc.textFile(sys.argv[1])
	rawTest = sc.textFile(sys.argv[2])

	trainingRDD = rawTraining.map(get_ratings_tuple).cache()
	testRDD = rawTest.map(get_ratings_tuple).cache()

	print ("-----------------------------")
	print ('\nNumber of test: %s\n' % testRDD.count())
	print ('\nNumber of train: %s\n' % trainingRDD.count())
	
	
	#Get mean rating of all movieID under traningRDD
	overallMeanRDD = trainingRDD.map(lambda (UserID, MovieID, Rating): (Rating)).mean()
	print ('\noverallMean: %s\n' % overallMeanRDD)
	#Calculate mean rating of each user
	userDeviationRDD = trainingRDD.map(lambda (UserID, MovieID, Rating): (UserID,Rating)).mapValues(lambda Rating: (Rating, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda v: v[0]/v[1]).collect()
	#print ('\nuserDeviationRDD: %s\n' % userDeviationRDD.take(10))
	#Calculate mean rating of each movie
	movieDeviationRDD = trainingRDD.map(lambda (UserID, MovieID, Rating): (MovieID,Rating)).mapValues(lambda Rating: (Rating, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda v: v[0]/v[1]).collect()
	#print ('\nmovieDeviationRDD: %s\n' % movieDeviationRDD.take(10))
	
	
	userRating={}
	movieRating={}
	
	#[index,key,value]
	for index in range(0,len(userDeviationRDD)):
		userRating[userDeviationRDD[index][0]] = userDeviationRDD[index][1]#key=value
	for index in range(0,len(movieDeviationRDD)):
		movieRating[movieDeviationRDD[index][0]] = movieDeviationRDD[index][1]#key=value
	#index uid mid = [index][0][1]
	testRDDForPrediction=testRDD.map(lambda (UserID, MovieID, Rating): (UserID, MovieID)).collect()
	testForPredicted=[]#List is used to store result and turn to RDD

	start = time.time()
	for index in range(0, len(testRDDForPrediction)):
		if testRDDForPrediction[index][0] in userRating:#testRDD.uid in userRating.key
			userAvg=userRating[testRDDForPrediction[index][0]]
		if testRDDForPrediction[index][1] in movieRating:#??testRDD?mid?movieRating??key?
			movieAvg=movieRating[testRDDForPrediction[index][1]]

		predictedRating=userAvg+movieAvg-overallMeanRDD 
		print("--------------------------")
		print('\npredictedRating:%s\n'%predictedRating)
		#([uid,mid,predictedRating])
		testForPredicted.append([testRDDForPrediction[index][0],testRDDForPrediction[index][1],predictedRating])
	

	predictedRdd = sc.parallelize(testForPredicted).map(lambda log : (log[0],log[1],log[2]))
	#print("---------------------------------------------------------")
	#print('\n\nPredictedRDD:%s\n\n'% predictedRdd.take(3))
	#print("---------------------------------------------------------")
 
	RMSE=computeError(predictedRdd,testRDD);
	end =time.time()-start
	print ('\nThe RMSE is %s\n' % RMSE)
	print('\nThe training time is %s\n'%end)
	