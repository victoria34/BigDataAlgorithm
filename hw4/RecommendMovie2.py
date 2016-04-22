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
        entry (str): a line in the ratings dataset in the f
		orm of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating, Timestamp)
    """
    items = entry.split(',')
    return int(items[0]), int(items[1]), float(items[2]),long(items[3])


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
    if len(sys.argv) != 2:
        print("Usage: RecommendMovie20M.py ratings.csv", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="RecommendMovies")
    #Create a SparkContext to store dat files.
    numPartitions = 2
    rawRatings = sc.textFile(sys.argv[1])
	#extract header from rating.csv
    header = rawRatings.first() 
    print ("-----------------------------")
    print ('\nHeader: %s\n' % header)
	#filter out header
    rawRatings = rawRatings.filter(lambda x:x !=header).repartition(numPartitions)    

    ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
    print ("-----------------------------")
    print ('\nRatings: %s\n' % ratingsRDD.take(3))

#Part 2: Collaborative Filtering
    #Sort the ratings data in ascending order of timestamp (past to present)	
    sortedRatingsRDD = ratingsRDD.sortBy(lambda r: r[3])
    print ("-----------------------------")
    print ('\nSorted Ratings: %s\n' % sortedRatingsRDD.take(3))
    noTimeSortedRatingsRDD = sortedRatingsRDD.map(lambda (UserID, MovieID, 
	Rating, Timestamp): (UserID, MovieID, Rating))

    #The training set contains the first 80% of the original dataset. 
    #The validation set contains the next 10% of the original dataset. 
    #The test set contains the last 10% of the original dataset. 
	#To randomly split the dataset into the multiple groups, we can use the pySpark 
	#randomSplit() transformation. 
	#randomSplit() takes a set of splits and seed and returns multiple RDDs.
    trainingRDD, validationRDD, testRDD = noTimeSortedRatingsRDD
	                               .randomSplit([8, 1, 1], seed=0L)
    print ('\nTraining: %s, validation: %s, test: %s\n' % 
	(trainingRDD.count(),validationRDD.count(),testRDD.count()))

	# For the prediction step, create an input RDD, validationForPredictRDD, 
	# consisting of (UserID, MovieID) pairs that we extract from validationRDD.
    validationForPredictRDD = validationRDD.map(lambda (UserID, MovieID, Rating)
	: (UserID, MovieID))
    trainingForPredictRDD = trainingRDD.map(lambda (UserID, MovieID, Rating)
	: (UserID, MovieID))

    seed = 5L
    iterations = 5
    regularizationParameter = 0.1
    ranks = [4, 8, 12]
    training_errors = [0, 0, 0] #training error
    validation_errors = [0, 0, 0]
    err = 0
    tolerance = 0.03
    training_time = [0,0,0];
    validation_time = [0,0,0];

    minError = float('inf')
    bestRank = -1
    bestIteration = -1

    for rank in ranks:
		train_start = time.time()
		#Training
        model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
		lambda_=regularizationParameter)
		training_time[err] = time.time()-train_start    
		#Training RMSE
		tra_predictedRatingsRDD = model.predictAll(trainingForPredictRDD)
		#Use the Root Mean Square Error (RMSE) or Root Mean Square Deviation 
		#(RMSD) to compute the error of each model.
		tra_error = computeError(tra_predictedRatingsRDD, trainingRDD)
		training_errors[err] = tra_error

        validation_start = time.time()
		#Validation
        val_predictedRatingsRDD = model.predictAll(validationForPredictRDD)
		validation_time[err] = time.time()-validation_start
		#Validation RMSE
        val_error = computeError(val_predictedRatingsRDD, validationRDD)
        validation_errors[err] = val_error
        if val_error < minError:
            minError = val_error
            bestRank = rank
        err += 1

	# Use the bestRank=8 to create a model for predicting the ratings 
	# for the test dataset and then we will compute the RMSE.
    myModel = ALS.train(trainingRDD, 8, seed=seed, iterations=iterations,
	lambda_=regularizationParameter)
    testForPredictingRDD = testRDD.map(lambda (UserID, MovieID, Rating)
	: (UserID, MovieID))
	test_start = time.time()
    predictedTestRDD = myModel.predictAll(testForPredictingRDD)
	# Test error
    testRMSE = computeError(testRDD, predictedTestRDD)
	# Test time
	test_time = time.time() - test_start

    i=0;
    for rank in ranks:
        print ("-----------------------------")
        print ('\nFor rank %s : \n' % rank)
		print ('the training error is %s\n' % training_errors[i])
		print ('the training time is %s\n' % training_time[i])
		print ('the validation error is %s\n' % validation_errors[i])
		print ('the validation time is %s\n' % validation_time[i])
		i += 1

    print ("-----------------------------")
    print ('\nThe best training model was trained with rank %s\n' % bestRank) 
    print ('\nThe model had a RMSE on the test set of %s\n' % testRMSE)
	print ('\nThe validation time is %s\n' % test_time)
   