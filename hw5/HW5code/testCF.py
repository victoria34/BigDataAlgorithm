from __future__ import print_function
import sys
from pyspark import SparkContext
from itertools import combinations
import numpy as np
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

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user) 
    '''
    item_pairs=[]
    for item1,item2 in combinations(items_with_rating,2):
	item_pairs.append([(item1[0],item2[0]),(item1[1],item2[1])])
    return item_pairs

def calcSim(item_pair,rating_pairs,movieDistance):
    ''' 
    For each item-item pair, return the specified similarity measure,
    along with co_raters_count
    '''
    sum_xy = (0.0)
    x_distance = 0.0
    y_distance = 0.0
    
    if item_pair[0] in movieDistance:
    	x_distance=movieDistance[item_pair[0]]

    if item_pair[1] in movieDistance:
    	y_distance=movieDistance[item_pair[1]]
		
    for rating_pair in rating_pairs:
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])

    cos_sim = cosine(sum_xy,x_distance,y_distance)

    return item_pair, cos_sim

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0
	
def keyOnFirstItem(item_pair,item_sim_data):
    '''
    For each item-item pair, make the first item's id the key
    '''
    (item1_id,item2_id) = item_pair
    return item1_id,(item2_id,item_sim_data)

def nearestNeighbors(item_id,item_sim,n):
    '''
    Sort the predictions list by similarity and select the top-N neighbors
    '''
    '''
    item_sim=[]
    for (item,sim) in items_and_sims :
	item_sim.append([item,sim])
    '''
    item_sim.sort(key=lambda x: x[1],reverse=True)
    return item_id, item_sim[:n]

def unpackpairs(movieID1,userID_rating,movieID2_sim):
    '''
    For each item-item pair, make the first item's id the key
    '''
    #(item1_id,item2_id) = item_pair
    #return item1_id,(item2_id,item_sim_data)

    (userID,rating) = userID_rating
    (movieID2,similarity) = movieID2_sim
    return (userID,movieID1),(similarity,similarity*rating)

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
	if len(sys.argv) != 4:
		print("Usage: testCF.py HW5train.txt HW5valid.txt HW5test.txt", file=sys.stderr)
		exit(-1)
    
	sc = SparkContext(appName="Model-Based")
    
	#Create a SparkContext to store txt files.   
	rawTraining = sc.textFile(sys.argv[1])
	rawValidation = sc.textFile(sys.argv[2])
	rawTest = sc.textFile(sys.argv[3])

	trainingRDD = rawTraining.map(get_ratings_tuple).cache()
	validationRDD = rawValidation.map(get_ratings_tuple).cache()
	testRDD = rawTest.map(get_ratings_tuple).cache()
	
	'''
	Model-Based CF
	'''
	
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
	
	# Use the bestRank=4 to create a model for predicting the ratings 
	# for the test dataset and then we will compute the RMSE.
    	myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations,
	lambda_=regularizationParameter)
    	testForPredictingRDD = testRDD.map(lambda (UserID, MovieID, Rating)
	: (UserID, MovieID))
	test_start = time.time()
    	predictedTestRDD = myModel.predictAll(testForPredictingRDD)
	# Test error
    	testRMSE = computeError(testRDD, predictedTestRDD)
	# Test time
	test_time = time.time() - test_start
	
	print ("----------- Model-Based CF ------------------")
	print ('\nNumber of test: %s\n' % testRDD.count())
	print ('\nNumber of valid: %s\n' % validationRDD.count())
	print ('\nNumber of train: %s\n' % trainingRDD.count())
	
    	i=0;
    	for rank in ranks:
        	print ("-----------------------------")
        	print ('\nFor rank %s : \n' % rank)
		print ('the Model-Based training error is %s\n' % training_errors[i])
		print ('the Model-Based training time is %s\n' % training_time[i])
		print ('the Model-Based validation error is %s\n' % validation_errors[i])
		print ('the Model-Based validation time is %s\n' % validation_time[i])
		i += 1   
    	print ("-----------------------------")
    	print ('\nThe best training model was trained with rank %s\n' % bestRank)
	
    	print ('\nThe model had a RMSE on the test set of %s\n' % testRMSE)
	print ('\nThe Model-Based test time is %s\n' % test_time)

	'''
	Baseline CF
	'''
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
		if testRDDForPrediction[index][1] in movieRating:#testRDD.mid in movieRating.key
			movieAvg=movieRating[testRDDForPrediction[index][1]]

		predictedRating=userAvg+movieAvg-overallMeanRDD 
		#print("--------------------------")
		#print('\npredictedRating:%s\n'%predictedRating)
		#([uid,mid,predictedRating])
		testForPredicted.append([testRDDForPrediction[index][0],testRDDForPrediction[index][1],predictedRating])
	

	predictedRdd = sc.parallelize(testForPredicted).map(lambda log : (log[0],log[1],log[2]))
	 
	RMSE=computeError(predictedRdd,testRDD);
	end =time.time()-start
	print("------------------------ Baseline CF ---------------------------------")
	print ('\nThe Baseline RMSE is %s\n' % RMSE)
	print('\nThe execution time is %s\n'%end)
	
	'''
	Item-Item CF
	'''
	''' 
        Obtain the sparse user-item matrix
        user_id -> [(item_id_1, rating_1),
                   [(item_id_2, rating_2),
                    ...]
    	'''
	
	trainingMRDD = trainingRDD.map(lambda (UserID, MovieID, Rating): (UserID, (MovieID, Rating))).groupByKey()
	#print ('\ntrainingRDD: %s\n' % trainingMRDD.take(10))

	#calculate Distance of each movie
	movieDistanceRDD= trainingRDD.map(lambda (UserID, MovieID, Rating): (MovieID, Rating)).reduceByKey(lambda rating1,rating2:math.sqrt(rating1*rating1+rating2*rating2)).collect()
	#print ('\nmovieDistanceRDD: %s\n' % movieDistanceRDD.take(10))
	
	#used in def calcSim(item_pair,rating_pairs,movieDistance)
	movieDistance={}
	for index in range(0,len(movieDistanceRDD)):
		movieDistance[movieDistanceRDD[index][0]] = movieDistanceRDD[index][1]#key=value
	
	'''
        Get all item-item pair combos
        (item1,item2) ->    [(item1_rating,item2_rating),
                             (item1_rating,item2_rating),
                             ...]
    	'''

	movie_pairsRDD = trainingMRDD.filter(lambda p: len(p[1]) > 1).flatMap(lambda p: findItemPairs(p[0],p[1])).groupByKey()
	print ('\nmovie_pairsRDD: %s\n' % movie_pairsRDD.take(10))
	
    	'''
    	Calculate the cosine similarity for each item pair
        (item1,item2) ->    similarity
        '''
	
        movie_simsRDD = movie_pairsRDD.map(lambda p: calcSim(p[0],p[1],movieDistance))
	#print ('\nmovie_simsRDD count: %s\n' % movie_simsRDD.count())	
	print ('\nmovie_simsRDD: %s\n' % movie_simsRDD.take(10))
	
	#(item1,item2) -> similarity change to item1,(item2,sim)
	item1_sim_item2_simRDD=movie_simsRDD.map(lambda (item_pair,item_sim_data):keyOnFirstItem(item_pair,item_sim_data))
	print ('\nitem1_sim_item2_simRDD: %s\n' % item1_sim_item2_simRDD.take(10))
	
	'''
	movieSim={}
	
	for index in range(0,len(movie_simsRDD))
		movieSim[movie_simsRDD[index][0]]= movie_simsRDD[index][1]
	'''
	
	#get movieID,userID from validation and join with movie_
	trainPredictedRDD = trainingRDD.map(lambda (userID,movieID,rating):(movieID,(userID,rating)))
	
	#predictedRatingRDD format ((user,item),(sim,sim*rating)),,,, ratingsInverse.join(similarities) fromating as (Item,((user,rating),(item,similar)))
	#predictedRatingRDD = trainPredictedRDD.join(item1_sim_item2_simRDD).map(lambda (movieID1,(userID,rating),(movieID2,similarity)):((userID,movieID1),(similarity,similarity*rating)))
	
 	predictedRatingRDD = trainPredictedRDD.join(item1_sim_item2_simRDD).map(lambda x:unpackpairs(x[0],x[1][0],x[1][1]))
	#finalPredictedRatingRDD format ((user,item),predict)
	finalPredictedRatingRDD = predictedRatingRDD.reduceByKey(lambda a,b:((a[0]+b[0]),(a[1]+b[1]))).map(lambda ((userID,movieID),(simsum,simratsum)):(userID,movieID,simratsum/simsum))
	print('\nfinalPredictedRatingRDD:%s\n' % finalPredictedRatingRDD.take(50))
	
	validation_start = time.time()
	# Validation error
    	validationRMSE = computeError(finalPredictedRatingRDD,validationRDD)
	# validation time
	validation_time = time.time() - validation_start

	test_start = time.time()
	# Test error
    	testRMSE = computeError(finalPredictedRatingRDD,testRDD)
	# Test time
	test_time = time.time() - test_start
	
	print("------------------------ Item-Item CF ---------------------------------")
	print('\nValidation RMSE:%s\n' % validationRMSE)
	print('\nValidation time:%s\n' % validation_time)
	print('\nTest RMSE:%s\n' % testRMSE)
	print('\nTest time:%s\n' % test_time)
	
	'''
	Use-User CF
	'''
	trainingUserRDD = trainingRDD.map(lambda (UserID, MovieID, Rating): (MovieID, (UserID, Rating))).groupByKey()
	print ('\ntrainingRDD: %s\n' % trainingRDD.take(10))

	#calculate Distance of each user
	userDistanceRDD= trainingRDD.map(lambda (UserID, MovieID, Rating): (UserID, Rating)).reduceByKey(lambda rating1,rating2:math.sqrt(rating1*rating1+rating2*rating2)).collect()
	#print ('\nuserDistanceRDD: %s\n' % movieDistanceRDD.take(10))
	
	#used in def calcSim(item_pair,rating_pairs,movieDistance)
	userDistance={}
	for index in range(0,len(userDistanceRDD)):
		userDistance[userDistanceRDD[index][0]] = userDistanceRDD[index][1]#key=value
	
	'''
        Get all item-item pair combos
        (user1,user2) ->    [(user1_rating,user2_rating),
                             (user1_rating,user2_rating),
                             ...]
    	'''

	user_pairsRDD = trainingUserRDD.filter(lambda p: len(p[1]) > 1).flatMap(lambda p: findUserPairs(p[0],p[1])).groupByKey()
	print ('\nuser_pairsRDD: %s\n' % user_pairsRDD.take(10))
	
    	'''
    	Calculate the cosine similarity for each item pair
        (item1,item2) ->    similarity
        '''
	
        user_simsRDD = user_pairsRDD.map(lambda p: calcSim(p[0],p[1],userDistance))
	#print ('\nmovie_simsRDD count: %s\n' % movie_simsRDD.count())	
	print ('\nmovie_simsRDD: %s\n' % user_simsRDD.take(10))
	
	#(item1,item2) -> similarity change to item1,(item2,sim)
	user1_sim_user2_simRDD=user_simsRDD.map(lambda (user_pair,user_sim_data):keyOnFirstItem(user_pair,user_sim_data))
	print ('\nuser1_sim_user2_simRDD: %s\n' % user1_sim_user2_simRDD.take(10))
	
	'''
	movieSim={}
	
	for index in range(0,len(movie_simsRDD))
		movieSim[movie_simsRDD[index][0]]= movie_simsRDD[index][1]
	'''
	
	#get movieID,userID from validation and join with movie_
	trainPredictedRDD = trainingRDD.map(lambda (userID,movieID,rating):(userID,(movieID,rating)))
	#print ('\nvalidationMoiveUnderSameUserRDD: %s\n' % validationMoiveUnderSameUserRDD.take(10))
	
	#predictedRatingRDD format ((user,item),(sim,sim*rating)),,,, ratingsInverse.join(similarities) fromating as (Item,((user,rating),(item,similar)))
	#predictedRatingRDD = trainPredictedRDD.join(item1_sim_item2_simRDD).map(lambda (movieID1,(userID,rating),(movieID2,similarity)):((userID,movieID1),(similarity,similarity*rating)))
 	predictedRatingRDD = trainPredictedRDD.join(user1_sim_user2_simRDD).map(lambda x:unpackpairs(x[0],x[1][0],x[1][1]))
	#finalPredictedRatingRDD format ((user,item),predict)
	finalPredictedRatingRDD = predictedRatingRDD.reduceByKey(lambda a,b:((a[0]+b[0]),(a[1]+b[1]))).map(lambda ((movieID,userID),(simsum,simratsum)):(userID,movieID,simratsum/simsum))
	print('\nfinalPredictedRatingRDD:%s\n' % finalPredictedRatingRDD.take(10))
	
	validation_start = time.time()
	# Validation error
    	validationRMSE = computeError(finalPredictedRatingRDD,validationRDD)
	# validation time
	validation_time = time.time() - validation_start

	test_start = time.time()
	# Test error
    	testRMSE = computeError(finalPredictedRatingRDD,testRDD)
	# Test time
	test_time = time.time() - test_start
	
	print("------------------------ User-User CF ---------------------------------")
	print('\nValidation RMSE:%s\n' % validationRMSE)
	print('\nValidation time:%s\n' % validation_time)
	print('\nTest RMSE:%s\n' % testRMSE)
	print('\nTest time:%s\n' % test_time)
    


    
    

		
		
	
