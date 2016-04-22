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
        entry (str): a line in the ratings dataset in the form of 
		UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2]) 

def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of 
		MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)

def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    list_tuples = [1.0*rating for rating in IDandRatingsTuple[1]]
    return (IDandRatingsTuple[0], (len(list_tuples), sum(list_tuples)/len(list_tuples)))

if __name__ == "__main__":
    start = time.time()	
    #Print error message when the command is wrong.
    if len(sys.argv) != 3:
        print("Usage: RecommendMovie1M.py ratings.dat movies.dat", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="RecommendMovies")
    #Create a SparkContext to store dat files.
    numPartitions = 2
    rawRatings = sc.textFile(sys.argv[1]).repartition(numPartitions)
    rawMovies = sc.textFile(sys.argv[2])

#Part 1: Basic Recommendations
    # Parsing the two files yields two RDDS and cache them. 
    ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
    moviesRDD = rawMovies.map(get_movie_tuple).cache()
	
    # From ratingsRDD with tuples of (UserID, MovieID, Rating) create an RDD with tuples of
    # the (MovieID, iterable of Ratings for that MovieID)
    movieIDsWithRatingsRDD = ratingsRDD.map(lambda (userid, movieid, rating): 
	(movieid, rating)).groupByKey()

    # Using `movieIDsWithRatingsRDD`, compute the number of ratings and average rating for 	
    # each movie to yield tuples of the form (MovieID, (number of ratings, average rating))
    movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(lambda rec: getCountsAndAverages(rec))

    # To `movieIDsWithAvgRatingsRDD`, apply RDD transformations that use `moviesRDD` to get the movie
    # names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form
    # (average rating, movie name, number of ratings)
    movieNameWithAvgRatingsRDD = moviesRDD.join(movieIDsWithAvgRatingsRDD).map(lambda 
	(movieID, (movieName, (numRatings, avgRating))): (avgRating, movieName, numRatings))

    # Apply an RDD transformation to `movieNameWithAvgRatingsRDD` to limit the results to movies with
    # ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by 
    # the average rating to get the movies in order of their rating (highest rating first)
    movieLimitedAndSortedByRatingRDD = movieNameWithAvgRatingsRDD.filter(lambda (_, __, numRatings): 
	numRatings > 500).sortBy(sortFunction, False)
    print ('The top 20 movies with highest average ratings and more than 500 reviews: %s\n' % 
	movieLimitedAndSortedByRatingRDD.take(20))	
	
