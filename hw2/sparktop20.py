from __future__ import print_function
import sys
from operator import add
from pyspark import SparkContext
import re
import time

#Using regular expression to remove punctuation in the file.
def removePunctuation(text):
    return re.sub(r'[^a-z0-9\s]','', text.lower().strip())

if __name__ == "__main__":
    start = time.time()	
	#Print error message when the command is wrong.
    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        exit(-1)
	#Create a RDD
    sc = SparkContext(appName="PythonWordCount")
	#Split the RDD into 8 partitions and remove punctuation.
    lines = (sc.textFile(sys.argv[1], 8).map(removePunctuation))
	#Transform the RDD into Key-Value pairs.
    counts = lines.flatMap(lambda x: x.split()) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
	#Get top 20 words in descending order.
    print(counts.takeOrdered(20,key = lambda x:-x[1]))
    end = time.time()
    print("Execution time:",end-start)
	#Shut down the SparkContexts
    sc.stop()

