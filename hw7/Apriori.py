from __future__ import print_function
from pyspark import SparkContext
from itertools import combinations
import os
import sys
import time
from operator import add
   
def findItemPairs(user_id,items_with_rating,preSetList):
    '''
    For each word1, find all word2-word3 combos and compare whether they are in 2-sets (preSetList).
    If they are, generate 3-sets (word1, word2, word3). 
    '''
    item_tris=[]
        
    for item1,item2 in combinations(items_with_rating,2):
        for singleSet in preSetList:
            if item1 in singleSet and item2 in singleSet:
	            item_tris.append([user_id,item1,item2])
    return item_tris

def findItemTris(user_id,items_with_rating,preSetList):
    ''' 
    For each (word1, word2), find all word3-word4 combos and compare whether they are in 3-sets (preSetList).
    If they are, generate 4-sets (word1, word2, word3, word4).
    '''
    item_fours=[]
        
    for item1,item2 in combinations(items_with_rating,2):
        for singleSet in preSetList:
            if user_id[1] in singleSet and item1 in singleSet and item2 in singleSet:
	            item_fours.append([user_id,item1,item2])
    return item_fours
    
def findSet(words,files):
    '''
    Count the number of all words occurence in same file.
    If all words in a set appear in the same .txt file, appear=appear+1. 
    '''    
    appear=0
    for file in files:
        num=0
        for word in words:
            if word in file.split(' '):
                num=num+1
                
        if num==len(words):
            appear=appear+1
            
    return words,appear
    
        
if __name__ == "__main__":	
    #Print error message when the command is wrong.
    if len(sys.argv) != 2:
        print("Usage: Apriori.py filterbbc/business", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="FrequentItemSet")

    start=time.time()
    #get words whose count>30 and only map words
    contentRDD = sc.textFile(sys.argv[1], 8)
    wordsRDD = contentRDD.flatMap(lambda x: x.split(' '))\
                         .map(lambda x: (x, 1)) \
                         .reduceByKey(add)\
                         .filter(lambda (word,count):count>30)\
                         .map(lambda (word,count):word)\
                         .cache()
    #wordnum=wordsRDD.count()
    #print("\n\nwordsRDD:\n\n",wordsRDD.take(wordnum))
    
    #read all files under a folder, key=file path & value=file content
    '''
    This is a folder under hdfs:
        hdfs://a-hdfs-path/part-00000 
        hdfs://a-hdfs-path/part-00001 
        ...
        hdfs://a-hdfs-path/part-nnnnn 
    Read:
        val rdd = sparkContext.wholeTextFile("hdfs://a-hdfs-path") 
    rdd contains:
        (a-hdfs-path/part-00000, its content) 
        (a-hdfs-path/part-00001, its content)
        ...
        (a-hdfs-path/part-nnnnn, its content)
    '''
    
    #store each .txt file to filesList.
    filesRDD=sc.wholeTextFiles(sys.argv[1], 8)\
               .map(lambda x:x[1])#get rid of path and keep content
    #print("\n\nfilesRDD:%s\n\n" % filesRDD.count())
    filesList=filesRDD.collect()
    #print("\n\nfilesList:%s\n\n" % len(filesList))
    
    #generate 2-sets
    wordsList=wordsRDD.collect()
    #combine words to pairs
    pairsList=list(combinations(wordsList,2))
    pairsRDD=sc.parallelize(pairsList)\
               .map(lambda x:findSet(x,filesList))\
               .filter(lambda x:x[1]>30)
    #pairnum=wordsRDD.count()
    #print("\n\npairsRDD:%s\n\n"%pairsRDD.take(pairnum))
    #genereate 2-sets list whose threshold>30
    pairsFilterList=pairsRDD.map(lambda x:x[0]).collect()
    #print("\n\npairsFilterList:%s\n\n" % pairsFilterList)
    
    #generate 3-sets
    trisRDD=pairsRDD.map(lambda x:[x[0][0],x[0][1]])\
                    .groupByKey()\
                    .flatMap(lambda x: findItemPairs(x[0],x[1],pairsFilterList))
    #print("\n\ntrisRDD:%s\n\n"%trisRDD.take(10))
    
    trisFilterRDD=trisRDD.map(lambda x:findSet(x,filesList))\
                         .filter(lambda x:x[1]>30)
    #trinum=wordsRDD.count()
    #print("\n\ntrisFilterRDD:\n\n",trisFilterRDD.take(trinum))
    
    #genereate 3-sets list whose threshold>30
    trisFliterList=trisFilterRDD.map(lambda x:x[0]).collect()
    #print("\n\ntrisFilterList:%s\n\n" % trisFliterList)
    
    #generate 4-sets
    foursRDD=trisFilterRDD.map(lambda x:x[0])\
                          .map(lambda x:((x[0],x[1]),x[2]))\
                          .groupByKey()\
                          .flatMap(lambda x: findItemTris(x[0],x[1],trisFliterList))
    #print("\n\nfoursRDD:%s\n\n"%foursRDD.take(10))
    #flatten 4-sets
    foursFilterRDD=foursRDD.map(lambda ((word1,word2),word3,word4):(word1,word2,word3,word4)).distinct()
    #print("\n\nfoursFilterRDD:%s\n\n"%foursFilterRDD.take(10))
    #filter 4-sets whose threshold>30
    top4setRDD=foursFilterRDD.map(lambda x:findSet(x,filesList)).filter(lambda x:x[1]>30)
    #get top 10 4-sets in descending order
    print("\nNumber of 4-sets:%s\n" % foursFilterRDD.count())
    print("\nTop Ten 4-sets:%s\n" % top4setRDD.takeOrdered(10,key = lambda x:-x[1]))
    
    end=time.time()-start
    print("\nExecution time: %s\n" % end)
    sc.stop()
