#store cleaned data into original each file like /bbc/business/001.txt
from __future__ import print_function
from pyspark import SparkContext
import os
import sys
import re
import time


def removePunctuation(text):
    return re.sub(r'[^a-z0-9\s]','', text.lower().strip())
    
split_regex = r'\W+'

def simpleTokenize(string):
    """ A simple implementation of input string tokenization
    Args:
        string (str): input string
    Returns:
        list: a list of tokens
    """
    return filter(lambda string: string != '', re.split(split_regex, string.lower()))

def tokenize(string):
    """ An implementation of input string tokenization that excludes stopwords
    Args:
        string (str): input string
    Returns:
        list: a list of tokens without stopwords
    """
    return filter(lambda tokens: tokens not in stopwords, simpleTokenize(string))

if __name__ == "__main__":	
    #Print error message when the command is wrong.
    if len(sys.argv) != 3:
        print("Usage: filterdata.py bbc/business stop-word-list.txt", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="FilterData")
    # get all stopwords
    stopwords = set(sc.textFile(sys.argv[2]).collect())
    #print ('These are the stopwords: %s' % stopwords)

    # read file from bbc
    for filename in os.listdir(sys.argv[1]):
        print("Filename:%s" % filename)
        path=""
        path=sys.argv[1]+'/'+filename
        rawFile=open(path,'r')
        rawContent=rawFile.read()
        rawFile.close()
        # Change all words to lower case, remove all punctuation marks and symbols
        filterContent=removePunctuation(rawContent)
        #remove the stop words in the stop-word-list.txt file 
        removeStop=tokenize(filterContent)
        #print(removeStop)
        newPath=""
        newPath='/home/hadoop/'+sys.argv[1]+'/'+filename
        filterFile=open(newPath,'w')
        filterFile.write(removePunctuation(str(removeStop)))
        filterFile.close()
    
    print("Filter data finished.")
        

