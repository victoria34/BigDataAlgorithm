# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 00:06:06 2016
@author: Jing Su
"""
import heapq
import time

start = time.clock()             # start time of CPU
#open the frirst 3 sections
file=open("E:\muppts\8001 Big Data Algorithm\hw1\First3Section.txt","r+")  
wordcount={}
#turn all words to lower case and split each word from the article
for word in file.read().lower().split(): 
    if word not in wordcount:    
        wordcount[word] = 1      #set wordcont=1 if the word appear once
    else:                           
        wordcount[word] += 1     #set wordcont+=1 if the word appear again 
#for key in wordcount.keys():
        #print the key-value pairs of all words
        #print ("%s %s " %(key , wordcount[key])) 
  
topNum = 10 
#get top 10 words that have largest values
nlargestList = heapq.nlargest(topNum, wordcount.values())   
for value in nlargestList: #find the top 10 values      
    for key in wordcount.keys(): #find keys that have largest values        
        if wordcount[key] == value:  
            print ("%s %s " %(key , wordcount[key])) 
            #Set wordcount[key]=0 after the word printed
            #So the word won't be printed the second time
            wordcount[key] = 0   
            
file.close();  #close the file

end = time.clock()  #end time of CPU
print ("Execution time:",end-start) #print CPU execution time