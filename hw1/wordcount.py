# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
file=open("C:/Users/Administrator/.spyder2-py3/Sonnets.txt","r+")
wordcount={}
for word in file.read().split():
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1
for key in wordcount.keys():
  print ("%s %s " %(key , wordcount[key]))
file.close();