from __future__ import print_function
import sys
from pyspark import SparkContext

if __name__ == "__main__":	
    #Print error message when the command is wrong.
    if len(sys.argv) != 2:
	print("Usage: split.py ratings.dat", file=sys.stderr)
	exit(-1)
		
    #Start split the data into training set, validation set, test set.
    lines=[]
    with open(sys.argv[1]) as f:
    	for i,line in enumerate(f):
		lines.append(line)
    numOfLine=i+1
	
    trainingFile=open('/home/hadoop/HW5train.txt','w')
    validationFile=open('/home/hadoop/HW5valid.txt','w')
    testingFile=open('/home/hadoop/HW5test.txt','w')

    #sort the lines by timestamp in descending order
    lines.sort(key=lambda x:int(x.split("::")[3]),reverse=False)

    i=0
    for line in lines:
	if i<100000:
		testingFile.write(line)
	elif i<200000:
		validationFile.write(line)
	elif i<numOfLine:
		trainingFile.write(line)
	i=i+1
		
    trainingFile.close()
    validationFile.close()
    testingFile.close()
    print("Split finished!")