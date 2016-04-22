from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from operator import add
from pyspark import SparkContext
import re
import time
import datetime
from pyspark.sql import Row

month_map = {'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12}

def parse_apache_time(s):
    """ Convert Apache time format into a Python datetime object
    Args:
        s (str): date and time in Apache time format
    Returns:
        datetime: datetime object (ignore timezone for now)
    """
    return datetime.datetime(int(s[7:11]),
                             month_map[s[3:6]],
                             int(s[0:2]),
                             int(s[12:14]),
                             int(s[15:17]),
                             int(s[18:20]))

def parseApacheLogLine(logline):
    """ Parse a line in the Apache Common Log format
    Args:
        logline (str): a line of text in the Apache Common Log format
    Returns:
        tuple: either a dictionary containing the parts of the Apache Access Log and 1,
               or the original invalid log line and 0
    """
    match = re.search('^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S*)\s*(.*)\s*(\S*)" (\d{3}) (\S+)', logline)
    if match is None:
        return (logline, 0)
    size_field = match.group(9)
    if size_field == '-':
        size = long(0)
    else:
        size = long(match.group(9))
    return (Row(
        host          = match.group(1),
        client_identd = match.group(2),
        user_id       = match.group(3),
        date_time     = parse_apache_time(match.group(4)),
        method        = match.group(5),
        endpoint      = match.group(6),
        protocol      = match.group(7),
        response_code = int(match.group(8)),
        content_size  = size
    ), 1)
	
if __name__ == "__main__":
    start = time.time()	
	#Print error message when the command is wrong.
    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        exit(-1)
	#Create a RDD to store parsed logs.
    sc = SparkContext(appName="PythonLog")
    parsed_logs = sc.textFile(sys.argv[1],8) \
                   .map(parseApacheLogLine) \
                   .cache() 
	#Create a RDD to store accessed logs.
    access_logs = parsed_logs.filter(lambda s: s[1] == 1) \
                   .map(lambda s: s[0]) \
                   .cache()
	#Create a RDD to store failed logs.
    failed_logs = parsed_logs.filter(lambda s: s[1] == 0) \
                   .map(lambda s: s[0])
    #Count the number of failed logs.
    failed_logs_count = failed_logs.count()
    #Print failed logs and the number of them.
    if failed_logs_count > 0:
        print ('Number of invalid logline: %d' % failed_logs.count())
        for line in failed_logs.take(20):
            print ('Invalid logline: %s' % line)

    print ('Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (parsed_logs.count(), access_logs.count(), failed_logs.count()))
    
    #Top Ten Error Endpoints. Create a sorted list containing top ten endpoints and the number of times that they were accessed with non-200 return code.
	#Filter accessed logs with non-200 return code.
	not200 = access_logs.filter(lambda log: log.response_code != 200)
    endpointCountPairTuple = not200.map(lambda log: (log.endpoint, 1))	
    endpointSum = endpointCountPairTuple.reduceByKey(lambda a, b: a + b)
	#Show the top 10 error urls in descending order.
    topTenErrURLs = endpointSum.takeOrdered(10, lambda s: -1 * s[1])
    print ('Top Ten failed URLs: %s' % topTenErrURLs)
	
    #Number of Unique Hosts.  
    hosts = access_logs.map(lambda log: log.host)
	#filter same hosts to make unique hosts
    uniqueHosts = hosts.distinct()
	#count the number of unique hosts.
    uniqueHostCount = uniqueHosts.count()
    print ('Unique hosts: %d' % uniqueHostCount)
    
    #Number of Unique Daily Hosts 
	#get unique logs for format day-host
    dayToHostPairTuple = access_logs.map(lambda log: (log.date_time.day, log.host)).distinct()
	#group hosts by day
    dayGroupedHosts = dayToHostPairTuple.groupBy(lambda (x,y): x)
	#count the number of different hosts that make requests each day
    dayHostCount = dayGroupedHosts.map(lambda (x, hosts): (x, len(hosts)))
	#sort day-count pairs by day in ascending order
    dailyHosts = (dayHostCount.sortByKey() \
                              .cache())
    # get the top 30 unique daily hosts
	dailyHostsList = dailyHosts.take(30)
    print ('Unique hosts per day: %s' % dailyHostsList)

    #Counting 404 Response Codes. How many 404 records are in the log?
	#filter the accessed logs whose response code is 404.
    badRecords = (access_logs.filter(lambda log: log.response_code == 404) \
                             .cache())
	#count the number of 404 logs
    print ('Found %d 404 URLs' % badRecords.count())

    #Listing the Top Twenty 404 Response Code Endpoints
    badEndpointsCountPairTuple = badRecords.map(lambda log: (log.endpoint, 1))
	#get endpoints with 404 response code and count number of each end ponits
    badEndpointsSum = badEndpointsCountPairTuple.groupByKey().map(lambda (endpoint, counts): (endpoint, sum(counts)))
	#sort the endpoint-count pairs by count in descending order and get the top 20 pairs.
    badEndpointsTop20 = badEndpointsSum.sortBy(lambda (endpoint,count): -count).take(20)
    print ('Top Twenty 404 URLs: %s' % badEndpointsTop20)

    end = time.time()
    print("Execution time:",end-start)
    

