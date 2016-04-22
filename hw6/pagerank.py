"""
This is an example implementation of PageRank. For more conventional use,
Please refer to PageRank implementation provided by graphx
"""
from __future__ import print_function

import re
import sys
import time
from operator import add
from pyspark import SparkContext



def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)

def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: pagerank <file> <iterations>", file=sys.stderr)
        exit(-1)

    print("""WARN: This is a naive implementation of PageRank and is
          given as an example! Please refer to PageRank implementation provided by graphx""",
          file=sys.stderr)

    # Initialize the spark context.
    start=time.time()
    sc = SparkContext(appName="PythonPageRank")

    # Loads in input file and use filter() to cut comment lines. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...
    lines = sc.textFile(sys.argv[1], 1).filter(lambda line: "#" not in line)

    # Loads all URLs from input file and initialize their neighbors.
    distinctLines = lines.map(lambda urls: parseNeighbors(urls)).distinct()
    #(fromNodeId1,[toNodeId1,toNodeId2,...]),(fromNodeId2,[toNodeId1,toNodeId2,...])
    links = distinctLines.groupByKey().cache()
    #count numbers of incoming links:node-number
    numOfIncomingLinksList = distinctLines.countByKey().items()
    

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(int(sys.argv[2])):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)
    
    #The top 20 highest ranked nodes in the decreasing order of their PageRank scores
    top20List=ranks.takeOrdered(20,key = lambda x:-x[1])
    
    #store all incoming links of each node
    linksList = links.mapValues(lambda x: list(x)).collect()
    #store node-rank for each node
    rankList = ranks.collect()
    
    # Collects top 20 URL ranks and dump them to console.
    for (link1, rank) in top20List:
        print("\n----------------------------------------------------------")
        print("%s has %s rank." % (link1, rank))
        for (link2, num) in numOfIncomingLinksList:
            if link1 == link2:
                print("Number of incoming links:%s." % num)
        #Average rank of neighbor nodes for each fromNode
        for (fromNodeId,toNodeIds) in linksList:
            #top20List.key=linksList.key
            if link1==fromNodeId:
                totalNeighborRank=0
                numToNodesIds=len(toNodeIds)
                avgNeighborRank=0
                #Iterate linksList.value
                for neighborNode in toNodeIds:
                    #get rank of neighbor nodes
                    for (node,rank) in rankList:
                        if neighborNode==node:
                            totalNeighborRank+=rank
                avgNeighborRank=totalNeighborRank/numToNodesIds
                print("The average rank of incomging links:%s." % avgNeighborRank)
        
    executionTime=time.time()-start 
    print("\nExecution time:%s\n"%executionTime)  
    sc.stop()