#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F

class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


def processItemSequence(spark, rawSampleDataPath):
    # rating data
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    # ratingSamples.show(5)
    # ratingSamples.printSchema()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds')) \
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))
    # userSeq.select("userId", "movieIdStr").show(10, truncate = False)
    return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' '))

def embeddingLSH(spark,moiveEmbMap):
    movieEmbSeq = []
    for key, embedding_list in moiveEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key,Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId","emb")
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb",outputCol="bucketId",bucketLength = 0.1, numHashTables = 3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)
    print("movieId,emb,bucketId schema:")
    embBucketResult.printSchema()
    print("movieId,emb bucketId data result:")
    embBucketResult.show(10,truncate = False)
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestneighbors(movieEmbDF,sampleEmb,5).show(truncate = False)


def trainItem2Vec(spark,samples,embLength,embOutputPath,saveToRedis,redisKeyPrefix):
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("158",20)
    for synonym,cosineSimilarity in synonyms:
        print(synonym,cosineSimilarity)
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    with open(embOutputDir,'w') as f:
        for movie_id in model.getVectors():
            vectors = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
            f.write(movie_id + ":" + vectors + "\n")
    embeddingLSH(spark,model.getVectors())
    return model

def generate_pair(x):
    pairSeq = []
    previousItem = ''
    for item in x:
        if not previousItem:
            previousItem = item
        else:
            pairSeq.append((previousItem,item))
            previousItem = item
    return pairSeq

def generateTransitionMatrix(samples):
    pairSamples = samples.flatMap(lambda x:generate_pair(x))
    pairCountMap = pairSamples.countByValue()
    pairTotalCount = 0
    transitionCountMatrix = defaultdict(dict)
    itemCountMap = defaultdict(int)
    for key,cnt in pairCountMap.items():
        key1,key2 = key
        transitionCountMatrix[key1][key2]  = cnt
        itemCountMap[key1] += cnt
        pairTotalCount += cnt
    trainsitionMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)
    for key1,transitionMap in transitionCountMatrix.items():
        for key2,cnt in transitionMap.items():
            trainsitionMatrix[key1][key2] = transitionCountMatrix[key1][key2] / itemCountMap[key1]
    for itemid,cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / pairTotalCount
    return transitionCountMatrix,itemDistribution

def oneRandomWalk(transitionMatrix,itemDistribution,sampleLength):
    sample = []
    randomDouble = random.random()
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in itemDistribution.items():
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break
    sample.append(firstItem)
    curElement = firstItem
    i = 1
    while i < sampleLength:
        if (curElement not in itemDistribution) or (curElement not in transitionMatrix):
            break
        probDistribution = transitionMatrix[curElement]
        randomDouble = random.random()
        accumulateProb = 0.0
        for item,prob in probDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)
        i += 1
    return sample

def randomWalk(transitionMatrix,itemDistribution,sampleCount,sampleLength):
    samples =[]
    for i in range(sampleCount):
        samples.append(oneRandomWalk(transitionMatrix,itemDistribution,sampleLength))
    return samples

def graphEmb(samples,spark,embLength,embOutputFilename,saveToRedis,rediskeyPrefix):
    transitionMatrix,itemDistribution = generateTransitionMatrix(samples)






if __name__ == '__main__':
    conf = SparkConf().setAppName('movie-rec').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    rawSampleDataPath = 'file:///D:\\self-doc\\self_project\\Search-Recommend-InAction\\data\\ml-25m\\ratings.csv'
    embLength = 10
    samples = processItemSequence(spark, rawSampleDataPath)
