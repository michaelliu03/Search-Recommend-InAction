#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
from __future__ import print_function
import sys
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import MatrixFactorizationModel

def create_rating(rating_record):
    tokens = rating_record.split(',')
    userID = int(tokens[0])
    productID = int(tokens[1])
    rating = float(tokens[2])
    return Rating(userID,productID,rating)

spark = SparkSession.builder.getOrCreate()

input_path = sys.argv[1]
rank = int(sys.argv[2])
num_of_iterations = int(sys.argv[3])

data = spark.sparkContext.textFile(input_path)
ratings = data.map(create_rating)
rank =10
num_of_iterations = 10
model = ALS.train(ratings,rank,num_of_iterations)

test_data = ratings.map(lambda r:(r[0],r[1]))
predictions = model.predictAll(test_data) \
                   .map(lambda r: ((r[0],r[1]),r[2]))

rates_and_predictions = ratings.map(lambda r:((r[0],r[1]),r[2])) \
                               .join(predictions)

MSE = rates_and_predictions.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error =" + str(MSE))


saved_path = "../chapter10/myCollaborativeFilter"
model.save(spark.sparkContext, saved_path)

same_model = MatrixFactorizationModel.load(spark.sparkContext, saved_path)

spark.stop()