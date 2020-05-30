import sys
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext

if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4-ORG")
    sc = SparkContext(conf=conf)
    sqlC = SQLContext(sc)
    
    data = sqlC.read.csv("hdfs:/user/ccsa32891888/ECBDL14_IR2.data", header=False, sep=",", inferSchema=True)
    data_formated = data.selectExpr("_c430 as PSSM_r2_minus1_V", "_c164 as AA_freq_global_F", "_c492 as PSSM_r2_3_R", "_c256 as PSSM_r1_0_Q", "_c239 as PSSM_r1_minus1_H", "_c600 as PSSM_central_1_I", "_c631 as class")
    df = data_formated.select("PSSM_r2_minus1_V", "AA_freq_global_F", "PSSM_r2_3_R", "PSSM_r1_0_Q", "PSSM_r1_minus1_H", "PSSM_central_1_I", "class")

    df.printSchema()
    df.coalesce(1).write.csv("hdfs:/user/ccsa32891888/filteredC-small-training.csv")
    sc.stop()
