import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F 
from pyspark.sql import SQLContext
from pyspark.sql.functions import col,sum,when
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4-ORG")
    sc = SparkContext(conf=conf)
    sqlC = SQLContext(sc)
    
    print("Version: " + sc.version)
    
    data = sqlC.read.csv("hdfs:/user/ccsa32891888/filteredC-small-training.csv", header=False, sep=",", inferSchema=True)
    data.printSchema()    
    df = data.selectExpr("_c0 as PSSM_r2_minus1_V", "_c1 as AA_freq_global_F", "_c2 as PSSM_r2_3_R", "_c3 as PSSM_r1_0_Q", "_c4 as PSSM_r1_minus1_H", "_c5 as PSSM_central_1_I", "_c6 as Class")
    df.printSchema()
    
    #Comprobacion de valores NAs
    df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show()

    #Comprobacion de valores distintos
    for col in df:
        col_count = df.select(col).distinct().count()
        print("variable " + col + " , count " + col_count)
    
    #Comprobacion de desbalanceo
    targets = df.groupby('Class').count().collect()
    categories = [i[0] for i in targets]
    counts = [i[1] for i in targets]
    print(categories)
    print(counts)

    #Undersampling
    atribute0 = df.filter(F.col("Class")=='0')
    atribute1 = df.filter(F.col("Class")=='1')

    sampleRatio = float(atribute1.count()) / float(df.count())
    atribute1sample = atribute0.sample(False, sampleRatio)
    
    train_data = atribute0.unionAll(atribute1)

    #Creador de vector de features
    assembler = VectorAssembler(
        inputCols=["PSSM_r2_minus1_V", "AA_freq_global_F", "PSSM_r2_3_R", "PSSM_r1_0_Q", "PSSM_r1_minus1_H", "PSSM_central_1_I"],
        outputCol="features")

    assembled_train = assembler.transform(train_data)
    assembled_train.select("features", "Class").show(truncate=False)
    training_set = assembled_train.select("features", "Class")

    #Split de los datos
    train_final, test_final = training_set.randomSplit([0.80, 0.20], seed = 13)
    train_final.describe().show() 
    test_final.describe().show()
	
    train_final = train_final.selectExpr("Class as label", "features as features")
    test_final = test_final.selectExpr("Class as label", "features as features")
    
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scalerModel = scaler.fit(train_final)
    scaledTData = scalerModel.transform(train_final)
    scaledTData = scaledTData.select("label", "scaledFeatures")
    scaledTData = scaledTData.selectExpr("label as label", "scaledFeatures as features")

    scalerModel = scaler.fit(test_final)
    scaledFData = scalerModel.transform(test_final)
    scaledFData = scaledFData.select("label", "scaledFeatures")
    scaledFData = scaledFData.selectExpr("label as label", "scaledFeatures as features")

    #Clasificador 2
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

    # train the model
    model = nb.fit(scaledTData)

    # select example rows to display.
    predictions = model.transform(scaledFData)
    predictions.show()

    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))

    sc.stop()
