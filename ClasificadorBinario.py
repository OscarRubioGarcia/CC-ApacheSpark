import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F 
from pyspark.sql import SQLContext
from pyspark.sql.functions import col,sum,when
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline


if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4-ORG")
    sc = SparkContext(conf=conf)
    sqlC = SQLContext(sc)
    
    data = sqlC.read.csv("hdfs:/user/ccsa32891888/ECBDL14_IR2_Simplified_file.csv", header=False, sep=",", inferSchema=True)
    data.printSchema()    
    df = data.selectExpr("_c0 as PSSM_r2_minus1_V", "_c1 as AA_freq_global_F", "_c2 as PSSM_r2_3_R", "_c3 as PSSM_r1_0_Q", "_c4 as PSSM_r1_minus1_H", "_c5 as PSSM_central_1_I")
    df.printSchema()
    
    #Comprobacion de valores NAs
    #df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show()

    #Comprobacion de valores distintos
    for col in df:
        col_count = df.select(col).distinct().count()
        print("variable " + col + " , count " + col_count)
    
    #Comprobacion de desbalanceo
    targets = df.groupby('PSSM_central_1_I').count().collect()
    categories = [i[0] for i in targets]
    counts = [i[1] for i in targets]
    print(categories)
    print(counts)

    #Undersampling
    atribute0 = df.filter(F.col("PSSM_central_1_I")=='-13')
    atribute1 = df.filter(F.col("PSSM_central_1_I")=='-1')
    atribute2 = df.filter(F.col("PSSM_central_1_I")=='-10')
    atribute3 = df.filter(F.col("PSSM_central_1_I")=='-11')
    atribute4 = df.filter(F.col("PSSM_central_1_I")=='-15')
    atribute5 = df.filter(F.col("PSSM_central_1_I")=='1')
    atribute6 = df.filter(F.col("PSSM_central_1_I")=='6')
    atribute7 = df.filter(F.col("PSSM_central_1_I")=='3')
    atribute8 = df.filter(F.col("PSSM_central_1_I")=='-9')
    atribute9 = df.filter(F.col("PSSM_central_1_I")=='-7')
    atribute10 = df.filter(F.col("PSSM_central_1_I")=='-5')
    atribute11 = df.filter(F.col("PSSM_central_1_I")=='5')
    atribute12 = df.filter(F.col("PSSM_central_1_I")=='-8')
    atribute13 = df.filter(F.col("PSSM_central_1_I")=='-6')
    atribute14 = df.filter(F.col("PSSM_central_1_I")=='9')
    atribute15 = df.filter(F.col("PSSM_central_1_I")=='4')
    atribute16 = df.filter(F.col("PSSM_central_1_I")=='8')
    atribute17 = df.filter(F.col("PSSM_central_1_I")=='7')
    atribute18 = df.filter(F.col("PSSM_central_1_I")=='-4')
    atribute19 = df.filter(F.col("PSSM_central_1_I")=='-2')
    atribute20 = df.filter(F.col("PSSM_central_1_I")=='-12')
    atribute21 = df.filter(F.col("PSSM_central_1_I")=='2')
    atribute22 = df.filter(F.col("PSSM_central_1_I")=='-3')
    atribute23 = df.filter(F.col("PSSM_central_1_I")=='0')

    sampleRatio = 0.1
    sampleRatio2 = float(atribute8.count()) / float(df.count())
    atribute1sample = atribute1.sample(False, sampleRatio)
    atribute5sample = atribute5.sample(False, sampleRatio)
    atribute6sample = atribute6.sample(False, sampleRatio)
    atribute7sample = atribute7.sample(False, sampleRatio)
    atribute8sample = atribute8.sample(False, sampleRatio)
    atribute9sample = atribute9.sample(False, sampleRatio)
    atribute10sample = atribute10.sample(False, sampleRatio)
    atribute11sample = atribute11.sample(False, sampleRatio)
    atribute12sample = atribute12.sample(False, sampleRatio)
    atribute13sample = atribute13.sample(False, sampleRatio)
    atribute15sample = atribute15.sample(False, sampleRatio)
    atribute17sample = atribute17.sample(False, sampleRatio)
    atribute18sample = atribute18.sample(False, sampleRatio)
    atribute19sample = atribute19.sample(False, sampleRatio)
    atribute21sample = atribute21.sample(False, sampleRatio)
    atribute22sample = atribute22.sample(False, sampleRatio)
    atribute23sample = atribute23.sample(False, sampleRatio)
    
    train_data = atribute2.unionAll(atribute3)
    train_data = train_data.unionAll(atribute0)
    train_data = train_data.unionAll(atribute4)
    train_data = train_data.unionAll(atribute14)
    train_data = train_data.unionAll(atribute16)
    train_data = train_data.unionAll(atribute20)

    train_data = train_data.unionAll(atribute1sample)
    train_data = train_data.unionAll(atribute5sample)
    train_data = train_data.unionAll(atribute6sample)
    train_data = train_data.unionAll(atribute7sample)
    train_data = train_data.unionAll(atribute8sample)
    train_data = train_data.unionAll(atribute9sample)
    train_data = train_data.unionAll(atribute10sample)
    train_data = train_data.unionAll(atribute11sample)
    train_data = train_data.unionAll(atribute12sample)
    train_data = train_data.unionAll(atribute13sample)
    train_data = train_data.unionAll(atribute15sample)
    train_data = train_data.unionAll(atribute17sample)
    train_data = train_data.unionAll(atribute18sample)
    train_data = train_data.unionAll(atribute19sample)
    train_data = train_data.unionAll(atribute21sample)
    train_data = train_data.unionAll(atribute22sample)
    train_data = train_data.unionAll(atribute23sample)

    #Comprobacion de desbalanceo
    targets = train_data.groupby('PSSM_central_1_I').count().collect()
    categories = [i[0] for i in targets]
    counts = [i[1] for i in targets]
    print(categories)
    print(counts)

    #Split de los datos
    train_final, test_final = train_data.randomSplit([0.80, 0.20], seed = 13)
    
    #Creador de vector de features
    assembler = VectorAssembler(
        inputCols=["PSSM_r2_minus1_V", "AA_freq_global_F", "PSSM_r2_3_R", "PSSM_r1_0_Q", "PSSM_r1_minus1_H"],
        outputCol="features")

    assembled_train = assembler.transform(train_final)
    assembled_train.select("features", "PSSM_central_1_I").show(truncate=False)
    training_set = assembled_train.select("features", "PSSM_central_1_I")

    assembled_test = assembler.transform(test_final)
    assembled_test.select("features", "PSSM_central_1_I").show(truncate=False)
    
    #Prueba de clasificacion binaria
    finalset = training_set.withColumn('New',when(training_set.PSSM_central_1_I <= -5, 0)
                            .otherwise(1)).drop(training_set.PSSM_central_1_I)\
        .select(col('New').alias('label'),col('features'))

    #Clasificador 1
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(finalset)
    #lrModel.summary()
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))
    
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))

    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))


    # df.collect() <- NO!
    sc.stop()


