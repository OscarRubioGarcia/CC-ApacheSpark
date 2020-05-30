import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F 
from pyspark.sql import SQLContext
from pyspark.sql.functions import col,sum,when
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4-ORG")
    sc = SparkContext(conf=conf)
    sqlC = SQLContext(sc)
    
    print("Version: " + sc.version)

    data = sqlC.read.csv("hdfs:/user/ccsa32891888/ECBDL14_IR2_Simplified_file.csv", header=False, sep=",", inferSchema=True)
    data.printSchema()    
    df = data.selectExpr("_c0 as PSSM_r2_minus1_V", "_c1 as AA_freq_global_F", "_c2 as PSSM_r2_3_R", "_c3 as PSSM_r1_0_Q", "_c4 as PSSM_r1_minus1_H", "_c5 as PSSM_central_1_I")
    df.printSchema()
    
    #Comprobacion de valores NAs
    df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show()

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
    
    for item in train_data.head(10): 
        print(item) 
        print('\n') 

    #Creador de vector de features
    assembler = VectorAssembler(
        inputCols=["PSSM_r2_minus1_V", "AA_freq_global_F", "PSSM_r2_3_R", "PSSM_r1_0_Q", "PSSM_r1_minus1_H"],
        outputCol="features")

    assembled_train = assembler.transform(train_data)
    assembled_train.select("features", "PSSM_central_1_I").show(truncate=False)
    training_set = assembled_train.select("features", "PSSM_central_1_I")

    #Split de los datos
    train_final, test_final = training_set.randomSplit([0.80, 0.20], seed = 13)
    train_final.describe().show() 
    test_final.describe().show()
    
    #Preprocesamiento de labels y rename de columnas
    train_final = train_final.withColumn('New', when(train_final.PSSM_central_1_I < -10, 0)
                                                    .when((train_final.PSSM_central_1_I >= -10) &
                                                          (train_final.PSSM_central_1_I <= -7), 1)
                                                    .when((train_final.PSSM_central_1_I >= -6) &
                                                          (train_final.PSSM_central_1_I <= -3), 2)
                                                    .when((train_final.PSSM_central_1_I >= -2) &
                                                          (train_final.PSSM_central_1_I <= 1), 3)
                                                    .when((train_final.PSSM_central_1_I >= 2) &
                                                          (train_final.PSSM_central_1_I <= 5), 4)
                                                    .when((train_final.PSSM_central_1_I >= 6), 5)
                                                               ) \
        .drop(train_final.PSSM_central_1_I) \
        .select(col('New').alias('label'), col('features'))
    test_final = test_final.withColumn('New', when(test_final.PSSM_central_1_I < -10, 0)
                                                    .when((test_final.PSSM_central_1_I >= -10) &
                                                          (test_final.PSSM_central_1_I <= -7), 1)
                                                    .when((test_final.PSSM_central_1_I >= -6) &
                                                          (test_final.PSSM_central_1_I <= -3), 2)
                                                    .when((test_final.PSSM_central_1_I >= -2) &
                                                          (test_final.PSSM_central_1_I <= 1), 3)
                                                    .when((test_final.PSSM_central_1_I >= 2) &
                                                          (test_final.PSSM_central_1_I <= 5), 4)
                                                    .when((test_final.PSSM_central_1_I >= 6), 5)
                                                               ) \
        .drop(test_final.PSSM_central_1_I) \
        .select(col('New').alias('label'), col('features'))
    
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_final)

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=6).fit(train_final)
    
    #Clasificador 3
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    model = pipeline.fit(train_final)

    predictions = model.transform(test_final)
    predictions.select("predictedLabel", "label", "features").show(5)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  

    sc.stop()
