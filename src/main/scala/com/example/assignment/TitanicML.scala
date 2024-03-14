package com.example.assignment

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object TitanicML {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Titanic Analysis")
      .master("local[*]")
      .getOrCreate()

    // Define a UDF for our new feature
    val economicStatusIndex = spark.udf.register("economicStatusIndex", (age: Double, fare: Double) => if (age > 0) fare / age else 0)

    val df = spark.read.option("header", "true").option("inferSchema", "true").csv("titanic/train.csv")

    //1. Data Analysis
    // get the count of missing values for each column
    df.columns.foreach { colName =>
      println(s"Number of missing values in $colName: " + df.filter(df(colName).isNull || df(colName) === "").count())
    }

    //2. Feature Engineering
    // pick the average value to fill missing values
    val filledColumns = Map("Age" -> 30.0, "Embarked" -> "S", "Fare" -> 20.0)
    val dfFilled = df.na.fill(filledColumns)

    // Add EconomicStatus column
    val dfWithFeature = dfFilled.withColumn("EconomicStatus", economicStatusIndex(col("Age"), col("Fare")))

    // Convert string to vector
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    // Generate feature vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex", "EconomicStatus"))
      .setOutputCol("features")

    // Split the data
    val Array(trainingData, testData) = dfWithFeature.randomSplit(Array(0.8, 0.2))

    //3. Prediction
    // Logistic Regression Model
    val lr = new LogisticRegression().setLabelCol("Survived").setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, assembler, lr))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    // Evaluate the Model
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("Survived").setRawPredictionCol("rawPrediction")
    val accuracy = evaluator.evaluate(predictions)

    println(s"Accuracy: $accuracy")
  }
}
