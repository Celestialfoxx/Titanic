package com.example.assignment

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

object TitanicAnalysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Titanic Data Analysis")
      .master("local[*]")
      .getOrCreate()

    val df = spark.read.option("header", "true").csv("titanic/train.csv")


    // 1. Average ticket fare for each Ticket class
    df.groupBy("Pclass")
      .agg(avg("Fare").alias("Average Fare"))
      .orderBy(col("Pclass"))
      .show()

    // 2. Survival percentage for each Ticket class
    val dfTyped = df.withColumn("Survived", col("Survived").cast(IntegerType))
    val survivalRate = dfTyped.groupBy("Pclass")
      .agg(expr("sum(Survived) / count(Survived)").alias("Survival Rate"))
      .orderBy(col("Pclass"))
    survivalRate.show()

    // 3. Possible passengers matching Rose DeWitt Bukater
    val possibleRose1 = df.filter("Age = 17 AND Sex = 'female' AND Pclass = 1 AND Parch = 1 AND Survived = 1")
        possibleRose1.show()
    val possibleRose2 = df.filter("Age = 17 AND Sex = 'female' AND Pclass = 1 AND (Parch + SibSp = 1) AND Survived = 1")
        possibleRose2.show()


    // 4. Possible passengers matching Jack Dawson
    val possibleJack = df.filter("Pclass = 3 AND Sex = 'male' AND Age between 19 AND 20 AND SibSp = 0 AND Parch = 0 AND Survived = 0")
    possibleJack.show()


    // 5. Relation between age groups and ticket fare, and survival
    val dfTyped2 = df.withColumn("Survived", col("Survived").cast(IntegerType))
      .withColumn("Age", col("Age").cast(IntegerType))

    val ageGrouped = dfTyped2.withColumn("AgeGroup", (col("Age") / 10).cast("int") * 10)

    ageGrouped.groupBy("AgeGroup")
      .agg(
        avg("Fare").alias("Average Fare"),
        sum("Survived").alias("Total Survived")
      )
      .orderBy(col("AgeGroup").asc)
      .show()

    val survivalRateByAgeGroup = ageGrouped.groupBy("AgeGroup")
      .agg(
        sum("Survived").alias("Number Survived"),
        count("Survived").alias("Total in Group"),
        (sum("Survived") / count("Survived") * 100).alias("Survival Rate")
      )
      .orderBy(col("Survival Rate").desc)

    survivalRateByAgeGroup.show()

    spark.stop()
  }
}
