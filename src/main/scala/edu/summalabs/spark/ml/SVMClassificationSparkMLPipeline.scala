package edu.summalabs.spark.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.Pipeline

/**
 * @author Sergey Kovalev.
 */
object SVMClassificationSparkMLPipeline {
  val conf = new SparkConf().setAppName("SVMClassification")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    val dataLines = sc.textFile("/home/lab225/data/input/features.txt").cache()

    val labeledPointsRDD: RDD[(String, Vector)] = dataLines.map(line => line.split(","))
      .map(err => {
         (
          err(0),
          Vectors.dense(err.slice(1, err.length).map(arr => arr.toDouble))
         )
        }
      )

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val rawDF: DataFrame = labeledPointsRDD.toDF("label", "vector")

    val pipeline = new Pipeline().setStages(Array(
    ))

  }
}
