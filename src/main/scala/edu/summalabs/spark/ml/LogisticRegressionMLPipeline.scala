package edu.summalabs.spark.ml

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.Row

/**
 * @author Sergey Kovalev.
 */
object LogisticRegressionMLPipeline {
  val conf = new SparkConf().setAppName("SVMClassificationSparkMLPipeline")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    val dataLines = sc.textFile("/home/lab225/data/input/features.txt").cache()

    val labeledPointsRDD: RDD[(Double, Vector)] = dataLines.map(line => line.split(","))
      .map(err => {
         (
           if (err(0).equals("m")) 1.0 else 0.0,
           Vectors.dense(err.slice(1, err.length).map(arr => arr.toDouble))
         )
        }
      )

    val splits = labeledPointsRDD.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val rawDF: DataFrame = training.toDF("label", "features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(8)

    val lr = new LogisticRegression()
//      .setFeaturesCol("pcaFeatures")
      .setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(
      pca, lr
    ))

    val paramGrid = new ParamGridBuilder()
//      .addGrid(pca.k, Array(8, 10))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val model: PipelineModel = pipeline.fit(rawDF)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(rawDF)

    val rawDFTest: DataFrame = test.toDF("label", "features")

    val valuesAndPreds: RDD[Row] = cvModel.transform(rawDFTest).select("label", "prediction").rdd
    val MSE = valuesAndPreds.map{case Row(label: Double, prediction: Double) =>
      math.pow((label - prediction), 2)}.mean()

    println("training Mean Squared Error = " + MSE)
  }
}
