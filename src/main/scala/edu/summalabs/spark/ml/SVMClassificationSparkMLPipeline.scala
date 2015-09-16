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
object SVMClassificationSparkMLPipeline {
  val conf = new SparkConf().setAppName("SVMClassificationSparkMLPipeline")
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
    val rawDF: DataFrame = labeledPointsRDD.toDF("label", "features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")

    val lr = new LogisticRegression()
      .setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(
      pca, lr
    ))

    val paramGrid = new ParamGridBuilder()
      .addGrid(pca.k, Array(8, 10))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val model: PipelineModel = pipeline.fit(rawDF)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(rawDF)

    val dataLinesTest = sc.textFile("/home/lab225/data/input/features.txt").cache()

    val labeledPointsRDDTest: RDD[(String, Vector)] = dataLinesTest.map(line => line.split(","))
      .map(err => {
      (
        err(0),
        Vectors.dense(err.slice(1, err.length).map(arr => arr.toDouble))
        )
    }
      )

    val rawDFTest: DataFrame = labeledPointsRDD.toDF("label", "features")

    cvModel.transform(rawDFTest)
      .select("label", "features", "probability", "prediction")
      .collect()
      .foreach { case Row(label: Long, features: String, prob: Vector, prediction: Double) =>
      println(s"($label, $features) --> prob=$prob, prediction=$prediction")
    }

  }
}
