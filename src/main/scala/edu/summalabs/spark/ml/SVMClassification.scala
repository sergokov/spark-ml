package edu.summalabs.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

/**
 * @author Sergey Kovalev.
 */
object SVMClassification {
  val conf = new SparkConf().setAppName("SVMClassification")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    val dataLines = sc.textFile("/home/lab225/data/input/features.txt").cache()
    val splittedData: RDD[Array[String]] = dataLines.map(line => line.split(","))
    val featuresVector = splittedData.map(err => err.slice(1, err.length)).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresMatrix: RowMatrix = new RowMatrix(featuresVector)
    val pcaMatrix: Matrix = featuresMatrix.computePrincipalComponents(8)
    val featuresPcaMatrix: RowMatrix = featuresMatrix.multiply(pcaMatrix)
    val indexedFeaturesPcaMatrix: RDD[(Long ,Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(kv => (kv._2, kv._1))

    val labels: RDD[(Long, Int)] = splittedData.map(arr => arr.slice(0, 1)).
      map(arr => {if (arr(0).equals("m")) 1 else 0}).zipWithIndex().map(kv => (kv._2, kv._1))

    val dataSet: RDD[LabeledPoint] = labels.join(indexedFeaturesPcaMatrix).map(m => {LabeledPoint(m._2._1, m._2._2)})
    // Split data into training (60%) and test (40%).
    val splits = dataSet.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    // Save and load model
    model.save(sc, "myModelPath")
    val sameModel = SVMModel.load(sc, "myModelPath")
  }
}
