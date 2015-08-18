package edu.summalabs.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

/**
 * @author Sergey Kovalev.
 */
object SVMRunner {
  val conf = new SparkConf().setAppName("pca_analysis")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
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

  def loadFeaturesAndApplyPCA(pathToFeatures:String): RDD[(Long ,Vector)] = {
    val features = sc.textFile(pathToFeatures).cache()
    val featuresVector = features.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresMatrix: RowMatrix = new RowMatrix(featuresVector)
    val pcaMatrix: Matrix = featuresMatrix.computePrincipalComponents(8)
    val featuresPcaMatrix: RowMatrix = featuresMatrix.multiply(pcaMatrix)
    val indexFeaturesPcaMatrix: RDD[(Long ,Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(kyVector => (kyVector._2, kyVector._1))
    indexFeaturesPcaMatrix
  }

  def loadLabels(pathToLabels:String): RDD[(Long ,Vector)] = {
    val features = sc.textFile(pathToLabels).cache()
    val featuresVector = features.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresMatrix: RowMatrix = new RowMatrix(featuresVector)
    val pcaMatrix: Matrix = featuresMatrix.computePrincipalComponents(8)
    val featuresPcaMatrix: RowMatrix = featuresMatrix.multiply(pcaMatrix)
    val indexFeaturesPcaMatrix: RDD[(Long ,Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(kyVector => (kyVector._2, kyVector._1))
    indexFeaturesPcaMatrix
  }
}
