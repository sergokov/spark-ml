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
  val conf = new SparkConf().setAppName("pca_analysis")
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    val features = sc.textFile("/home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv")
    val featuresVector = features.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresMatrix: RowMatrix = new RowMatrix(featuresVector)
    val pcaMatrix: Matrix = featuresMatrix.computePrincipalComponents(8)
    val featuresPcaMatrix: RowMatrix = featuresMatrix.multiply(pcaMatrix)
    val indexFeaturesPcaMatrix: RDD[(Long ,Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(kyVector => (kyVector._2, kyVector._1))

    val main = sc.textFile("/home/lab225/data/input/ptd2_list_id.csv_kovadesc_main.csv")
    val labels: RDD[(Long, Int)] = main.filter(line => !line.contains("#Idx")).map(line => line.split(","))
      .map(arr => arr.slice(0, 2))
      .map(arr => {
      var label = if (arr(1).equals("m")) 1 else 0
      (arr(0).toLong, label)
    })

    val data: RDD[LabeledPoint] = labels.join(indexFeaturesPcaMatrix).map(m => {LabeledPoint(m._2._1, m._2._2)})
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

  def loadLabels(pathToLabels:String): RDD[(Long ,Int)] = {
    val features = sc.textFile(pathToLabels).cache()
    features.filter(line => !line.contains("#Idx"))
      .map(line => line.split(","))
      .map(arr => arr.slice(0, 2))
      .map(arr => {
      var label = if (arr(1).equals("m")) 1 else 0
      (arr(0).toLong, label)
    })
  }
}
