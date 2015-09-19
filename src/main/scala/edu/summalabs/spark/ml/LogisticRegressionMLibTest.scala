package edu.summalabs.spark.ml

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Vectors, Vector}
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * @author Sergey Kovalev.
 */
object LogisticRegressionMLibTest {
  val conf = new SparkConf().setAppName("LogisticRegressionClassification")
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

    val featuresMatrix: RowMatrix = new RowMatrix(labeledPointsRDD.map(t => t._2))
    val pcaMatrix: Matrix = featuresMatrix.computePrincipalComponents(8)
    val featuresPcaMatrix: RowMatrix = featuresMatrix.multiply(pcaMatrix)

    val indexedFeaturesPcaMatrix: RDD[(Long ,Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(kv => (kv._2, kv._1))

    val labels: RDD[(Long, Double)] = labeledPointsRDD.map(t => t._1).zipWithIndex().map(kv => (kv._2, kv._1))

    val dataSet: RDD[LabeledPoint] = labels.join(indexedFeaturesPcaMatrix).map(m => {LabeledPoint(m._2._1, m._2._2)})

    // Split data into training (70%) and test (30%).
    val splits = dataSet.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Building the model
    val numIterations = 100
    val model = LinearRegressionWithSGD.train(training, numIterations)


    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    val valuesAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)
  }
}
