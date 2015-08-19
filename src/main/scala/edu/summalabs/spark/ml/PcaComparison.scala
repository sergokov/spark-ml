package edu.summalabs.spark.ml

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Comparison of existing PCA features calculation with PCA features calculated on Apache Spark
 * args(0) - path to feature files before PCA
 * args(1) - path to save PCA calculated by Spark
 * args(2) - path to already calculated PCA file
 * args(3) - path to save correlation coefficient
 * @author Sergey Kovalev.
 */
object PcaComparison {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("pca_comparison")
    val sc = new SparkContext(conf)

    val initFeatures = sc.textFile(args(0)).cache()
    val initFeaturesVector = initFeatures.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val initFeaturesMatrix: RowMatrix = new RowMatrix(initFeaturesVector)
    val pcaMatrix: Matrix = initFeaturesMatrix.computePrincipalComponents(8)
    val initFeaturesPcaMatrix: RowMatrix = initFeaturesMatrix.multiply(pcaMatrix)
    val indexInitFeaturesPca: RDD[(Long ,Vector)] = initFeaturesPcaMatrix.rows.zipWithIndex().map(kyVector => (kyVector._2, kyVector._1))

    indexInitFeaturesPca.saveAsTextFile(args(1))

    val featuresPca = sc.textFile(args(2)).cache()
    val featuresVectorPca = featuresPca.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresPcaMatrix: RowMatrix = new RowMatrix(featuresVectorPca)
    val indexFeaturesPca: RDD[(Long, Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(kyVector => (kyVector._2, kyVector._1))

    val joinedVectors = indexInitFeaturesPca.join(indexFeaturesPca)

    joinedVectors.map(pairs => {
      var pearson = new PearsonsCorrelation
      val correlationCoefficient = pearson.correlation(pairs._2._1.toArray, pairs._2._2.toArray)
      (pairs._1, correlationCoefficient)
    }).saveAsTextFile(args(3))
  }
}
