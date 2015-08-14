package edu.lab225.spark.pca

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

/**
 * @author Sergey Kovalev.
 */
object PcaRunner {

  def main(args: Array[String]) {
    val initFeaturesFiles = "/home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv"
    val conf = new SparkConf().setAppName("pca_analysis").setMaster("master-host")
    val sc = new SparkContext(conf)
    val initFeatures = sc.textFile(initFeaturesFiles).cache()
    val initFeaturesVector = initFeatures.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val initFeaturesMatrix: RowMatrix = new RowMatrix(initFeaturesVector)
    val pcMatrix: Matrix = initFeaturesMatrix.computePrincipalComponents(5)
    val initFeaturesPcaMatrix: RowMatrix = initFeaturesMatrix.multiply(pcMatrix)
    val indexInitFeaturesPcaMatrix: RDD[(Vector, Long)] = initFeaturesPcaMatrix.rows.zipWithIndex()

    println("R number: " + pcMatrix.numRows + " C number: " + pcMatrix.numCols)
    println("Principal components are:\n" + pcMatrix)

    val datafile = "/home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv"
    val featuresPca = sc.textFile(initFeaturesFiles).cache()
    val featuresVectorPca = featuresPca.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresPcaMatrix: RowMatrix = new RowMatrix(featuresVectorPca)
    val indexFeaturesPcaMatrix: RDD[(Vector, Long)] = featuresPcaMatrix.rows.zipWithIndex()

    val map: RDD[(Long, Vector)] = indexInitFeaturesPcaMatrix.union(indexFeaturesPcaMatrix).map(t => (t._2, t._1))
  }

}
