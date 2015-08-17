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
 * @author Sergey Kovalev.
 */
object PcaRunner {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("pca_analysis").setMaster("master-host")
    val sc = new SparkContext(conf)

    val initFeaturesFiles = "/home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv"
    val initFeatures = sc.textFile(initFeaturesFiles).cache()
//    initFeatures.zipWithIndex().map(t => (t._2, t._1)).filter(t => t._1 == 0).foreach(tapl => println(tapl._2))
    val initFeaturesVector = initFeatures.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val initFeaturesMatrix: RowMatrix = new RowMatrix(initFeaturesVector)
    val pcMatrix: Matrix = initFeaturesMatrix.computePrincipalComponents(8)
    val initFeaturesPcaMatrix: RowMatrix = initFeaturesMatrix.multiply(pcMatrix)
    val indexInitFeaturesPcaMatrix: RDD[(Long ,Vector)] = initFeaturesPcaMatrix.rows.zipWithIndex().map(t => (t._2, t._1))
    val filter1: RDD[(Long, Vector)] = indexInitFeaturesPcaMatrix.filter(t => t._1 == 1)
    filter1.foreach(tapl => println(tapl._2))

//    println("R number: " + pcMatrix.numRows + " C number: " + pcMatrix.numCols)
//    println("Principal components are:\n" + pcMatrix)

    val datafile = "/home/lab225/data/input/ptd2_list_id.csv_tupledscPCA_1024a_p2i_8_1_3_5_.csv"
    val featuresPca = sc.textFile(datafile).cache()
//    featuresPca.zipWithIndex().map(t => (t._2, t._1)).filter(t => t._1 == 0).foreach(tapl => println(tapl._2))
    val featuresVectorPca = featuresPca.map(line => line.split(",")).map(arr => arr.map(el => el.toDouble)).map(arr => Vectors.dense(arr))
    val featuresPcaMatrix: RowMatrix = new RowMatrix(featuresVectorPca)
    val indexFeaturesPcaMatrix: RDD[(Long, Vector)] = featuresPcaMatrix.rows.zipWithIndex().map(t => (t._2, t._1))
    val filter2: RDD[(Long, Vector)] = indexFeaturesPcaMatrix.filter(t => t._1 == 1)

//    filter1.foreach(tapl => println(tapl._2))
//    filter2.foreach(tapl => println(tapl._2))


    val join = indexInitFeaturesPcaMatrix.join(indexFeaturesPcaMatrix)

    join.map(pairs => {
      var personCorr = new PearsonsCorrelation
      var correlationCoefficient = personCorr.correlation(pairs._2._1.toArray, pairs._2._2.toArray)
      (pairs._1, correlationCoefficient)
    }).saveAsTextFile("/home/lab225/data/input/1/correlation.txt")

  }

}
