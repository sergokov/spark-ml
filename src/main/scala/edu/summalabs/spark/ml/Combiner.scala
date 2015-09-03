package edu.summalabs.spark.ml

import java.io.{BufferedWriter, FileWriter}

/**
 * @author Sergey Kovalev.
 */
object Combiner {
  def main(args:Array[String]):Unit = {
    val dataSet1: Iterator[(Int, String, Int)] = scala.io.Source.fromFile("/home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv").getLines().zipWithIndex.map(t => (2, t._1, t._2))
    println(dataSet1.length)
    val dataSet2: Iterator[(Int, String, Int)] = scala.io.Source.fromFile("/home/lab225/data/input/ptd2_list_id.csv_kovadesc_main.csv").getLines().filterNot(t => t.startsWith("#Idx,")).zipWithIndex.map(t => (1, t._1, t._2))
    println(dataSet2.length)
    val resultFile = new BufferedWriter(new FileWriter("/home/lab225/data/input/features.txt"))
    val joinedDataSet: List[(Int, String, Int)] = (dataSet1 ++ dataSet2).toList
    println(joinedDataSet.size)
    val groupedByIndex: Map[Int, List[(Int, String, Int)]] = joinedDataSet.groupBy(_._3)
    println(groupedByIndex.keys.size)
    for (l <- groupedByIndex) {
      var lable:String = null
      var fetures:String = null
      l._2.foreach(t => {
        if (t._1 == 1) {
          lable = t._2.split(",")(1)
        }
        if(t._1 == 2) {
          fetures =t._2
        }
      })
      resultFile.write(lable + "," + fetures + System.lineSeparator())
    }
    resultFile.close()
  }
}
