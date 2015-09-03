package edu.summalabs.spark.ml

import java.io.{BufferedWriter, FileWriter}

/**
 * @author Sergey Kovalev.
 */
object Combiner {
  def main(args:Array[String]):Unit = {
    val tuples: Iterator[(Int, String, Int)] = scala.io.Source.fromFile("/home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv").getLines().zipWithIndex.map(t => (2, t._1, t._2))
    val tuples1: Iterator[(Int, String, Int)] = scala.io.Source.fromFile("/home/lab225/data/input/ptd2_list_id.csv_kovadesc_main.csv").getLines().zipWithIndex.map(t => (1, t._1, t._2))
    val w = new BufferedWriter(new FileWriter("/home/lab225/data/input/result.txt"))
    val groupBy: Map[Int, List[(Int, String, Int)]] = tuples.++(tuples1).toList.groupBy(_._3)
    for (l <- groupBy) {
      val line = _
      if (l._1 == 1) {
//        line +
      }

      w.write("" + "\\n")
    }
    w.close()
  }
}
