#!/bin/bash

./spark-submit --class edu.summalabs.spark.ml.PcaRunner --master local[6] /home/lab225/data/input/spark-ml-1.0-SNAPSHOT-jar-with-dependencies.jar  /home/lab225/data/input/ptd2_list_id.csv_tupledsc_1024a_p2i_8_1_3_5_.csv /home/lab225/data/output/pca/ /home/lab225/data/input/ptd2_list_id.csv_tupledscPCA_1024a_p2i_8_1_3_5_.csv /home/lab225/data/output/correlation/
