package otto

import breeze.linalg.{convert, DenseVector, DenseMatrix}
import grizzled.slf4j.Logging


object Runner extends Logging {

  val file = "src/main/resources/train_clean.csv"
  //val file = "src/test/resources/train_sample.csv"
  val loader = new DataLoader(file, 0.8, 0.2)

  def main (args: Array[String]): Unit = {

  }



}
