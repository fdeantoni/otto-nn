package otto

import breeze.linalg.{convert, DenseVector, DenseMatrix}
import grizzled.slf4j.Logging


object Runner extends Logging {

  val file = "src/main/resources/train_clean.csv"
  val loader = new DataLoader(file, 0.8, 0.2)

  def main (args: Array[String]): Unit = {

    val trainData = new PrepareData(loader.trainData)
    val network = new SimpleNetwork(Seq(93, 68, 9))
    network.train(trainData.X, trainData.y, 1.0, 300)
    val (trainError, trainLogloss) = network.test(trainData.X, trainData.y)
    println(s"Accuracy: $trainError")
    println(s"Logloss: $trainLogloss")
    val testData = new PrepareData(loader.testData)
    val (testError, testLogloss) = network.test(testData.X, testData.y)
    println(s"Accuracy: $testError")
    println(s"Logloss: $testLogloss")

  }



}
