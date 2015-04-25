package otto

import breeze.linalg.{convert, DenseVector, DenseMatrix}
import grizzled.slf4j.Logging


object Runner extends Logging {

  val file = "src/main/resources/train_clean.csv"
  val loader = new DataLoader(file, 0.8, 0.2)

  def main (args: Array[String]): Unit = {
    val layers = Seq(93, 500, 9)
    val lambda = 5
    val iterations = 1000
    val trainData = new PrepareData(loader.trainData)
    val network = SimpleNetwork(93, 100, 9)
    network.train(trainData.X, trainData.y, lambda, iterations)
    val (trainError, trainLogloss) = network.test(trainData.X, trainData.y)
    println(s"Layers: ${layers.mkString(",")}")
    println(s"Lambda: $lambda")
    println(s"Iterations: $iterations")
    println(s"Training accuracy: $trainError")
    println(s"Training logloss: $trainLogloss")
    val testData = new PrepareData(loader.testData)
    val (testError, testLogloss) = network.test(testData.X, testData.y)
    println(s"Test Accuracy: $testError")
    println(s"Test logloss: $testLogloss")

  }



}
