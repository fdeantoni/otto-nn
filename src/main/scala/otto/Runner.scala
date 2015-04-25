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
    val network = SimpleNetwork(93, 100, 9).train(trainData.X, trainData.y, lambda, iterations)
    val result = network.test(trainData.ids, trainData.X, trainData.y)
    println(s"Layers: ${layers.mkString(",")}")
    println(s"Lambda: $lambda")
    println(s"Iterations: $iterations")
    println(s"Training accuracy: ${result.accuracy}")
    println(s"Training logloss: ${result.logloss}")
    val testData = new PrepareData(loader.testData)
    val test = network.test(trainData.ids, testData.X, testData.y)
    println(s"Test Accuracy: ${test.accuracy}")
    println(s"Test logloss: ${test.logloss}")
    val errors = result.output.filter(sample => sample.probability < 0.8)
    println(s"Results with < 80% probability: ${errors.length}")
    println(errors.mkString("\n"))

  }



}
