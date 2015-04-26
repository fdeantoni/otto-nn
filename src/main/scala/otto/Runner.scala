package otto

import breeze.linalg.{convert, DenseVector, DenseMatrix}
import grizzled.slf4j.Logging


object Runner extends Logging {

  val file = "src/main/resources/train_clean.csv"


  def main (args: Array[String] = Array.empty[String]): Unit = {
    one(100, 5)
  }

  def one(iterations: Int, lambda: Double, hidden: Int = 68, prune: Seq[Double] = Seq.empty): SimpleNetwork.TestResults = {
    val loader: DataLoader = new DataLoader(fileName = file, train = 0.8, test = 0.2)
    val trainData = new PrepareData(loader.trainData, prune = prune)
    val network = SimpleNetwork(93, hidden, 9).train(trainData.X, trainData.y, lambda, iterations)
    val result = network.test(trainData.ids, trainData.X, trainData.y)
    println(s"Layers: ${network.layers.mkString(",")}")
    println(s"Lambda: $lambda")
    println(s"Iterations: $iterations")
    println(s"Training accuracy: ${result.accuracy}")
    println(s"Training logloss: ${result.logloss}")
    val testData = new PrepareData(loader.testData)
    val test = network.test(trainData.ids, testData.X, testData.y)
    println(s"Test Accuracy: ${test.accuracy}")
    println(s"Test logloss: ${test.logloss}")
    println("Results:")
    println(s" < 0.5% probability: ${test.output.count(sample => sample.probability < 0.005)}")
    println(s" < 2.5% probability: ${test.output.count(sample => sample.probability < 0.025 && sample.probability > 0.005)}")
    println(s" < 5% probability: ${test.output.count(sample => sample.probability < 0.05 && sample.probability > 0.025)}")
    println(s" < 10% probability: ${test.output.count(sample => sample.probability < 0.1 && sample.probability > 0.05)}")
    println(s" < 25% probability: ${test.output.count(sample => sample.probability < 0.25 && sample.probability > 0.1)}")
    println(s" < 50% probability: ${test.output.count(sample => sample.probability < 0.5 && sample.probability > 0.25)}")
    println(s" < 75% probability: ${test.output.count(sample => sample.probability < 0.75 && sample.probability > 0.5)}")
    println(s" < 99% probability: ${test.output.count(sample => sample.probability < 0.99 && sample.probability > 0.75)}")
    println(s" > 99% probability: ${test.output.count(sample => sample.probability > 0.99)}")
    test
  }

  def filtering(iterations: Int, lambda: Double, hidden: Int = 68): (SimpleNetwork, SimpleNetwork.TestResults) = {
    val loader: DataLoader = new DataLoader(fileName = file, train = 0.9, test = 0.1)
    var network: SimpleNetwork = SimpleNetwork(93, hidden, 9)
    println(s"Layers: ${network.layers.mkString(",")}")
    println(s"Lambda: $lambda")
    println(s"Iterations: $iterations")
    var threshold = 100
    var error = Seq.empty[Double]
    while(threshold > 10) {
      println("------------------------------------------------------")
      val trainData = new PrepareData(loader.trainData, prune = error)
      network = network.train(trainData.X, trainData.y, lambda, iterations)
      val result = network.test(trainData.ids, trainData.X, trainData.y)
      println(s"Train accuracy: ${result.accuracy}")
      println(s"Train logloss: ${result.logloss}")
      val below = result.output.filter(sample => sample.probability < 0.005)
      threshold = below.length
      println(s" < 0.5% probability: $threshold")
      error = error ++ below.map(_.id)
    }
    val testData = new PrepareData(loader.testData)
    val test = network.test(testData.ids, testData.X, testData.y)
    println(s"Test accuracy: ${test.accuracy}")
    println(s"Test logloss: ${test.logloss}")
    println("Results:")
    println(s" < 0.5% probability: ${test.output.count(sample => sample.probability < 0.005)}")
    println(s" < 2.5% probability: ${test.output.count(sample => sample.probability < 0.025 && sample.probability > 0.005)}")
    println(s" < 5% probability: ${test.output.count(sample => sample.probability < 0.05 && sample.probability > 0.025)}")
    println(s" < 10% probability: ${test.output.count(sample => sample.probability < 0.1 && sample.probability > 0.05)}")
    println(s" < 25% probability: ${test.output.count(sample => sample.probability < 0.25 && sample.probability > 0.1)}")
    println(s" < 50% probability: ${test.output.count(sample => sample.probability < 0.5 && sample.probability > 0.25)}")
    println(s" < 75% probability: ${test.output.count(sample => sample.probability < 0.75 && sample.probability > 0.5)}")
    println(s" < 99% probability: ${test.output.count(sample => sample.probability < 0.99 && sample.probability > 0.75)}")
    println(s" > 99% probability: ${test.output.count(sample => sample.probability > 0.99)}")
    (network, test)
  }

  def submit(network: SimpleNetwork): Unit = {
    val actual = new ActualData("src/main/resources/test.csv")
    val results = actual.classify(network)
    val writer = new DataWriter(results)
    writer.save("target/submit.csv")
  }

  def save(network: SimpleNetwork): Unit = {
    network.save("target/network.json")
  }



}
