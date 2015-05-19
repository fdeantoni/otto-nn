package otto

import grizzled.slf4j.Logging


object Runner extends Logging {

  val file = "src/main/resources/train_clean.csv"
  val outputs = 9

  def main (args: Array[String] = Array.empty[String]): Unit = {
    val (network, _) = one()
    save(network)
    submit(network)
  }

  def one(iterations: Int = 500, lambda: Double = 0.5, hidden: Int = 300, prune: Seq[Double] = Seq.empty): (SimpleNetwork, SimpleNetwork.TestResults) = {
    val loader: DataLoader = new DataLoader(fileName = file, train = 0.8, test = 0.2)
    val trainData = new PrepareData(loader.trainData, prune = prune)
    val network = SimpleNetwork(trainData.X.cols, hidden, outputs).train(trainData.X, trainData.y, lambda, iterations)
    println(s"Layers: ${network.layers.mkString(",")}")
    println(s"Lambda: $lambda")
    println(s"Iterations: $iterations")
    val result = network.test(trainData.ids, trainData.X, trainData.y)
    println(s"Training accuracy: ${result.accuracy}")
    println(s"Training logloss: ${result.logloss}")
    val testData = new PrepareData(loader.testData)
    val test = network.test(trainData.ids, testData.X, testData.y)
    println(s"Test Accuracy: ${test.accuracy}")
    println(s"Test logloss: ${test.logloss}")
    println("Results:")
    println(s" < 0.5% probability: ${test.output.count(sample => sample.actual < 0.005)}")
    println(s" < 2.5% probability: ${test.output.count(sample => sample.actual < 0.025 && sample.actual > 0.005)}")
    println(s" < 5% probability: ${test.output.count(sample => sample.actual < 0.05 && sample.actual > 0.025)}")
    println(s" < 10% probability: ${test.output.count(sample => sample.actual < 0.1 && sample.actual > 0.05)}")
    println(s" < 25% probability: ${test.output.count(sample => sample.actual < 0.25 && sample.actual > 0.1)}")
    println(s" < 50% probability: ${test.output.count(sample => sample.actual < 0.5 && sample.actual > 0.25)}")
    println(s" < 75% probability: ${test.output.count(sample => sample.actual < 0.75 && sample.actual > 0.5)}")
    println(s" < 99% probability: ${test.output.count(sample => sample.actual < 0.99 && sample.actual > 0.75)}")
    println(s" > 99% probability: ${test.output.count(sample => sample.actual > 0.99)}")
    (network, test)
  }

  def submit(network: SimpleNetwork, file: String = "target/submit.csv"): Unit = {
    val actual = new ActualData("src/main/resources/test.csv")
    val results = actual.classify(network)
    val writer = new DataWriter(results)
    writer.save(file)
  }

  def load(file: String): SimpleNetwork = {
    SimpleNetwork.load(file)
  }

  def save(network: SimpleNetwork, file: String = "target/network.json"): Unit = {
    network.save(file)
  }

}
