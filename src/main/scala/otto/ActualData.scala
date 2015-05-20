package otto

import java.io.File

import breeze.linalg._
import grizzled.slf4j.Logging

class ActualData(fileName: String) extends Logging {

  val data = csvread(file = new File(fileName), skipLines = 1)

  val ids: Ids = data(::, 0)

  val X: Features = {
    val parameters: DenseMatrix[Double] = data(::, 1 to data.cols - 1)
    FeatureNormalize.log10(parameters)
  }

  def classify(networks: Seq[SimpleNetwork]): Seq[ActualData.Prediction] = {
    assert(networks.length > 0, "Number of networks must be at least 1!")
    var predictions: DenseMatrix[Double] = networks.head.predict(X)
    for(network <- networks.drop(1)) {
      predictions = network.predict(predictions)
    }
    Seq.tabulate(predictions.rows){ i =>
      val id = ids(i)
      val prediction = predictions(i, ::).inner
      ActualData.Prediction(id, prediction)
    }
  }

}

object ActualData {
  case class Prediction(id: Double, probability: DenseVector[Double])
}
