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

  def classify(network: SimpleNetwork): Seq[ActualData.Prediction] = {
    val predictions = network.predict(X)
    logger.debug(s"Predictions made: ${predictions.rows}")
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
