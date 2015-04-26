package otto

import java.io.File

import breeze.linalg._

class ActualData(fileName: String) {

  val data = csvread(file = new File(fileName), skipLines = 1)

  val ids: Ids = data(::, 0)

  val X: Features = {
    val parameters: DenseMatrix[Double] = data(::, 1 to data.cols)
    PrepareData.normalize(parameters)
  }

  def classify(network: SimpleNetwork): Seq[ActualData.Prediction] = {
    val predictions = network.predict(X)
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
