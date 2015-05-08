import breeze.linalg.{sum, DenseVector, DenseMatrix}
import breeze.numerics._

package object otto {

  type Ids = DenseVector[Double]
  type Features = DenseMatrix[Double]
  type Labels = DenseMatrix[Double]

  def logloss(prediction: DenseMatrix[Double], actual: DenseMatrix[Double]) = {
    val epsilon = 1e-15
    val probability = prediction.map { p =>
      if(p >= (1-epsilon)) 1-epsilon else if(p < epsilon) epsilon else p
    }
    val pos = actual:*log(probability)
    val neg = (1D-actual) :* log(1D-probability)
    sum(pos + neg) * (-1D/prediction.rows)
  }

}
