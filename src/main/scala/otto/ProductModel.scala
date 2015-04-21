package otto

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import grizzled.slf4j.Logging
import nak.nnet.{NeuralNetwork, NNObjective}

class ProductModel(nn: NeuralNetwork) {

  def prediction(inputs: DenseVector[Double]): DenseVector[Double] = nn(inputs)

  def test(X: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {
    val results = ProductModel.asVector(X, y)
    val probabilities: IndexedSeq[Double] = for((p, actual) <- results) yield {
      log(sum(p * actual.t))
    }
    probabilities.foldLeft(0D)(_ + _) * (-1/X.rows)
  }

}

object ProductModel extends Logging {

  def train(X: DenseMatrix[Double], y: DenseMatrix[Double]): ProductModel = {

    logger.debug(s"Matrix X: ${X.rows}x${X.cols} \n${X(0 to 9, ::)}")
    logger.debug(s"Matrix y: \n${y(0 to 9, ::)}")

    val nnObj = new NNObjective[DenseVector[Double]](asVector(X, y), loss, Array(93, 45, 18, 9))
    val weights = minimize(nnObj, nnObj.initialWeightVector)
    val nn = nnObj.extract(weights)
    new ProductModel(nn)

  }

  private def loss(p: DenseVector[Double], y: DenseVector[Double]) = {
    val hx = sigmoid(p)
    val y0: DenseVector[Double] = (DenseVector.ones[Double](p.length) - y) :* log(DenseVector.ones[Double](p.length) - hx) :* -1D
    val y1 = -y :* log(hx)
    val error = sum(y0+y1)
    val grad = hx :* (DenseVector.ones[Double](p.length) - hx)
    logger.debug(s">>>>>>>>> ERROR/GRAD: $error / $grad")
    error -> grad
  }

  private def asVector(X: DenseMatrix[Double], y: DenseMatrix[Double]): IndexedSeq[(DenseVector[Double], DenseVector[Double])] = {
    Array.tabulate(X.rows)(i => (X(i,::).inner, y(i,::).inner)).toIndexedSeq
  }

}
