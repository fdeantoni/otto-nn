package otto

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import grizzled.slf4j.Logging

class PrepareData(data: DenseMatrix[Double]) extends Logging {

  val ids: Ids = data(::, 1)

  val X: Features = {
    val parameters: DenseMatrix[Double] = data(::, 1 to (data.cols - 2))
    PrepareData.normalize(parameters)
  }

  val y: Labels = {
    val labels: DenseMatrix[Double] = DenseMatrix.eye[Double](9)
    val idx = data(::, data.cols - 1)
    val vecs = for(i <- 0 to idx.length - 1) yield {
      val classifier = idx(i).toInt - 1
      labels(classifier,::).inner.toScalaVector()
    }
    DenseMatrix(vecs:_*)
  }

}

object PrepareData {

  def normalize(data: DenseMatrix[Double]) = {
    val mv: DenseMatrix[MeanAndVariance] = meanAndVariance(data(::,*))
    val cols = for(i <- 0 to mv.cols - 1) yield {
      val mvi = mv(0,i)
      data(::,i).map( v => (v - mvi.mean) / sqrt(mvi.variance) )
    }
    DenseMatrix(cols.map(_.data):_*).t
  }

}
