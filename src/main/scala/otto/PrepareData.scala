package otto

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import grizzled.slf4j.Logging

class PrepareData(data: DenseMatrix[Double], prune: Seq[Double] = Seq.empty) extends Logging {

  private var pruned: DenseMatrix[Double] = data

  for(id <- prune) {
    val selector: DenseVector[Int] = (pruned(::,0) :== id).map(item => if(item) 1 else 0)
    val row: Int = argmax(selector)
    pruned = pruned.delete(row, Axis._0)
  }

  val ids: Ids = pruned(::, 0)

  val X: Features = {
    val parameters: DenseMatrix[Double] = pruned(::, 1 to (data.cols - 2))
    PrepareData.normalize(parameters)
  }

  val y: Labels = {
    val labels: DenseMatrix[Double] = DenseMatrix.eye[Double](9)
    val idx = pruned(::, data.cols - 1)
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
