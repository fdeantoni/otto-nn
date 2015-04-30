package otto

import breeze.linalg._
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
    FeatureNormalize.log10(parameters)
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

  logger.info(s"Count of samples per classes: ${pruned(::, data.cols - 1).toScalaVector().groupBy(c => c).map(t => (t._1, t._2.length.toDouble))}")

}

