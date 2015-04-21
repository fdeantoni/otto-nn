package otto

import breeze.linalg.{convert, DenseVector, DenseMatrix}
import grizzled.slf4j.Logging


object Runner extends Logging {

  val file = "src/main/resources/train_clean.csv"
  //val file = "src/test/resources/train_sample.csv"
  val loader = new DataLoader(file, 0.8, 0.2)

  def main (args: Array[String]) {

    val model = train
    val logloss = test(model)
    logger.info(s"Trained model's logloss: $logloss")

  }

  def train = {
    val X = loader.trainData(::, 1 to (loader.trainData.cols - 2))
    val y = {
      val labels: DenseMatrix[Double] = DenseMatrix.eye[Double](9)
      val idx = loader.trainData(::, loader.trainData.cols - 1)
      val vecs = for(i <- 0 to idx.length - 1) yield {
        val classifier = idx(i).toInt - 1
        labels(classifier,::).inner.toScalaVector()
      }
      DenseMatrix(vecs:_*)
    }
    ProductModel.train(X, y)
  }

  def test(model: ProductModel) = {
    val X = loader.testData(::, 1 to (loader.testData.cols - 2))
    val y = {
      val labels: DenseMatrix[Double] = DenseMatrix.eye[Double](9)
      val idx = loader.testData(::, loader.testData.cols - 1)
      val vecs = for(i <- 0 to idx.length - 1) yield {
        val classifier = idx(i).toInt - 1
        labels(classifier,::).inner.toScalaVector()
      }
      DenseMatrix(vecs:_*)
    }
    model.test(X, y)
  }

}
