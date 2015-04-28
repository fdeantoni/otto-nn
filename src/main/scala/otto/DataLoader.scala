package otto

import java.io.File

import breeze.numerics._
import breeze.linalg._
import grizzled.slf4j.Logging

class DataLoader(fileName: String, train: Double, test: Double) extends Logging {

  if(train + test > 1) throw new RuntimeException("The train + test ratio cannot be more than 1!")

  val data = csvread(file = new File(fileName), skipLines = 1)
  val classes = data(::,94).toScalaVector().distinct

  logger.info(s"Data size: ${data.rows}x${data.cols}")
  logger.info(s"Classes: $classes")

  case class RandomIndex(train: Seq[Int], test: Seq[Int], cv: Seq[Int])
  def randomIndexPerClass(c: Double) = {
    val index: List[Int] = (data(::,94) :== c).activeKeysIterator.toList
    val trainSize = round(index.length * train).toInt
    val testSize = round(index.length * test).toInt
    val randomIndex = DenseVector(util.Random.shuffle(index):_*)
    val trainIndex = randomIndex(0 to trainSize - 1).toScalaVector()
    val testIndex = randomIndex(trainSize to (trainSize + testSize - 1)).toScalaVector()
    val cvIndex = randomIndex( (trainSize + testSize) to index.length - 1 ).toScalaVector()
    RandomIndex(trainIndex, testIndex, cvIndex)
  }

  private var trainIndex: Seq[Int] = Seq.empty
  private var testIndex: Seq[Int] = Seq.empty
  private var cvIndex: Seq[Int] = Seq.empty

  for(c <- classes) {
    val indexes = randomIndexPerClass(c)
    trainIndex = trainIndex ++ indexes.train
    testIndex = testIndex ++ indexes.test
    cvIndex = cvIndex ++ indexes.cv
  }

  trainIndex = util.Random.shuffle(trainIndex)
  testIndex = util.Random.shuffle(testIndex)
  cvIndex = util.Random.shuffle(cvIndex)

  val trainSize = round(train * data.rows).toInt
  val testSize = round(test * data.rows).toInt
  val cvSize = round(data.rows - trainSize - testSize)

  logger.debug(s"Training index is:\n$trainIndex")
  logger.debug(s"Testing index is:\n$testIndex")
  logger.debug(s"CV index is:\n$cvIndex")

  val trainData: DenseMatrix[Double] = data(trainIndex, ::).toDenseMatrix
  val testData: DenseMatrix[Double] = data(testIndex, ::).toDenseMatrix
  val cvData: DenseMatrix[Double] = if(cvSize > 0) {
    data(cvIndex, ::).toDenseMatrix
  } else {
    DenseMatrix.zeros[Double](0,0)
  }


}
