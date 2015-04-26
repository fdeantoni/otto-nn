package otto

import java.io.File

import breeze.numerics._
import breeze.linalg._
import grizzled.slf4j.Logging

class DataLoader(fileName: String, train: Double, test: Double) extends Logging {

  val data = csvread(file = new File(fileName), skipLines = 1)

  logger.info(s"Data size: ${data.rows}x${data.cols}")

  val trainSize = round(train * data.rows).toInt
  val testSize = round(test * data.rows).toInt
  val cvSize = round(data.rows - trainSize - testSize)

  logger.info(s"Training rows: " + trainSize)
  logger.info(s"Testing rows: " + testSize)
  logger.info(s"CV rows: " + cvSize)

  private val randomIndex =  DenseVector(util.Random.shuffle(0 to data.rows - 1).seq.toArray)
  private val trainIndex = randomIndex(0 to trainSize - 1).toScalaVector()
  private val testIndex = randomIndex(trainSize to (trainSize + testSize - 1)).toScalaVector()
  private val cvIndex = randomIndex( (trainSize + testSize) to data.rows - 1 ).toScalaVector()

  logger.debug(s"randomIndex is:\n$randomIndex")
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
