package otto

import breeze.linalg.DenseMatrix
import org.scalatest.{Matchers, FunSuite}

class LoglossTest extends FunSuite with Matchers  {

  test("Check logloss function") {
    val actual = DenseMatrix( (1D, 0D, 0D), (0D, 1D, 0D), (0D, 0D, 1D) )
    val prediction = DenseMatrix( (0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.6, 0.3, 0.1) )
    val result = logloss(prediction, actual)
    println(result)
    result should be > (2D)
  }

  test("Check extremes") {
    val actual = DenseMatrix( (1D, 0D, 0D), (0D, 1D, 0D), (0D, 0D, 1D) )
    val prediction = DenseMatrix( (1D, 0.0, 0.0), (1D, 0.0, 0.0), (0.0, 1D, 0.0) )
    val result = logloss(prediction, actual)
    println(result)
    result should be > (45D)
  }

}
