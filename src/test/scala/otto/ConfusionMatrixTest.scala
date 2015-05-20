package otto

import breeze.linalg._
import org.scalatest.{Matchers, FunSuite}

class ConfusionMatrixTest extends FunSuite with Matchers {

  test("Formatting of confusion matrix") {
    val output1 = SimpleNetwork.TestOutput(1D, DenseVector(1D, 2D, 3D, 4D), 1, DenseVector(0.1, 0.1, 0.7, 0.1), 0.1, 3)  // observed 3 actual 1
    val output2 = SimpleNetwork.TestOutput(2D, DenseVector(5D, 1D, 3D, 2D), 4, DenseVector(0.1, 0.1, 0.2, 0.6), 0.6, 4) // observed 4 actual 4
    val output3 = SimpleNetwork.TestOutput(3D, DenseVector(2D, 6D, 4D, 8D), 1, DenseVector(0.5, 0.1, 0.2, 0.1), 0.5, 1) // observed 1 actual 1
    val output4 = SimpleNetwork.TestOutput(4D, DenseVector(5D, 2D, 3D, 2D), 2, DenseVector(0.8, 0.1, 0.05, 0.05), 0.1, 1) // observed 1 actual 2
    val results = SimpleNetwork.TestResults(0.5, 2.0, Seq(output2, output1, output4, output3), DenseMatrix.zeros(4,4), Seq(1,2,3,4))
    println(results.confusion.table)
    println(results.confusion)
  }

}
