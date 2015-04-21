package otto

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import org.scalatest.{Matchers, FunSuite}

class PrepareDataTest extends FunSuite with Matchers {

  test("Prepare training data") {
    val fileName = "src/test/resources/train_sample.csv"
    val loader = new DataLoader(fileName, 0.8, 0.2)
    val data = new PrepareData(loader.trainData)
    println("X:\n" + data.X)
    println("y:\n" + data.y)
    data.X.rows should be > 100
    data.y.rows should equal (data.X.rows)
    data.y.cols should equal (9)
  }


  test("Obtain feature normalization") {
    val dm = DenseMatrix((1.0,2.0,3.0),(1.0,2.0,3.0),(4.0,5.0,6.0))
    val mv: DenseMatrix[MeanAndVariance] = meanAndVariance(dm(::,*))
    val cols = for(i <- 0 to mv.cols - 1) yield {
      val mvi = mv(0,i)
      dm(::,i).map( v => (v - mvi.mean) / sqrt(mvi.variance) )
    }
    val result = DenseMatrix(cols.map(_.data):_*).t
    println(result)

  }

}
