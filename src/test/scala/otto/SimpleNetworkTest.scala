package otto

import java.io.File

import breeze.linalg._
import breeze.numerics._
import org.scalatest.{Matchers, FunSuite}

class SimpleNetworkTest extends FunSuite with Matchers {

  test("Create a network") {
    val network = SimpleNetwork(1, 5, 2)
    println(s"Thetas:\n" + network.thetas)
    network.thetas.w1.rows should equal (5)
    network.thetas.w1.cols should equal (1)
    network.thetas.w2.rows should equal (2)
    network.thetas.w2.cols should equal (5)
  }

  test("Cost function check") {
    val w1 = DenseMatrix(
      (0.1, 0.3, 0.5),
      (0.2, 0.4, 0.6)
    )
    val w2 = DenseMatrix(
      (0.7, 1.1, 1.5),
      (0.8, 1.2, 1.6),
      (0.9, 1.3, 1.7),
      (1.0, 1.4, 1.8)
    )
    val X = cos(DenseMatrix( (1, 2), (3, 4), (5, 6) ))
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D) )
    val lambda = 3
    val thetas = SimpleNetwork.Thetas(w1, w2)
    val result = thetas.costFunction(X, y, lambda)
    println(result)
    result._1 should be (16.457 +- 1e-3)
  }

  test("Gradient function check without regularization") {
    val w1 = DenseMatrix(
      (0.1, 0.3, 0.5),
      (0.2, 0.4, 0.6)
    )
    val w2 = DenseMatrix(
      (0.7, 1.1, 1.5),
      (0.8, 1.2, 1.6),
      (0.9, 1.3, 1.7),
      (1.0, 1.4, 1.8)
    )
    val X = cos(DenseMatrix( (1, 2), (3, 4), (5, 6) ))
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D) )
    val lambda = 0
    val thetas = SimpleNetwork.Thetas(w1, w2)
    val result = thetas.costFunction(X, y, lambda)
    println(result)
    result._1 should be (7.407 +- 1e-3)
    result._2.length should equal (18)
    result._2(0) should be (0.766138 +- 1e-5)
    result._2(17) should be (0.322331 +- 1e-5)
  }

  test("Gradient function check with regularization") {
    val w1 = DenseMatrix(
      (0.1, 0.3, 0.5),
      (0.2, 0.4, 0.6)
    )
    val w2 = DenseMatrix(
      (0.7, 1.1, 1.5),
      (0.8, 1.2, 1.6),
      (0.9, 1.3, 1.7),
      (1.0, 1.4, 1.8)
    )
    val X = cos(DenseMatrix( (1, 2), (3, 4), (5, 6) ))
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D) )
    val lambda = 3
    val thetas = SimpleNetwork.Thetas(w1, w2)
    val result = thetas.costFunction(X, y, lambda)
    println(result)
    result._1 should be (16.457 +- 1e-3)
    result._2.length should equal (18)
    result._2(0) should be (0.76614 +- 1e-5)
    result._2(17) should be (2.12233 +- 1e-5)
  }

  test("Gradient function check with regularization with additional samples") {
    val w1 = DenseMatrix(
      (0.1, 0.3, 0.5),
      (0.2, 0.4, 0.6)
    )
    val w2 = DenseMatrix(
      (0.7, 1.1, 1.5),
      (0.8, 1.2, 1.6),
      (0.9, 1.3, 1.7),
      (1.0, 1.4, 1.8)
    )
    val X = DenseMatrix( (1D, 2D), (3D, 4D), (5D, 6D), (0D, 1D), (1D, 2D) )
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (1D,0D,0D,0D), (0D,1D,0D,0D) )
    val lambda = 4
    val thetas = SimpleNetwork.Thetas(w1, w2)
    val result = thetas.costFunction(X, y, lambda)
    println(result)
    result._1 should be (17.441 +- 1e-3)
    result._2.length should equal (18)
    result._2(0) should be (0.42849 +- 1e-5)
    result._2(17) should be (2.1242 +- 1e-5)
  }

  test("A simple neural network") {
    val Xt: DenseMatrix[Double] = {
      val data = DenseMatrix(
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D),
        (1D, 2D), (3D, 4D), (5D, 6D), (7D, 8D)
      )
      PrepareData.normalize(data)
    }
    val yt = DenseMatrix(
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D),
      (1D, 0D, 0D , 0D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (0D,0D,0D,1D)
    )
    var X = Xt
    var y = yt
    for(i <- 1 to 100) {
      X = DenseMatrix.vertcat(X, Xt)
      y = DenseMatrix.vertcat(y, yt)
    }
    val ids = DenseVector( Seq.tabulate(X.rows){i => i.toDouble}:_* )
    println(s"Size of ids: ${ids.length}")
    println(s"Size of X: ${X.rows}x${X.cols}")
    println(s"Size of y: ${y.rows}x${y.cols}")
    val lambda = 0
    val trained = SimpleNetwork(2, 4, 4).train(X, y, lambda, 50)
    val result = trained.test(ids, X, y)
    println(s"Accuracy: ${result.accuracy}")
    println(s"Logloss: ${result.logloss}")
    result.accuracy should (be > 0.95)
    val errors = result.output.filter(sample => sample.probability < 0.8)
    println(s"Results with < 99% probability: ${errors.length}")
    println(errors.mkString("\n"))
  }

  test("A neural network with otto data") {
    val fileName = "src/main/resources/train_clean.csv"
    val loader = new DataLoader(fileName, 0.8, 0.2)
    val data = new PrepareData(loader.trainData)
    val network = SimpleNetwork(93, 68, 9).train(data.X, data.y, 1.0, 100)
    val result = network.test(data.ids, data.X, data.y)
    println(s"Accuracy: ${result.accuracy}")
    println(s"Logloss: ${result.logloss}")
    val testData = new PrepareData(loader.testData)
    val test = network.test(testData.ids, testData.X, testData.y)
    println(s"Test Accuracy: ${test.accuracy}")
    println(s"Test logloss: ${test.logloss}")
    val errors = result.output.filter(sample => sample.probability < 0.005)
    println(s"Results with < 0.5% probability: ${errors.length} / ${testData.X.rows}")
    println(errors.mkString("\n"))
  }

  test("Save and load a network") {
    val target = "target/test-network.json"
    val w1 = DenseMatrix(
      (0.1, 0.3, 0.5),
      (0.2, 0.4, 0.6)
    )
    val w2 = DenseMatrix(
      (0.7, 1.1, 1.5),
      (0.8, 1.2, 1.6),
      (0.9, 1.3, 1.7),
      (1.0, 1.4, 1.8)
    )
    val thetas = SimpleNetwork.Thetas(w1, w2)
    val network = new SimpleNetwork(thetas)
    network.save(target)
    val file = new File(target)
    file.exists() should equal (true)
    val check = SimpleNetwork.load(target)
    check.thetas.w1 should equal (w1)
    check.thetas.w2 should equal (w2)
  }





}
