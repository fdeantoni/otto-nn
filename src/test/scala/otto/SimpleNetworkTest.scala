package otto

import breeze.linalg.{*, normalize, DenseMatrix}
import breeze.numerics.cos
import org.scalatest.{Matchers, FunSuite}

class SimpleNetworkTest extends FunSuite with Matchers {

  test("Create one hidden layer network") {
    val network = new SimpleNetwork(Seq(1, 5, 2))
    for (theta <- network.thetas) {
      println("Theta\n" + theta)
    }
    network.thetas(0).rows should equal (5)
    network.thetas(0).cols should equal (2)
    network.thetas(1).rows should equal (2)
    network.thetas(1).cols should equal (6)
  }

  test("Create two hidden layer network") {
    val network = new SimpleNetwork(Seq(2, 2, 4))
    for (theta <- network.thetas) {
      println("Theta\n" + theta)
    }
  }

  test("Cost function check") {
    val network = new SimpleNetwork(Seq(2, 2, 4))
    val thetas = List(
      DenseMatrix(
        (0.1, 0.3, 0.5),
        (0.2, 0.4, 0.6)
      ),
      DenseMatrix(
        (0.7, 1.1, 1.5),
        (0.8, 1.2, 1.6),
        (0.9, 1.3, 1.7),
        (1.0, 1.4, 1.8)
      )
    )
    val X = cos(DenseMatrix( (1, 2), (3, 4), (5, 6) ))
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D) )
    val lambda = 3
    val result = network.costFunction(X, y, lambda, thetas)
    println(result)
    result._1 should be (16.457 +- 1e-3)
  }

  test("Gradient function check without regularization") {
    val network = new SimpleNetwork(Seq(2, 2, 4))
    val thetas = List(
      DenseMatrix(
        (0.1, 0.3, 0.5),
        (0.2, 0.4, 0.6)
      ),
      DenseMatrix(
        (0.7, 1.1, 1.5),
        (0.8, 1.2, 1.6),
        (0.9, 1.3, 1.7),
        (1.0, 1.4, 1.8)
      )
    )
    val X = cos(DenseMatrix( (1, 2), (3, 4), (5, 6) ))
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D) )
    val lambda = 0
    val result = network.costFunction(X, y, lambda, thetas)
    println(result)
    result._1 should be (7.407 +- 1e-3)
    result._2.length should equal (18)
    result._2(0) should be (0.766138 +- 1e-5)
    result._2(17) should be (0.322331 +- 1e-5)
  }

  test("Gradient function check with regularization") {
    val network = new SimpleNetwork(Seq(2, 2, 4))
    val thetas = List(
      DenseMatrix(
        (0.1, 0.3, 0.5),
        (0.2, 0.4, 0.6)
      ),
      DenseMatrix(
        (0.7, 1.1, 1.5),
        (0.8, 1.2, 1.6),
        (0.9, 1.3, 1.7),
        (1.0, 1.4, 1.8)
      )
    )
    val X = cos(DenseMatrix( (1, 2), (3, 4), (5, 6) ))
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D) )
    val lambda = 3
    val result = network.costFunction(X, y, lambda, thetas)
    println(result)
    result._1 should be (16.457 +- 1e-3)
    result._2.length should equal (18)
    result._2(0) should be (0.76614 +- 1e-5)
    result._2(17) should be (2.12233 +- 1e-5)
  }

  test("Gradient function check with regularization with additional samples") {
    val network = new SimpleNetwork(Seq(2, 2, 4))
    val thetas = List(
      DenseMatrix(
        (0.1, 0.3, 0.5),
        (0.2, 0.4, 0.6)
      ),
      DenseMatrix(
        (0.7, 1.1, 1.5),
        (0.8, 1.2, 1.6),
        (0.9, 1.3, 1.7),
        (1.0, 1.4, 1.8)
      )
    )
    val X = DenseMatrix( (1D, 2D), (3D, 4D), (5D, 6D), (0D, 1D), (1D, 2D) )
    val y = DenseMatrix( (0D, 0D, 0D , 1D), (0D, 1D, 0D, 0D), (0D, 0D, 1D, 0D), (1D,0D,0D,0D), (0D,1D,0D,0D) )
    val lambda = 4
    val result = network.costFunction(X, y, lambda, thetas)
    println(result)
    result._1 should be (17.441 +- 1e-3)
    result._2.length should equal (18)
    result._2(0) should be (0.42849 +- 1e-5)
    result._2(17) should be (2.1242 +- 1e-5)
  }

  test("Reshaping and flattening of Thetas") {
    val network = new SimpleNetwork(Seq(2, 2, 4))
    val thetas = List(
      DenseMatrix(
        (0.1, 0.3, 0.5),
        (0.2, 0.4, 0.6)
      ),
      DenseMatrix(
        (0.7, 1.1, 1.5),
        (0.8, 1.2, 1.6),
        (0.9, 1.3, 1.7),
        (1.0, 1.4, 1.8)
      )
    )
    val flattened = network.flattenThetas(thetas)
    println(s"Flattened\n$flattened")
    val reshaped = network.reshapeThetas(flattened, thetas)
    println(s"Reshaped\n$reshaped")
    thetas should equal (reshaped)
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
      normalize(data(::,*))
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
    println(s"Size of X: ${X.rows}x${X.cols}")
    println(s"Size of y: ${y.rows}x${y.cols}")
    val lambda = 0
    val network = new SimpleNetwork(Seq(2, 4, 4))
    network.train(X, y, lambda, 100)
    val (error, logloss) = network.test(X, y)
    println(s"Error: $error")
    println(s"Logloss: $logloss")

  }




  test("A neural network with otto data") {
    val fileName = "src/main/resources/train_clean.csv"
    val loader = new DataLoader(fileName, 0.8, 0.2)
    val data = new PrepareData(loader.trainData)
    val network = new SimpleNetwork(Seq(93, 100, 9))
    network.train(data.X, data.y, 0.5, 300)
    val test = new PrepareData(loader.testData)
    val (error, logloss) = network.test(test.X, test.y)
    println(s"Error: $error")
    println(s"Logloss: $logloss")
  }




}
