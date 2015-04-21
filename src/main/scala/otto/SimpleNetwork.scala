package otto

import breeze.linalg._
import breeze.numerics.{log, sigmoid}
import breeze.optimize._
import grizzled.slf4j.Logging

class SimpleNetwork(layers: Seq[Int]) extends Logging {

  val layerIndex = 0 to (layers.length - 1)

  var thetas: List[Theta] = layers.sliding(2).map { layer =>
    initializeTheta(layer(0), layer(1))
  }.toList

  private def initializeTheta(input: Int, output: Int): Theta = {
    val epsilon = 0.12
    val random: DenseMatrix[Double] = DenseMatrix.rand[Double](output, input + 1) :* (2 * epsilon)
    random - epsilon
  }

  def train(X: Features, y: Labels, lambda: Double, maxIterations: Int): Unit = {
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(theta: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val rolled = reshapeThetas(theta, thetas)
        costFunction(X, y, lambda, rolled)
      }
    }
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = maxIterations)
    val unrolledThetas = flattenThetas(thetas)
    val result = lbfgs.minimize(f, unrolledThetas)
    thetas = reshapeThetas(result, thetas)
  }

  def flattenThetas(thetas: List[Theta]): DenseVector[Double] = {
    DenseVector.vertcat(thetas.map(_.toDenseVector).toSeq:_*)
  }

  def reshapeThetas(vector: DenseVector[Double], originalThetas: List[Theta]): List[Theta] = {
    List.tabulate(thetas.length){ i =>
      val offset = if(i > 0) originalThetas(i-1).rows * originalThetas(i-1).cols else 0
      new DenseMatrix(originalThetas(i).rows, originalThetas(i).cols, vector.data, offset)
    }
  }


  def costFunction(X: Features, y: Labels, lambda: Double, thetas: List[Theta]): (Cost, Gradients) = {
    assert(X.rows == y.rows)
    var activations: Seq[DenseMatrix[Double]] = Seq( DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1), X) )
    var zs: Seq[DenseMatrix[Double]] = Seq.empty[DenseMatrix[Double]]
    val steps = 0 to (thetas.length -1)
    for(i <- steps) {
      val z: DenseMatrix[Double] = activations.last * thetas(i).t
      zs :+= z
      val a: DenseMatrix[Double] = sigmoid(z)
      activations :+= {
        if(i == steps.last) a else DenseMatrix.horzcat(DenseMatrix.ones[Double](z.rows, 1), a) // add bias unit
      }
    }
    val cost = error(activations.last, y, X.rows)
    val reg = regularization(lambda, X.rows, thetas)
    val g = gradients(zs, activations, y, X.rows, thetas, lambda)
    (cost + reg, g)
  }

  def error(h: DenseMatrix[Double], actual: DenseMatrix[Double], m: Double): Cost = {
    sum( (-actual :* log(h)) :- ((1.0 :- actual) :* log(1.0 :- h)) ) / m
  }

  def regularization(lambda: Double, m: Double, thetas: List[Theta]): Double = {
    val thetasSquared: Seq[Double] = thetas.map { theta =>
      sum( theta(::,1 to theta.cols - 1):^2D )
    }
    thetasSquared.sum * lambda / (2 * m)
  }

  def gradients(zs: Seq[DenseMatrix[Double]], activations: Seq[DenseMatrix[Double]], y: Labels, m: Double, thetas: List[Theta], lambda: Double): Gradients = {
    var deltas: Seq[DenseMatrix[Double]] = List(activations.last - y)
    val idx = (1 to (activations.length - 2)).reverse  // back-propagate excluding input and output activations
    assert(idx.length+1 == thetas.length)
    for(i <- idx) {
      val theta = thetas(i)(::,1 to thetas(i).cols-1) // remove bias unit
      val z = zs(i-1)
      val r: DenseMatrix[Double] = deltas.last * theta
      val s: DenseMatrix[Double] = sigmoid(z) :* (1.0:- sigmoid(z))
      deltas :+= r :* s
    }
    val gradients: Seq[DenseMatrix[Double]] = deltas.reverse.zip(activations.dropRight(1)).map { item =>
      val delta: DenseMatrix[Double] = item._1
      val activation: DenseMatrix[Double] = item._2
      val result: DenseMatrix[Double] = delta.t * activation
      result :/ m
    }
    val gradientsRegularized: Seq[DenseMatrix[Double]] = gradients.zip(thetas).map { item =>
      val unregularized: DenseMatrix[Double] = item._1
      val theta: DenseMatrix[Double] = item._2
      val reg: DenseMatrix[Double] = theta :* (lambda/m)
      val regularized = unregularized + reg
      regularized(::,0) := unregularized(::,0)
      regularized
    }
    DenseVector.vertcat(gradientsRegularized.map(_.toDenseVector):_*)
  }




}


