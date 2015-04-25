package otto

import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import grizzled.slf4j.Logging

class SimpleNetwork(val thetas: SimpleNetwork.Thetas) extends Logging {

  val layers = Seq(thetas.w1.cols, thetas.w1.rows, thetas.w2.rows)

  def train(X: Features, y: Labels, lambda: Double, maxIterations: Int): SimpleNetwork = {
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(vector: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val updated = thetas.update(vector)
        updated.costFunction(X, y, lambda)
      }
    }
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = maxIterations)
    val result = lbfgs.minimize(f, thetas.flatten)
    new SimpleNetwork(thetas.update(result))
  }

  def predict(X: Features): Labels = {
    thetas.activations(X).a3
  }

  def test(ids: Ids, X: Features, y: Labels): SimpleNetwork.TestResults = {
    val predictions = predict(X)
    val list = Seq.tabulate(predictions.rows){ i =>
      val id = ids(i)
      val features = X(i,::).inner
      val label = y(i,::).inner
      val prediction = predictions(i, ::).inner
      val probability = sum(label :* prediction)
      SimpleNetwork.TestOutput(id, features, label, prediction, probability)
    }
    val probabilities = for(sample <- list) yield {
      if (sample.probability >= (1 - 1e-15))
        1 - 1e-15
      else if (sample.probability < 1e-15)
        1e-15
      else sample.probability
    }
    import breeze.stats._
    val accuracy: Double = mean(DenseVector(probabilities:_*))
    val logloss: Double = probabilities.map(log(_)).sum * (-1D/probabilities.length)
    SimpleNetwork.TestResults(accuracy, logloss, list)
  }

}

object SimpleNetwork extends Logging {
  
  case class TestOutput(id: Double, features: DenseVector[Double], label: DenseVector[Double], prediction: DenseVector[Double], probability: Double)
  case class TestResults(accuracy: Double, logloss: Double, output: Seq[TestOutput])

  case class Activations(a1: DenseMatrix[Double], a2: DenseMatrix[Double], a3: DenseMatrix[Double])

  case class Thetas(w1: DenseMatrix[Double], w2: DenseMatrix[Double]) {
    def flatten = {
      DenseVector.vertcat(Seq(w1.toDenseVector, w2.toDenseVector):_*)
    }
    def update(vector: DenseVector[Double]) = {
      val nw1 = new DenseMatrix(w1.rows, w1.cols, vector.data, 0)
      val nw2 = new DenseMatrix(w2.rows, w2.cols, vector.data, w1.rows * w1.cols)
      new Thetas(nw1, nw2)
    }
    def activations(X: Features) = {
      val a1 = bias(X)
      val a2 = bias(activate(a1, w1))
      val a3 = activate(a2, w2)
      Activations(a1, a2, a3)
    }
    private def bias(input: DenseMatrix[Double]) = {
      DenseMatrix.horzcat(DenseMatrix.ones[Double](input.rows, 1), input)
    }
    private def activate(input: DenseMatrix[Double], weights: DenseMatrix[Double]): DenseMatrix[Double] = {
      sigmoid(input * weights.t)
    }

    def costFunction(X: Features, y: Labels, lambda: Double): (Double, DenseVector[Double]) = {
      val a = activations(X)
      val cost = mse(a.a3, y, X.rows)
      val reg = regularization(lambda, X.rows)
      val grads = gradients(a, y, lambda)
      (cost + reg, grads)
    }

    private def mse(h: DenseMatrix[Double], actual: DenseMatrix[Double], m: Double): Double = {
      sum( (-actual :* log(h)) :- ((1.0 :- actual) :* log(1.0 :- h)) ) / m
    }

    private def regularization(lambda: Double, m: Double): Double = {
      val w1reg: Double = sum(w1(::,1 to w1.cols - 1):^2D)
      val w2reg: Double = sum(w2(::,1 to w2.cols - 1):^2D)
      (w1reg + w2reg) * lambda / (2 * m)
    }

    private def gradients(activations: Activations, y: Labels, lambda: Double): DenseVector[Double] = {
      val m: Double = activations.a1.rows
      val d3: DenseMatrix[Double] = activations.a3 - y
      val d2: DenseMatrix[Double] = sigmoidGradient(unbiased(activations.a2)) :* (d3 * unbiased(w2))
      val w2Gradient: DenseMatrix[Double] = (1/m) :* (d3.t * activations.a2)
      val w1Gradient: DenseMatrix[Double] = (1/m) :* (d2.t * activations.a1)
      val w1Reg = (lambda/m) :* w1
      val w2Reg = (lambda/m) :* w2
      val w1GradientReg = w1Gradient + w1Reg
      w1GradientReg(::,0) := w1Gradient(::,0)
      val w2GradientReg = w2Gradient + w2Reg
      w2GradientReg(::,0) := w2Gradient(::,0)
      val list = Seq(w1GradientReg.toDenseVector, w2GradientReg.toDenseVector)
      DenseVector.vertcat(list:_*)
    }

    private def sigmoidGradient(activation: DenseMatrix[Double]) = {
      activation :* (1.0:-activation)
    }

    private def unbiased(activation: DenseMatrix[Double]) = {
      activation(::, 1 to activation.cols - 1)
    }

    override def toString = {
      s"w1:\n$w1\nw2:\n$w2"
    }

  }

  def apply(input: Int, hidden: Int, output: Int) = {
    val w1 = initializeTheta(input, hidden)
    val w2 = initializeTheta(hidden, output)
    new SimpleNetwork(Thetas(w1, w2))
  }

  private def initializeTheta(input: Int, output: Int): DenseMatrix[Double] = {
    val epsilon = 0.12
    val random: DenseMatrix[Double] = DenseMatrix.rand[Double](output, input + 1) :* (2 * epsilon)
    random - epsilon
  }
  
  

}


