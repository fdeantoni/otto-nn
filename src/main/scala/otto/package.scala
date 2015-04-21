import breeze.linalg.{DenseVector, DenseMatrix}

package object otto {

  type Features = DenseMatrix[Double]
  type Labels = DenseMatrix[Double]
  type Theta = DenseMatrix[Double]
  type Cost = Double
  type Gradients = DenseVector[Double]

}
