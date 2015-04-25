import breeze.linalg.{DenseVector, DenseMatrix}

package object otto {

  type Ids = DenseVector[Double]
  type Features = DenseMatrix[Double]
  type Labels = DenseMatrix[Double]


}
