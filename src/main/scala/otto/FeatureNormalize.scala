package otto

import breeze.linalg._


object FeatureNormalize {

  /**
   * Sometimes features are not normally distributed. In those cases they may have log-normal distributions instead. In
   * such cases, after log-transformation, the values become normally distributed
   */
  def log10(data: DenseMatrix[Double]) = {
    breeze.numerics.log10(1D:+data)
  }

  /**
   * Useful when the features are normally distributed. This will bring them into a -1 to 1 range.
   */
  def std(data: DenseMatrix[Double]) = {
    import breeze.stats._
    val mv: DenseMatrix[MeanAndVariance] = meanAndVariance(data(::,*))
    val cols = for(i <- 0 to mv.cols - 1) yield {
      val mvi = mv(0,i)
      data(::,i).map( v => (v - mvi.mean) / breeze.numerics.sqrt(mvi.variance) )
    }
    DenseMatrix(cols.map(_.data):_*).t
  }

}
