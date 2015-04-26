package otto

import org.scalatest.{Matchers, FunSuite}

class DataLoaderTest extends FunSuite with Matchers {

  test("Load the data without cross-validation set") {
    val fileName = "src/test/resources/train_sample_small.csv"
    val loader = new DataLoader(fileName, 0.5, 0.5)
    loader.data.rows should equal (20)
    loader.trainSize should equal (10)
    loader.testSize should equal (10)
    loader.cvSize should equal (0)
    loader.trainData.rows should equal (10)
    loader.testData.rows should equal (10)
    loader.cvData.rows should equal (0)
  }

  test("Load the data with a cross-validation set") {
    val fileName = "src/test/resources/train_sample_small.csv"
    val loader = new DataLoader(fileName, 0.5, 0.25)
    loader.data.rows should equal (20)
    loader.trainSize should equal (10)
    loader.testSize should equal (5)
    loader.cvSize should equal (5)
    loader.trainData.rows should equal (10)
    loader.testData.rows should equal (5)
    loader.cvData.rows should equal (5)
  }

}
