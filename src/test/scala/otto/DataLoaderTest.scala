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

  test("Prune some samples") {
    val fileName = "src/test/resources/train_sample_small.csv"
    val ids = Seq(1D, 2D, 3D, 4D, 5D)
    println("ids for pruning: " + ids)
    val loader = new DataLoader(fileName, 0.8, 0.2, ids)
    println("training data loaded: \n" + loader.trainData)
    val data = loader.trainData(::,0).toScalaVector()
    for(id <- ids) {
      data should not contain id
    }
  }

}
