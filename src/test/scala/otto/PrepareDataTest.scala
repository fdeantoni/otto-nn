package otto

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

  test("Prune some samples") {
    val fileName = "src/test/resources/train_sample_small.csv"
    val loader = new DataLoader(fileName, 0.8, 0.2)
    val ids = loader.trainData(0 to 4, 0).toScalaVector()
    println("ids for pruning: " + ids)
    val result = new PrepareData(loader.trainData, ids).ids.toScalaVector()
    println("result: " + result)
    for(id <- ids) {
      result should not contain id
    }
  }

}
