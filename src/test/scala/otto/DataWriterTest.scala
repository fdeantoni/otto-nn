package otto

import java.io.File

import breeze.linalg.DenseVector
import com.google.common.base.Charsets
import com.google.common.io.Files
import org.scalatest.{Matchers, FunSuite}

import scala.util.Random

class DataWriterTest extends FunSuite with Matchers {

  test("Write test output to file") {
    val file = "target/test-output.csv"
    val output = for(i <- 1 to 10) yield {
      val features = DenseVector.rand[Double](4)
      val label = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
      val prediction = DenseVector.rand[Double](9)
      val prob = Random.nextDouble()
      SimpleNetwork.TestOutput(id = i.toDouble, features = features, label = label, prediction = prediction, probability = prob)
    }
    val results = SimpleNetwork.TestResults(0.1, 0.1, output)
    val writer = new DataWriter(results)
    writer.save(file)
    val check = new File(file)
    check.exists() should equal (true)
    import scala.collection.JavaConverters._
    val lines = Files.readLines(new File(file), Charsets.UTF_8).asScala.toSeq
    println(lines)
    lines.length should equal (11)
  }

}
