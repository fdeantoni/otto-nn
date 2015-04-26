package otto

import java.io.File

import breeze.linalg.DenseVector
import com.google.common.base.Charsets
import com.google.common.io.Files
import org.scalatest.{Matchers, FunSuite}
import otto.ActualData.Prediction

class DataWriterTest extends FunSuite with Matchers {

  test("Write test output to file") {
    val file = "target/test-output.csv"
    val output = for(i <- 1 to 10) yield {
      val probability = DenseVector.rand[Double](9)
      new Prediction(i.toInt, probability)
    }
    val writer = new DataWriter(output)
    writer.save(file)
    val check = new File(file)
    check.exists() should equal (true)
    import scala.collection.JavaConverters._
    val lines = Files.readLines(new File(file), Charsets.UTF_8).asScala.toSeq
    println(lines)
    lines.length should equal (11)
  }

}
