package otto

import java.io.File

import breeze.linalg.DenseVector
import com.google.common.base.Charsets
import com.google.common.io.Files
import grizzled.slf4j.Logging
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization._


case class SimpleNetworkIO(layers: Seq[Int], thetas: Seq[Double]) {

  def toFile(fileName: String): Unit = {
    implicit val format = Serialization.formats(NoTypeHints)
    val json = write(this)
    val file = new File(fileName)
    val resource = Files.asByteSink(file)
    resource.write(json.getBytes("UTF8"))
  }

}

object SimpleNetworkIO extends Logging {

  def save(file: String, network: SimpleNetwork): Unit = {
    val io = new SimpleNetworkIO(network.layers, network.thetas.flatten.toScalaVector())
    io.toFile(file)
  }

  def load(fileName: String) = {
    implicit val format = Serialization.formats(NoTypeHints)
    val json = Files.readFirstLine(new File(fileName), Charsets.UTF_8)
    val io = read[SimpleNetworkIO](json)
    val network = SimpleNetwork(io.layers(0), io.layers(1), io.layers(2))
    logger.debug(s"Network: $network")
    logger.debug(s"Thetas:\n ${network.thetas}")
    val vector = DenseVector(io.thetas:_*)
    logger.debug(s"Loaded thetas: \n$vector")
    network.update(vector)
  }

}
