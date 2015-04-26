package otto

import java.io.File

import com.google.common.io.{FileWriteMode, Files}

import scala.language.reflectiveCalls

class DataWriter(results: SimpleNetwork.TestResults) {

  val header = "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n"

  def save(fileName: String): Unit = {
    val file = new File(fileName)
    if(file.exists()) file.delete()
    val resource = Files.asByteSink(file, FileWriteMode.APPEND)
    resource.write(header.getBytes("UTF8"))
    for(output <- results.output) {
      val id = output.id.toInt
      val prediction = output.prediction.toScalaVector().mkString(",")
      resource.write(s"$id,$prediction\n".getBytes("UTF8"))
    }
  }

}
