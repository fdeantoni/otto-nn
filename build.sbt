name := "otto-nn"

version := "1.0"

scalaVersion := "2.11.5"

testOptions in Test += Tests.Argument("-oDSF")

fork in run := true

javaOptions in run ++= Seq("-Xms256M", "-Xmx1G")

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.clapper" %% "grizzled-slf4j" % "1.0.2",
  "org.scala-lang" % "scala-reflect" % "2.11.5",
  "com.google.guava" % "guava" % "18.0",
  "org.json4s" %% "json4s-jackson" % "3.2.11",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "org.scalacheck" %% "scalacheck" % "1.12.2" % "test",
  "ch.qos.logback" % "logback-classic" % "1.0.7" % "runtime"
)
