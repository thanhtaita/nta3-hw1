
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.15"

lazy val root = (project in file("."))
  .settings(
    name := "LLMProject1",
    libraryDependencies += "org.scalameta" %% "munit" % "1.0.0" % Test,
    libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.4.0",
    libraryDependencies += "org.apache.hadoop" % "hadoop-hdfs" % "3.4.0",
    libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.4.0",
    libraryDependencies += "org.apache.hadoop" % "hadoop-yarn-client" % "3.4.0",
    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7",
    libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",
    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta7",
    libraryDependencies += "org.slf4j" % "slf4j-api" % "2.0.12",               // SLF4J 2.x for logging
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.3.0",    // Logback for SLF4J 2.x compatibility
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test, // ScalaTest for testing
    libraryDependencies += "org.mockito" %% "mockito-scala" % "1.17.37" % Test, // Mockito for mocking
    libraryDependencies += "org.mockito" % "mockito-core" % "3.12.4" % Test,
    libraryDependencies += "org.scalatestplus" %% "mockito-3-4" % "3.2.10.0" % Test,
    libraryDependencies += "com.typesafe" % "config" % "1.4.3",
    // Assembly merge strategy
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) =>
          xs match {
              case "MANIFEST.MF" :: Nil     => MergeStrategy.discard
              case "services" :: _          => MergeStrategy.concat
              case _                        => MergeStrategy.discard
          }
      case "reference.conf"            => MergeStrategy.concat
      case x if x.endsWith(".proto")   => MergeStrategy.rename
      case x if x.contains("hadoop")   => MergeStrategy.first
      case _                           => MergeStrategy.first
    }

  )



