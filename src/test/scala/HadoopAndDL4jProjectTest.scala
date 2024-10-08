import HadoopAndDL4jProject.{DoubleArrayWritable, EmbeddingGather, TokenizerMapper, getWeightedAverageEmbeddingWord}
import com.typesafe.config.ConfigFactory
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.mockito.{ArgumentCaptor, ArgumentMatchers, MockitoSugar}
import org.scalatest.flatspec.AnyFlatSpec
import org.slf4j.LoggerFactory

import java.io.File
import scala.jdk.CollectionConverters.{CollectionHasAsScala, IterableHasAsJava}


class HadoopAndDL4jProjectTest extends AnyFlatSpec with MockitoSugar {
  private val log = LoggerFactory.getLogger(HadoopAndDL4jProject.getClass)

  def roundToTwoDecimals(value: Double): Double = {
    BigDecimal(value).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  // Rounding an array of doubles to two decimal places
  def roundArrayToTwoDecimals(arr: Array[Double]): Array[Double] = {
    arr.map(roundToTwoDecimals)
  }

  "TokenizerMapper" should "process input file and write key-value pairs" in {
    // Mock the Mapper.Context
    val context = mock[Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context]
    val inputText = new Text("src/test/resources/file1_dir.txt") // Mock input

    // When
    val mapper = new TokenizerMapper
    mapper.map(new LongWritable(1),
      inputText,
      context)

    // Capture the output from the context write
    val wordCaptor = ArgumentCaptor.forClass(classOf[Text])
    val writableCaptor = ArgumentCaptor.forClass(classOf[DoubleArrayWritable])

    // Verify and capture the key-value pairs sent to the context
    verify(context, atLeastOnce).write(wordCaptor.capture(), writableCaptor.capture())

    // Print the first few captured words and their corresponding embeddings and frequencies
    val capturedWords = wordCaptor.getAllValues
    val capturedEmbeddings = writableCaptor.getAllValues.asScala.map(_.asInstanceOf[DoubleArrayWritable])

    // Print only the first few values (e.g., the first 3)
    val fewPairs = capturedWords.asScala.zip(capturedEmbeddings).take(3)
    fewPairs.foreach { case (word, writable) =>
      val (frequency, embedding) = writable.get()
    }

    // Assert that words were written to context
    assert(capturedWords.size() > 0, "Mapper should have written words to context")
  }


  "TokenizerMapper" should "handle file not found error gracefully" in {
    // Mock the Mapper.Context
    val context = mock[Mapper[LongWritable, Text, Text, HadoopAndDL4jProject.DoubleArrayWritable]#Context]

    // Example non-existent file path
    val nonExistentFilePath = "nonexistent_file.txt"
    val inputText = new Text(nonExistentFilePath)

    // Create an instance of the mapper
    val mapper = new HadoopAndDL4jProject.TokenizerMapper

    // Call the mapper's map method
    mapper.map(new LongWritable(1), inputText, context)

    // Verify that no output was written to the context (since the file was not found)
    verify(context, never).write(ArgumentMatchers.any(classOf[Text]), ArgumentMatchers.any(classOf[HadoopAndDL4jProject.DoubleArrayWritable]))

    // Optionally, check the output logs if necessary
    // If you have a log capturing system, you can assert the log message or printed output
    log.info(s"File not found: $nonExistentFilePath") // Ensure the message matches the behavior
  }

  "EmbeddingGather" should "reducer should write to context" in {
    val context = mock[Reducer[Text, DoubleArrayWritable, Text, Text]#Context]

    // Input values to the reducer
    val inputKey = new Text("word1")
    val inputValues = List(
      new DoubleArrayWritable(2, Array(0.5, 0.3, 0.7)),
      new DoubleArrayWritable(3, Array(0.6, 0.2, 0.4))
    ).asJava

    // Create an instance of the reducer
    val reducer = new EmbeddingGather

    // Call the reducer's reduce method
    reducer.reduce(inputKey, inputValues, context)

    // Capture the output written to the context
    val keyCaptor = ArgumentCaptor.forClass(classOf[Text])
    val valueCaptor = ArgumentCaptor.forClass(classOf[Text])
    verify(context).write(keyCaptor.capture(), valueCaptor.capture())
  }

  "getWeightedAverageEmbeddingWord" should "calculate correct embedding" in {
    val inputValues = List(
      new DoubleArrayWritable(2, Array(0.5, 0.3, 0.7)),
      new DoubleArrayWritable(3, Array(0.6, 0.2, 0.4))
    ).asJava
    val expectedTotalCounts: Int = 5

    val (capturedTotalCounts, capturedEmbedding) = getWeightedAverageEmbeddingWord(inputValues)

    // The expected result, rounded to two decimal places
    val expectedEmbedding = Array(0.56, 0.24, 0.52)
    val roundedCapturedEmbedding = roundArrayToTwoDecimals(capturedEmbedding)
    val roundedExpectedEmbedding = roundArrayToTwoDecimals(expectedEmbedding)

    assert(expectedTotalCounts==capturedTotalCounts)
    assert(roundedCapturedEmbedding.sameElements(roundedExpectedEmbedding), "The averaged embeddings are incorrect")
  }

  "EmbeddingGather" should "complete reducer, handle a single value by outputting it unchanged" in {
    // Mock context
    val context = mock[Reducer[Text, DoubleArrayWritable, Text, Text]#Context]

    // Input values to the reducer (only one value)
    val inputKey = new Text("word2")
    val inputValues = List(
      new DoubleArrayWritable(1, Array(0.8, 0.4, 0.9))
    ).asJava  // Convert to java.lang.Iterable

    // Create an instance of the reducer
    val reducer = new HadoopAndDL4jProject.EmbeddingGather

    // Call the reducer's reduce method
    reducer.reduce(inputKey, inputValues, context)

    // Capture the output written to the context
    val keyCaptor = ArgumentCaptor.forClass(classOf[Text])
    val valueCaptor = ArgumentCaptor.forClass(classOf[Text])
    verify(context).write(keyCaptor.capture(), valueCaptor.capture())

    // test key captured
    val capturedKey = keyCaptor.getValue
    assert(capturedKey.toString == "word2", "The key should be 'word2'")

    // test value
    val (capturedFrequency, capturedEmbedding) = getWeightedAverageEmbeddingWord(inputValues)

    // The expected result, rounded to two decimal places
    val expectedEmbedding = Array(0.8, 0.4, 0.9)
    val roundedCapturedEmbedding = roundArrayToTwoDecimals(capturedEmbedding)
    val roundedExpectedEmbedding = roundArrayToTwoDecimals(expectedEmbedding)

    assert(roundedCapturedEmbedding.sameElements(roundedExpectedEmbedding), "The embeddings should be the same as the input")
  }

}
