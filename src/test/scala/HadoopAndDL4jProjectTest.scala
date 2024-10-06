import HadoopAndDL4jProject.{DoubleArrayWritable, TokenizerMapper}
import com.typesafe.config.ConfigFactory
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.mockito.{ArgumentCaptor, ArgumentMatchers, MockitoSugar}
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import scala.jdk.CollectionConverters.{CollectionHasAsScala, IterableHasAsJava}


class HadoopAndDL4jProjectTest extends AnyFlatSpec with MockitoSugar {

  def roundToTwoDecimals(value: Double): Double = {
    BigDecimal(value).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  // Rounding an array of doubles to two decimal places
  def roundArrayToTwoDecimals(arr: Array[Double]): Array[Double] = {
    arr.map(roundToTwoDecimals)
  }

  "trainModel" should "train and save the Word2Vec model" in {
    // Given
    val inputFilePath = "C:\\Users\\taita\\IdeaProjects\\LLMProject1\\src\\test\\resources\\text_sentences.txt" // Small test file
    val outputFilePath = "C:\\Users\\taita\\IdeaProjects\\LLMProject1\\src\\test\\resources\\output_model.txt"

    val config = ConfigFactory.load()

    // When
    HadoopAndDL4jProject.trainModel(config, inputFilePath, outputFilePath)

    // Then
    val modelFile = new File(outputFilePath)
    assert(modelFile.exists(), s"Model file should be created at: $outputFilePath")
  }

  "TokenizerMapper" should "process input file and write key-value pairs" in {
    // Mock the Mapper.Context
    val context = mock[Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context]
    val inputText = new Text("C:\\Users\\taita\\IdeaProjects\\LLMProject1\\src\\test\\resources\\file1_dir.txt") // Mock input

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
    println(fewPairs)
    fewPairs.foreach { case (word, writable) =>
      print(writable.getClass)
      val (frequency, embedding) = writable.get()
      println(s"Word: ${word.toString}, Frequency: $frequency, Embedding: ${embedding.mkString(", ")}")
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
    println(s"File not found: $nonExistentFilePath") // Ensure the message matches the behavior
  }

  "EmbeddingGather" should "combine frequencies and average embeddings" in {
    // Mock context
    val context = mock[Reducer[Text, DoubleArrayWritable, Text, DoubleArrayWritable]#Context]

    // Input values to the reducer
    val inputKey = new Text("word1")
    val inputValues = List(
      new DoubleArrayWritable(2, Array(0.5, 0.3, 0.7)),
      new DoubleArrayWritable(3, Array(0.6, 0.2, 0.4))
    ).asJava  // Convert to java.lang.Iterable

    // Create an instance of the reducer
    val reducer = new HadoopAndDL4jProject.EmbeddingGather

    // Call the reducer's reduce method
    reducer.reduce(inputKey, inputValues, context)

    // Capture the output written to the context
    val keyCaptor = ArgumentCaptor.forClass(classOf[Text])
    val writableCaptor = ArgumentCaptor.forClass(classOf[DoubleArrayWritable])
    verify(context).write(keyCaptor.capture(), writableCaptor.capture())

    // Get the captured values
    val capturedKey = keyCaptor.getValue
    val capturedWritable = writableCaptor.getValue.asInstanceOf[DoubleArrayWritable]
    val (capturedFrequency, capturedEmbedding) = capturedWritable.get()

    // Assertions
    assert(capturedKey.toString == "word1", "The key should be 'word1'")
    assert(capturedFrequency == 5, "The total frequency should be 5")

    // The expected result, rounded to two decimal places
    val expectedEmbedding = Array(0.56, 0.24, 0.52)
    val roundedCapturedEmbedding = roundArrayToTwoDecimals(capturedEmbedding)
    val roundedExpectedEmbedding = roundArrayToTwoDecimals(expectedEmbedding)

    assert(roundedCapturedEmbedding.sameElements(roundedExpectedEmbedding), "The averaged embeddings are incorrect")
  }

  "EmbeddingGather" should "handle a single value by outputting it unchanged" in {
    // Mock context
    val context = mock[Reducer[Text, DoubleArrayWritable, Text, DoubleArrayWritable]#Context]

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
    val writableCaptor = ArgumentCaptor.forClass(classOf[DoubleArrayWritable])
    verify(context).write(keyCaptor.capture(), writableCaptor.capture())

    // Get the captured values
    val capturedKey = keyCaptor.getValue
    val capturedWritable = writableCaptor.getValue.asInstanceOf[DoubleArrayWritable]
    val (capturedFrequency, capturedEmbedding) = capturedWritable.get()

    // Assertions
    assert(capturedKey.toString == "word2", "The key should be 'word2'")
    assert(capturedFrequency == 1, "The total frequency should be 1")

    // The expected result, rounded to two decimal places
    val expectedEmbedding = Array(0.8, 0.4, 0.9)
    val roundedCapturedEmbedding = roundArrayToTwoDecimals(capturedEmbedding)
    val roundedExpectedEmbedding = roundArrayToTwoDecimals(expectedEmbedding)

    assert(roundedCapturedEmbedding.sameElements(roundedExpectedEmbedding), "The embeddings should be the same as the input")
  }

}
