import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, LineSentenceIterator, SentenceIterator, SentencePreProcessor}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text, Writable}
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.Mapper
import org.apache.hadoop.mapreduce.Reducer
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.deeplearning4j.models.embeddings
import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.typesafe.config.{Config, ConfigFactory}


import java.io.{DataInput, DataOutput, File, IOException}
import java.util.StringTokenizer
import scala.jdk.CollectionConverters._  // Scala 2.13 uses CollectionConverters

object HadoopAndDL4jProject {

  private val log = LoggerFactory.getLogger(HadoopAndDL4jProject.getClass)

  def trainModel(config: Config, inputFilePath: String, outputFilePath: String): Unit = {
    log.info("Load data...")
    val iter: SentenceIterator = new LineSentenceIterator(new File(inputFilePath))
    iter.setPreProcessor((sentence: String) => sentence.toLowerCase())

    // split on the white spaces in the line to get words
    val tokenizer: TokenizerFactory = new DefaultTokenizerFactory()
    tokenizer.setTokenPreProcessor(new CommonPreprocessor())

    // train the model
    log.info("Building model...")
    val vec: Word2Vec = new Word2Vec.Builder()
      .minWordFrequency(config.getInt("word2vec.minWordFrequency"))
      .layerSize(config.getInt("word2vec.layerSize"))
      .seed(config.getLong("word2vec.seed"))
      .windowSize(config.getInt("word2vec.windowSize"))
      .iterate(iter)
      .tokenizerFactory(tokenizer)
      .build()

    log.info("Fitting Word2Vec.model...")
    vec.fit()
    // write word vectors
    WordVectorSerializer.writeWord2VecModel(vec, outputFilePath)
    log.info("Stored model")
  }

  def getClosestWords(modelVec: Word2Vec, targetWord: String): Unit = {
    log.info("Closest Words:")
    val lst = modelVec.wordsNearest(targetWord, 10)
    print(lst)
    log.info("Complete get closest words")
  }

  // class to create Writable for an array of double
  class DoubleArrayWritable(var frequency: Int, var array: Array[Double]) extends Writable {
    def this() = this(0, Array[Double]())

    // serialize
    override def write(out: DataOutput): Unit = {
      out.writeInt(frequency)
      out.writeInt(array.length)
      array.foreach(out.writeDouble)
    }

    // deserialize
    override def readFields(in: DataInput): Unit = {
      frequency = in.readInt()
      val length = in.readInt()
      array = Array.fill(length)(in.readDouble())
    }

    def get(): (Int, Array[Double]) = {
      (frequency, array)
    }
  }

  // the mapper
  class TokenizerMapper extends Mapper[LongWritable, Text, Text, DoubleArrayWritable] {

    @throws(classOf[IOException])
    @throws(classOf[InterruptedException])
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context): Unit = {
      // Assume the directory of the file is passed through the mapper context
      val inputFilePath = value.toString

      // Logging for debugging
      log.info(s"Processing file path: $inputFilePath")
      val config = context.getConfiguration

      try {
        // Access the file from HDFS in the HDFS
        // Commented-out code for potential HDFS usage
        //    val conf = context.getConfiguration
        //    val fs = FileSystem.get(conf)
        //    val path = new Path(inputFilePath)
        //    val inputStream = fs.open(path)

        // This is for testing from a test environment (non-HDFS path)
        // Step 1: Check if the first file exists and throw an exception if not
        val file = new File(inputFilePath)
        if (!file.exists()) {
          throw new IOException(s"File not found: $inputFilePath")
        }

        // Step 2: Read the content of the first file (expecting it to point to a second file path)
        val content = scala.io.Source.fromFile(file).getLines().mkString("\n")
        println(s"File content (path to second file): $content")

        // Step 3: Check if the second file exists and throw an exception if not
        val file1 = new File(content)
        if (!file1.exists()) {
          throw new IOException(s"Second file not found: $content")
        }

        // Step 4: Initialize SentenceIterator to read the second file
        val iter: SentenceIterator = new BasicLineIterator(file1)
        iter.setPreProcessor((sentence: String) => sentence.toLowerCase())

        // Step 5: Check if the iterator has sentences and throw an exception if the file is empty
        if (!iter.hasNext) {
          throw new IOException(s"The file $content is empty.")
        }

        println("Start training the Word2Vec model...")

        // Step 6: Initialize the tokenizer and train the model
        val tokenizer: TokenizerFactory = new DefaultTokenizerFactory()
        tokenizer.setTokenPreProcessor(new CommonPreprocessor())

        log.info("Building Word2Vec model...")
        val vec: Word2Vec = new Word2Vec.Builder()
          .minWordFrequency(config.getInt("word2vec.minWordFrequency",5))
          .layerSize(config.getInt("word2vec.layerSize",100))
          .seed(config.getLong("word2vec.seed", 42L))
          .windowSize(config.getInt("word2vec.windowSize", 5))
          .iterate(iter)
          .tokenizerFactory(tokenizer)
          .build()

        log.info("Fitting the Word2Vec model...")
        vec.fit()

        // Step 7: Pass key-value pair of words and embeddings to the reducer
        val allWords = vec.vocab().words().asScala  // Use asScala from CollectionConverters
        println(s"Vocabulary words: $allWords")

        allWords.foreach(word => {
          context.write(new Text(word), new DoubleArrayWritable(vec.vocab.wordFrequency(word), vec.getWordVector(word)))
        })

      } catch {
        case e: IOException =>
          log.info("Path not found")
      }
    }
  }

  class EmbeddingGather extends Reducer[Text, DoubleArrayWritable, Text, DoubleArrayWritable] {
    @throws(classOf[IOException])
    @throws(classOf[InterruptedException])
    override def reduce(key: Text, values: java.lang.Iterable[DoubleArrayWritable], context: Reducer[Text, DoubleArrayWritable, Text, DoubleArrayWritable]#Context): Unit = {
      // Use reduce to combine the frequencies and embeddings
      // for frequencies
      log.info("Running reduce")
      val scalaValues = values.asScala  // Use asScala from CollectionConverters
      val totalCounts = scalaValues.map(_.frequency).sum
      // for embedding
      val freqEmbeddings = scalaValues.map((ele) => {
        ele.array.map(_ * ele.frequency)
      })
      //// sum the embedding
      val sumEmbeddings = freqEmbeddings.reduce((arr1, arr2) => arr1.zip(arr2).map{case (x: Double, y: Double) => x + y})
      //// average the embedding
      val finalEmbeddings = sumEmbeddings.map(_/totalCounts)

      // Write the result to context
      context.write(key, new DoubleArrayWritable(totalCounts, finalEmbeddings))
    }
  }


  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.load()
    val hadoopConfig = new Configuration()
    val job = Job.getInstance(hadoopConfig, "embedding")
    job.setJarByClass(HadoopAndDL4jProject.getClass)
    // mapper and reducer classes
    job.setMapperClass(classOf[TokenizerMapper])
    job.setReducerClass(classOf[EmbeddingGather])
    // set the output key and value types for the mapper
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[DoubleArrayWritable])
    // set the output key and value types for the reducer
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[DoubleArrayWritable])
    // set th input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, DoubleArrayWritable]])
    // set input and output path
    FileInputFormat.addInputPath(job, new Path(args(1)))
    FileOutputFormat.setOutputPath(job, new Path(args(2)))
    val result = if (job.waitForCompletion(true)) 0 else 1
    sys.exit(result)
  }
}
