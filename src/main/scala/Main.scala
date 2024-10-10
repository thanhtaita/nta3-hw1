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
import com.typesafe.config.{Config, ConfigException, ConfigFactory}

import java.io.{BufferedInputStream, BufferedReader, DataInput, DataOutput, File, IOException, InputStreamReader}
import java.util.StringTokenizer
import scala.jdk.CollectionConverters._  // Scala 2.13 uses CollectionConverters

object HadoopAndDL4jProject {

  private val log = LoggerFactory.getLogger(HadoopAndDL4jProject.getClass)

  def isHDFSPath(filePath: String): Boolean = {
    filePath.startsWith("hdfs://");
  }


  // get suitable SentenceIterator depends on system to work with Word2Vec from deeplearning4j
  def getSentenceIterator (filePath: String): SentenceIterator = {
    val config = new Configuration()
    val fs = FileSystem.get(config)
    try {
      if (isHDFSPath(fs.getUri.toString)) {
        val inputStream = fs.open(new Path(fs.getUri + filePath))
        new BasicLineIterator(new BufferedInputStream(inputStream))
      } else {
        val file = new File(filePath)
        if (!file.exists()) {
          log.error(s"Second file not found: $filePath")
          throw new IOException(s"Second file not found: $filePath")
        }
        new LineSentenceIterator(file)
      }
    } catch {
      case e: IOException => throw new IOException(s"Can't open file: ${filePath}", e)
    }
  }

  def trainModel(inputFilePath: String): Word2Vec = {
    try {
      val config: Config = ConfigFactory.load()
      log.info("Load data...")

      log.info("Start training the Word2Vec model...")
      val iter: SentenceIterator = getSentenceIterator(inputFilePath)
      iter.setPreProcessor((sentence: String) => sentence.toLowerCase())

      if (!iter.hasNext) {
        log.error(s"The file is empty.")
        throw new IOException(s"The file is empty.")
      }

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
      vec
    } catch {case e: ConfigException.Missing =>
      throw new RuntimeException(s"Configuration key is missing: ${e.getMessage}", e)

    case e: ConfigException.Parse =>
      throw new RuntimeException(s"Configuration file is malformed: ${e.getMessage}", e)

    case e: ConfigException.IO =>
      throw new RuntimeException(s"Could not load configuration file: ${e.getMessage}", e)

    case e: IOException => throw new IOException(s"Can't start training model: ${e.getMessage}", e)

    case e: Exception =>
      throw new RuntimeException(s"Unexpected error loading configuration: ${e.getMessage}", e)
    }
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

    override def toString: String = {
      s"${frequency},${array.mkString(",")}"
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

      try {
        val vec: Word2Vec = trainModel(inputFilePath)

        // Step 7: Pass key-value pair of words and embeddings to the reducer
        val allWords = vec.vocab().words().asScala  // Use asScala from CollectionConverters

        // for testing
//        context.write(new Text("word"), new DoubleArrayWritable(2, Array(1.1, 1.2, 1.3)))

        allWords.foreach(word => {
          context.write(new Text(word), new DoubleArrayWritable(vec.vocab.wordFrequency(word), vec.getWordVector(word)))
        })
      } catch {
        case e: RuntimeException => log.error(s"${e.getMessage}")
        case e: IOException => log.error(s"${e.getMessage}")
        case e: Exception => log.error(s"${e.getMessage}")
      }
    }
  }

  def getWeightedAverageEmbeddingWord(values: java.lang.Iterable[DoubleArrayWritable]): (Int, Array[Double]) = {
    val scalaValues = values.asScala
    // get total appearances of the word
    val totalCounts = scalaValues.map(_.frequency).sum
    // for embedding
    val freqEmbeddings = scalaValues.map((ele) => {
      ele.array.map(_ * ele.frequency)
    })
    //// sum the embedding
    val sumEmbeddings = freqEmbeddings.reduce((arr1, arr2) => arr1.zip(arr2).map{case (x: Double, y: Double) => x + y})
    //// average the embedding
    val finalEmbeddings = sumEmbeddings.map(_/totalCounts)
    (totalCounts, finalEmbeddings)
  }
  class EmbeddingGather extends Reducer[Text, DoubleArrayWritable, Text, Text] {
    @throws(classOf[IOException])
    @throws(classOf[InterruptedException])
    override def reduce(key: Text, values: java.lang.Iterable[DoubleArrayWritable], context: Reducer[Text, DoubleArrayWritable, Text, Text]#Context): Unit = {
      // Use reduce to combine the frequencies and embeddings
      log.info("Running reduce")
      val (frequency, embeddings) = getWeightedAverageEmbeddingWord(values)
      // Write the result to context (as csv format)
      context.write(key, new Text ((new DoubleArrayWritable(frequency, embeddings)).toString))
    }
  }


  def main(args: Array[String]): Unit = {
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
    job.setOutputValueClass(classOf[Text])
    // set th input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]])
    // set input and output path
    FileInputFormat.addInputPath(job, new Path(args(1)))
    FileOutputFormat.setOutputPath(job, new Path(args(2)))
    val result = if (job.waitForCompletion(true)) 0 else 1
    sys.exit(result)
  }
}
