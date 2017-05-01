import org.apache.spark.SparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.recommendation._

object ALSEval {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: ALSEval <input file> <output file>");
      System.exit(1);
    }

    val spark = SparkSession.builder().master("local[2]").appName("ALSEval").config("spark.sql.warehouse.dir", "/Users/gauravkhandave/Documents/BigDataWorkspace/ALSRecomEval/warehouse").getOrCreate()
    import spark.implicits._

    val userArtistSchema =
      StructType(
        StructField("userId", IntegerType, false) ::
          StructField("artistId", IntegerType, true) ::
          StructField("playCount", DoubleType, true) :: Nil)

    val artistAliasSchema =
      StructType(
        StructField("badId", IntegerType, false) ::
          StructField("goodId", IntegerType, true) :: Nil)

    val userArtistRddRaw = spark.read.text("data/user_artist_data.txt").rdd.map(line => (line.toString().split(" ")(0).replaceAll("\t", " "), line.toString().split(" ")(1).replaceAll("\t", " "), line.toString().split(" ")(2).replaceAll("\t", " ")))

    val artistAliasRddRaw = spark.read.text("data/artist_alias.txt").rdd.map(line => line.toString().replaceAll("\t", " ")).map(line => (line.split(" ")(0), line.split(" ")(1)))

    val userArtistRdd = userArtistRddRaw.map(line => (line._1.replace("[", ""), line._2, line._3.replace("]", ""))).map(line => Row(line._1.toInt, line._2.toInt, line._3.toDouble))

    val artistAliasRdd = artistAliasRddRaw.map(line => (line._1.replace("[", ""), line._2.replace("]", ""))).filter(line => (line._1 != "")).map(line => Row(line._1.toInt, line._2.toInt))

    val artistAliasDF = spark.createDataFrame(artistAliasRdd, artistAliasSchema)

//    artistAliasDF.show(5)

    val userArtistDF = spark.createDataFrame(userArtistRdd, userArtistSchema)

//    userArtistDF.show(5)

    userArtistDF.createOrReplaceTempView("userArtist")

    artistAliasDF.createOrReplaceTempView("artistAlias")

    val joinedDF = spark.sql("select uda.userId as user_id, uda.artistId as artist_id, uda.playcount as playcount, aa1.goodId as good_id from userartist uda left outer join artistalias aa1 on uda.artistId = aa1.badId")

    joinedDF.createOrReplaceTempView("joined")

    val cleanedDF = spark.sql("select user_id, good_id as artist_id, playcount from joined where good_id IS NOT NULL union select user_id, artist_id, playcount from joined where good_id IS NULL")

    val ratingRdd = cleanedDF.map(line => Rating(line(0).asInstanceOf[Int], line(1).asInstanceOf[Int], line(2).asInstanceOf[Double])).rdd

    val Array(training, testing) = ratingRdd.randomSplit(Array(0.8, 0.2))

    spark.sparkContext.setCheckpointDir("data/recommendation/checkpoints/")

    val model = ALS.trainImplicit(training, 40, 20, 0.01, 0.02)

    val userArtist = testing.map { case Rating(userId, artistId, pCount) => (userId, artistId) }
    val predictions = model.predict(userArtist).map { case Rating(userId, artistId, pCount) => (userId, (artistId, pCount)) }
    val sortedPred = predictions.sortBy(-_._2._2)

    val actual = testing.map { case Rating(userId, artistId, playCount) => (userId, (artistId, playCount)) }
    val joined = actual.join(sortedPred)

    val rankedUsers = joined.map(l => (l._1, getPercentileRank(Array(l._2._1), Array(l._2._2))))
    val sum = rankedUsers.map(l => l._2).sum
    val count = rankedUsers.count()
    val averageRank = sum / count
    
    println("Average Rank of model is: " + averageRank)

  }
  
  def addArrayElems(ad: Array[Double]): Double = {
    var sum = 0.0
    var i = 0
    while (i < ad.length) { sum += ad(i); i += 1 }
    sum
  }

  def getPercentileRank(actual: Array[(Int, Double)], predicted: Array[(Int, Double)]): Double = {
    val totalElems = predicted.size
    var percentileRank: Double = 0

    val actRatArray = actual.map(l => l._2)

    val actualRateSum = addArrayElems(actRatArray)
    predicted.map(line => percentileRank += ((line._2 * (predicted.indexOf(line) + 1) / totalElems)).toDouble / actualRateSum)
    percentileRank
  }
}