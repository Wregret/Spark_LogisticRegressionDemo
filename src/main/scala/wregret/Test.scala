package wregret



import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

object Test {
  def main(args: Array[String]): Unit = {
    var spark=SparkSession.builder().appName("logisticregression").master("local").getOrCreate()
    var sc=spark.sparkContext

    val model=LogisticRegressionModel.load(sc,"model")
    println(model.predict(Vectors.dense(2.0,2.0,3.0,4.0,1.0)))
  }

}
