package wregret

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.sql._
import org.apache.log4j.{Level, Logger}

object App {
  case class user(id:Double,gender:Double,job:Double,education:Double,marriage:Double,residence:Double)
  case class record(id:Double,label:Double)

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)



    var spark=SparkSession.builder().appName("logisticregression").master("local").getOrCreate()
    var sc=spark.sparkContext
    val sq=spark.sqlContext
    sc.setLogLevel("WARN")

    import sq.implicits._

    //用户信息
    var userInfo=sc.textFile("traindata/user_info_train.txt").map(_.split(","))
    //逾期记录
    var overDue=sc.textFile("traindata/overdue_train.txt").map(_.split(","))

    val userInfoDF=userInfo.map(u=>user(u(0).toDouble,u(1).toDouble,u(2).toDouble,u(3).toDouble,u(4).toDouble,u(5).toDouble)).toDF()
    val overDueDF=overDue.map(o=>record(o(0).toDouble,o(1).toDouble)).toDF()
    val resultDF=userInfoDF.join(overDueDF,"id")

    //2,2,3,4,1,0
    val input=resultDF.rdd.map(r=>LabeledPoint(r.getDouble(6),Vectors.dense(Array(r.getDouble(1),r.getDouble(2),r.getDouble(3),r.getDouble(4),r.getDouble(5)))))
    val model=new LogisticRegressionWithLBFGS().setNumClasses(2).run(input)

    println(model.predict(Vectors.dense(2.0,2.0,3.0,4.0,1.0)))


    model.save(sc,"model")

    spark.stop()


  }
}
