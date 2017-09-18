package wregret

import java.sql.Timestamp
import java.time.{ZoneId, ZonedDateTime}

import com.cloudera.sparkts.models.{ARIMA, HoltWinters}
import com.cloudera.sparkts.{DateTimeIndex, MonthFrequency, TimeSeriesRDD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.dmg.pmml.TimeSeriesModel

object TimeSeries {

  def main(args: Array[String]): Unit = {
    var spark=SparkSession.builder().appName("timeseries").master("local").getOrCreate()
    var sc=spark.sparkContext
    val sql=spark.sqlContext

    //导入数据，创建DF
    val row=sc.textFile("traindata/timeseriesdata.txt").map{
      line=>
        val tokens=line.split("\t")
        val dt=ZonedDateTime.of(tokens(0).substring(0,4).toInt,tokens(0).substring(4).toInt,1,0,0,0,0,ZoneId.systemDefault())
        val profit=tokens(1).toDouble
        Row(Timestamp.from(dt.toInstant),"key",profit)
    }
    val fields=Seq(
      StructField("Timestamp",TimestampType,true),
      StructField("Key", StringType, true),
      StructField("Profit",DoubleType,true)
    )
    val schema=StructType(fields)
    val df=spark.createDataFrame(row,schema)

    //创建数据中的时间跨度
    val zone=ZoneId.systemDefault()
    val dtIndex=DateTimeIndex.uniformFromInterval(
      //起始时间
      ZonedDateTime.of(2010, 1, 1, 0, 0, 0, 0, zone),
      //结束时间
      ZonedDateTime.of(2011, 12, 1, 0, 0, 0, 0, zone),
      new MonthFrequency(1)
    )

    //创建TimeSeriesRDD(key, DenseVector(series))
    val trainTsRDD=TimeSeriesRDD.timeSeriesRDDFromObservations(dtIndex,df,"Timestamp","Key","Profit")

    //HoltWinters
    val modelRDD=trainTsRDD.map{
      line=>
        line match{
          case(key,denseVector)=>(HoltWinters.fitModel(denseVector,4,"additive"/*"Multiplicative"*/),denseVector)
        }
    }

    var forecastArray=new Array[Double](3)
    val forecastVector=Vectors.dense(forecastArray)

    val forecast=modelRDD.map{
      row=>
        row match{
          case (model,vector)=>{
            model.forecast(vector,forecastVector)
          }
        }
    }

    val forecastValue=forecast.map(_.toArray.mkString(","))
    forecastValue.collect().map{
      row=>println("HoltWinters forecast of next 3 observations: "+row)
    }

    //ARIMA
    /*val modelRDD=trainTsRDD.map{
      line=>
        line match{
          case(key,vector)=>(ARIMA.autoFit(vector),vector)
        }
    }
    val coefficients=modelRDD.map{
      line=>
        line match{
          case(model,vector)=>{
            (model.coefficients.mkString(","),(model.p,model.d,model.q))
          }
        }
    }
    val forecast=modelRDD.map{
      row=>
        row match{
          case(model,vector)=>{
            model.forecast(vector,3)
          }
        }
    }
    val forecastValue=forecast.map(_.toArray.mkString(","))

    val forecastValueRDD=forecastValue.map{
      parts=>
        val array=parts.split(",")
        for(i<-array.length-3 until array.length) yield array(i)
    }.map(_.toArray.mkString(","))

    var result=List[Double](3)

    forecastValueRDD.collect().foreach(println(_))*/

    spark.stop()





  }

}
