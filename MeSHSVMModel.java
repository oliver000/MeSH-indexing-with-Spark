package MeSHClassification;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.HingeGradient;
import org.apache.spark.mllib.optimization.LBFGS;
import org.apache.spark.mllib.optimization.SquaredL2Updater;

import scala.Tuple2;

public class MeSHSVMModel extends MeSHClassificationModel<SVMModel> {
	
	public MeSHSVMModel(int class_num, int vocab_size, JavaSparkContext sc, 
			ArrayList<Vector> train_data, ArrayList<int[]> train_labels,
			JavaRDD<Vector> dev_data_rdd, ArrayList<int[]> dev_labels) {
		super(class_num, vocab_size, sc);
		// TODO Auto-generated constructor stub
		for (int class_id = 0; class_id < class_num; class_id++) {
			ArrayList<Tuple2<Object, Vector>> data = new ArrayList<Tuple2<Object, Vector>>();
			for (int i = 0; i < train_data.size(); i++) {
				data.add(new Tuple2<Object, Vector>(
						(double)train_labels.get(i)[class_id], train_data.get(i)));
			}
			
			int numCorrections = 10;
			double convergenceTol = 0.0;
			int maxNumIterations = 150;
			double[] regParams = {0.0, 1e-7, 1e-6, 1e-5, 1e-4};
//			double[] regParams = {0.0};
			Vector initialWeightsWithIntercept = Vectors.dense(new double[vocab_size + 1]);
			
			double bestf1 = 0.0;
			SVMModel bestModel = null;
			for (double regParam : regParams) {
				Tuple2<Vector, double[]> result = LBFGS.runLBFGS(
						sc.parallelize(data).rdd(), 
						new HingeGradient(), 
						new SquaredL2Updater(),
						numCorrections,
						convergenceTol,
						maxNumIterations,
						regParam,
						initialWeightsWithIntercept
				);
				PrintLog("data/svmlog_" + class_id + ".txt", result._2());
				SVMModel model = new SVMModel(result._1(), 0.0);
				double f1 = EvaluteResult(class_id, dev_data_rdd, dev_labels, model);
				if (f1 > bestf1) {
					bestf1 = f1;
					bestModel = model;
				}
				PrintLog("data/svmf1.txt", new double[]{class_id, f1});
				initialWeightsWithIntercept = result._1();
			}
			models.add(bestModel);
		}	
	}
	
}
