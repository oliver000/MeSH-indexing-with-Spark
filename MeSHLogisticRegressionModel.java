package MeSHClassification;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class MeSHLogisticRegressionModel extends MeSHClassificationModel<LogisticRegressionModel> {

	public MeSHLogisticRegressionModel(int class_num, int vocab_size, JavaSparkContext sc, 
			ArrayList<Vector> train_data, ArrayList<int[]> train_labels,
			JavaRDD<Vector> dev_data_rdd, ArrayList<int[]> dev_labels) {
		super(class_num, vocab_size, sc);
		// TODO Auto-generated constructor stub
		for (int class_id = 0; class_id < class_num; class_id++) {
			ArrayList<LabeledPoint> data = new ArrayList<LabeledPoint>(); 
			for (int i = 0; i < train_data.size(); i++) {
				data.add(new LabeledPoint((double)train_labels.get(i)[class_id], train_data.get(i)));
			}

			double[] regParams = {0.005, 0.01, 0.05, 0.1, 0.5};
//			double[] regParams = {0.01};
			double bestf1 = 0.0;
			LogisticRegressionModel bestModel = null;
			for (double regParam : regParams) {
				LogisticRegressionWithLBFGS lrwlbfgs = new LogisticRegressionWithLBFGS();
				lrwlbfgs.optimizer().setRegParam(regParam);
				LogisticRegressionModel model = lrwlbfgs.setIntercept(false)
						.run(sc.parallelize(data).rdd());
				double f1 = EvaluteResult(class_id, dev_data_rdd, dev_labels, model);
				PrintLog("data/lrf1_" + class_id + ".txt", new double[]{class_id, f1});
				if (f1 > bestf1) {
					bestf1 = f1;
					bestModel = model;
				}
			}
			models.add(bestModel);
		}	
	}
	
}
