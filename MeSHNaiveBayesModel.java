package MeSHClassification;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class MeSHNaiveBayesModel extends MeSHClassificationModel<NaiveBayesModel> {

	public MeSHNaiveBayesModel(int class_num, int vocab_size, JavaSparkContext sc, String modeltype, 
			ArrayList<Vector> train_data, ArrayList<int[]> train_labels,
			JavaRDD<Vector> dev_data_rdd, ArrayList<int[]> dev_labels) {
		super(class_num, vocab_size, sc);
		// TODO Auto-generated constructor stub
		for (int class_id = 0; class_id < class_num; class_id++) {
			ArrayList<LabeledPoint> data = new ArrayList<LabeledPoint>();
			for (int i = 0; i < train_data.size(); i++) {
				data.add(new LabeledPoint(train_labels.get(i)[class_id], train_data.get(i)));
			}
			
			double[] smoothParams = new double[]{0.5, 1.0, 1.5, 2.0, 5.0};
			double bestf1 = 0.0;
			NaiveBayesModel bestModel = null;
			for (double lambda : smoothParams) {
				NaiveBayesModel model = NaiveBayes.train(sc.parallelize(data).rdd(), lambda, modeltype);
				double f1 = EvaluteResult(class_id, dev_data_rdd, dev_labels, model);
				if (f1 > bestf1) {
					bestf1 = f1;
					bestModel = model;
				}
			}
			models.add(bestModel);
		}
	}

}
