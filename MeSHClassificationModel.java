package MeSHClassification;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
//import org.apache.spark.mllib.classification.ClassificationModel;
import org.apache.spark.mllib.linalg.Vector;

public class MeSHClassificationModel<ClassificationModel> {

	protected int class_num;
	protected int vocab_size;
	protected JavaSparkContext sc;
	protected ArrayList<ClassificationModel> models;
	
	public MeSHClassificationModel(int class_num, int vocab_size, JavaSparkContext sc) {
		this.class_num = class_num;
		this.vocab_size = vocab_size;
		this.sc = sc;
		models = new ArrayList<ClassificationModel>();
	}
	
	protected double EvaluteResult(int class_id, JavaRDD<Vector> features, ArrayList<int[]> labels, ClassificationModel model) {
//		System.out.println(features.count() + " " + class_id);
		try {
			List<Double> res =
					((JavaRDD<Double>)model.getClass()
							.getMethod("predict", features.getClass()).invoke(model, features)).collect();
			int TP = 0, TPFP = 0, TPFN = 0;
			for (int i = 0; i < res.size(); i++) {
				if ((Double) res.get(i) > 0.0)
					TPFN++;
				if (labels.get(i)[class_id] > 0.0)
					TPFP++;
				if ((Double) res.get(i) > 0.0 && labels.get(i)[class_id] > 0.0)
					TP++;
			}
			
			double prec = ((double) TP) / TPFP;
			double recall = ((double) TP) / TPFN;
			
			return (2*prec*recall)/(prec+recall);
		} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException | NoSuchMethodException
				| SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		List<Double> res = model.predict(features).collect();
//		System.out.println(features.count() + " " + class_id + " " + res.size());
//		try {
//			if (class_id == 0) {
//				PrintWriter wr = new PrintWriter(new FileWriter("data/prediction.txt", true));
//				wr.println("---------------------------------------------------------");
//				for (Double r : res) {
//					if (r < 0.5) wr.println(r);
//				}
//				wr.close();
//			}
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		return 0.0;
	}
	
	public double[] EvaluteResult(JavaRDD<Vector> features, ArrayList<int[]> labels) {
		// TODO Auto-generated method stub
		
		double[] res = new double[class_num + 1];
		for (int class_id = 0; class_id < class_num; class_id++) {
			res[class_id + 1] = EvaluteResult(class_id, features, labels, models.get(class_id));
			res[0] += res[class_id + 1];
		}
		res[0] /= class_num;
		return res;
	}
	
	protected void PrintLog(String logfile, double[] loss) {
		try {
			PrintWriter wr = new PrintWriter(new FileWriter(logfile, true));
			wr.println("---------------------------------------------------------");
			for (double l : loss) {
				wr.println(l);
			}
			wr.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
