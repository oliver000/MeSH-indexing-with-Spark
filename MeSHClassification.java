package MeSHClassification;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.util.MLUtils;

public class MeSHClassification {
	
	private int vocab_size;
	private int class_num;
	
	private JavaSparkContext sc;
	private ArrayList<Vector> train_data;
	private JavaRDD<Vector> train_data_rdd, dev_data_rdd, test_data_rdd;
	private ArrayList<int[]> train_labels, dev_labels, test_labels; 
	
	public MeSHClassification(JavaSparkContext sc, int vocab_size, int class_num) {
		this.sc = sc;
		this.vocab_size = vocab_size;
		this.class_num = class_num;
	}
	
	private ArrayList<Vector> LoadFeatures(String data_file) {
		ArrayList<Vector> res = new ArrayList<Vector>();
		try {
			BufferedReader rd;
			rd = new BufferedReader(new FileReader(data_file));
			String line = null;
			while ((line = rd.readLine()) != null) {
				String[] lineList = line.split(" ");
				ArrayList<Integer> index = new ArrayList<Integer>();
				ArrayList<Double> values = new ArrayList<Double>();
				for (String word : lineList) {
					String[] ww = word.split(":");
					index.add(Integer.parseInt(ww[0]));
					values.add(Double.parseDouble(ww[1]));
				}
				Integer[] i = new Integer[index.size()];
				Double[] v = new Double[values.size()];
				res.add(MLUtils.appendBias(new SparseVector(vocab_size, 
						ArrayUtils.toPrimitive(index.toArray(i)), ArrayUtils.toPrimitive(values.toArray(v)))));
			}
			rd.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return res;
	}
	
	public void LoadFeaturesData(String train_file, String dev_file, String test_file) {
		train_data = LoadFeatures(train_file);
		train_data_rdd = sc.parallelize(train_data);
		dev_data_rdd = sc.parallelize(LoadFeatures(dev_file));
		test_data_rdd = sc.parallelize(LoadFeatures(test_file));
	}
	
	private ArrayList<int[]> LoadLabels(String labels_file) {
		ArrayList<int[]> res = new ArrayList<int[]>();
		try {
			BufferedReader rd = new BufferedReader(new FileReader(labels_file));
			String line;
			while ((line = rd.readLine()) != null) {
				String[] lineList = line.split(" ");
				int[] labels = new int[class_num];
				for (int i = 0; i < class_num; i++) {
					labels[i] = Integer.parseInt(lineList[i]);
				}
				res.add(labels);
			}
			rd.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return res;
	}
	
	public void LoadLabelsData(String train_file, String dev_file, String test_file) {
		train_labels = LoadLabels(train_file);
		dev_labels = LoadLabels(dev_file);
		test_labels = LoadLabels(test_file);
	}
	
	public void TrainSVMModel(String features) {
		MeSHSVMModel model = new MeSHSVMModel(class_num, vocab_size, sc,
				train_data, train_labels, dev_data_rdd, dev_labels);
		double[][] res = new double[3][];
		res[0] = model.EvaluteResult(train_data_rdd, train_labels);
		res[1] = model.EvaluteResult(dev_data_rdd, dev_labels);
		res[2] = model.EvaluteResult(test_data_rdd, test_labels);
		WriteResult("results/svm_" + features + ".txt", res);
	}
	
	public void TrainLRModel(String features) {
		MeSHLogisticRegressionModel model = new MeSHLogisticRegressionModel(class_num, vocab_size, sc,
				train_data, train_labels, dev_data_rdd, dev_labels);
		double[][] res = new double[3][];
		res[0] = model.EvaluteResult(train_data_rdd, train_labels);
		res[1] = model.EvaluteResult(dev_data_rdd, dev_labels);
		res[2] = model.EvaluteResult(test_data_rdd, test_labels);
		WriteResult("results/lr_" + features + ".txt", res);
	}
	
	public void TrainNaiveBayesModel(String modeltype) {
		MeSHNaiveBayesModel model = new MeSHNaiveBayesModel(class_num, vocab_size, sc, modeltype,
				train_data, train_labels, dev_data_rdd, dev_labels);
		double[][] res = new double[3][];
		res[0] = model.EvaluteResult(train_data_rdd, train_labels);
		res[1] = model.EvaluteResult(dev_data_rdd, dev_labels);
		res[2] = model.EvaluteResult(test_data_rdd, test_labels);
		WriteResult("results/nb_" + modeltype + ".txt", res);
	}
	
	public void TrainDecisionTree(String features) {
		MeSHDecisionTreeModel model = new MeSHDecisionTreeModel(class_num, vocab_size, sc,
				train_data, train_labels, dev_data_rdd, dev_labels);
		double[][] res = new double[3][];
		res[0] = model.EvaluteResult(train_data_rdd, train_labels);
		res[1] = model.EvaluteResult(dev_data_rdd, dev_labels);
		res[2] = model.EvaluteResult(test_data_rdd, test_labels);
		WriteResult("results/decisiontree_" + features + ".txt", res);
	}
	
	public void TrainRandomForest(String features) {
	}
	
	public void WriteResult(String outputfile, double[][] results) {
		PrintWriter wr;
		try {
			wr = new PrintWriter(new FileWriter(outputfile));
			for (int i = 0; i < results.length; i++) {
				for (int j = 0; j < results[0].length; j++)
					wr.print(results[i][j] + " ");
				wr.println();
			}
			wr.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		SparkConf sparkConf = 
				new SparkConf().setAppName("MeSH Classification")
				.setMaster("local[*]")
				.set("spark.driver.maxResultSize", "30G")
				.set("spark.executor.memory", "30G")
				.set("spark.python.worker.memory", "30G")
				.set("spark.driver.cores", "4")
				.set("spark.default.parallelism", "8");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		
		MeSHClassification classification = new MeSHClassification(sc, 120000, 10);
		classification.LoadLabelsData("data/train_label.txt", "data/dev_label.txt", "data/test_label.txt");
		
//		classification.LoadFeaturesData("data/train_tfidf.txt", "data/dev_tfidf.txt", "data/test_tfidf.txt");
//		classification.TrainSVMModel("tfidf");
//		classification.TrainLRModel("tfidf");
//		classification.TrainDecisionTree("tfidf");
		
//		classification.LoadFeaturesData("data/train_freq.txt", "data/dev_freq.txt", "data/test_freq.txt");
//		classification.TrainLRModel("freq");
//		classification.TrainDecisionTree("freq");
		
		classification.LoadFeaturesData("data/train_occurrence.txt", "data/dev_occurrence.txt", "data/test_occurrence.txt");
//		classification.TrainNaiveBayesModel("bernoulli");
		classification.TrainDecisionTree("occurrence");

//		classification.LoadFeaturesData("data/train_occ_freq.txt", "data/dev_occ_freq.txt", "data/test_occ_freq.txt");
//		classification.TrainNaiveBayesModel("multinomial");
	}
}
