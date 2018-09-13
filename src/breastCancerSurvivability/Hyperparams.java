package breastCancerSurvivability;

import java.util.Random;
import java.io.FileWriter;
import java.io.BufferedWriter;

import weka.core.Instances;
import weka.classifiers.CostMatrix;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.adtree.*;
import weka.classifiers.functions.Logistic;

import breastCancerSurvivability.Dataset;

public class Hyperparams {
	
	public static void setCostMatrix(CostMatrix cost, double FP, double FN) {
		cost.setElement(0, 1, FN);
		cost.setElement(1, 0, FP);
	}
	
	public static double[] linspace(double min, double max, int points) {  
	    double[] d = new double[points];  
	    for (int i = 0; i < points; i++){  
	        d[i] = min + i * (max - min) / (points - 1);  
	    }  
	    return d;  
	}  
	
	private static void findOptimCostMatrix(CostMatrix CM, CostSensitiveClassifier cla,
			Instances data, int n_exps, String filename) throws Exception {
		/** Finds the optimal values for the cost matrix balancing checking multiple 
		 * random seeds.
		 * Args:
		 * 	(CostMatrix) CM: base cost matrix.
		 * 	(CostSensitiveClassifier) cla: Meta-classifier to analyze.
		 * 	(Instances) data: Dataset over which finding optimal values.
		 * 	(int) n_exps: Number of experiments (# random seeds to try).
		 * 	(String) filename: File exporting results to.
		 */
		Dataset.rmFileExists(filename);
		BufferedWriter file = new BufferedWriter(new FileWriter(filename, true));
		// Initialize variables
		double AUC, bestAUC = 0.0, bestFP = 0.0, bestFN = 0.0;
		Runtime garbage;
		double cost[] = {0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
		file.write("Seed\tFP\tFN\tAUC\n");
		double[] seeds = linspace(1, 500, n_exps);
		for (int idx = 0; idx < seeds.length; idx++) {
			long seed = (long) seeds[idx];
			// Evaluate every possible combination FP-FN
			for (int FP = 0; FP < cost.length; FP++) {
				for (int FN = 0; FN < cost.length; FN++) {
					setCostMatrix(CM, cost[FP], cost[FN]);
					Evaluation eval = new Evaluation(data);
					eval.crossValidateModel(cla, data, 5, new Random(seed));
					ThresholdCurve tc = new ThresholdCurve();
					Instances result = tc.getCurve(eval.predictions());
					AUC = ThresholdCurve.getROCArea(result);
					if (AUC > bestAUC) {
						bestAUC = AUC;
						bestFP = cost[FP];
						bestFN = cost[FN];
					}
					eval = null; tc = null; result = null;
					garbage = Runtime.getRuntime();
					garbage.gc();
				}
			}
			System.out.println("SEED = " + seed + "\tFP = " + bestFP + "\tFN = " + bestFN + "\tAUC = " + bestAUC);
			file.write(seed + "\t" + bestFP + "\t" + bestFN + "\t" + bestAUC + "\n");
		}
		file.close();
	}
	
	private static void findOptimCostMatrix(CostMatrix CM, CostSensitiveClassifier cla, 
			Instances data) throws Exception {
		/** Finds the optimal values for the cost matrix balancing.
		 * Args:
		 * 	(CostSensitiveClassifier) cla: Meta-classifier to analyze.
		 * 	(Instances) data: Dataset over which finding optimal values.
		 */
		// Initialize variables
		double AUC, bestAUC = 0.0, bestFP = 0.0, bestFN = 0.0;
		Runtime garbage;
		double cost[] = {0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

		// Evaluate every possible combination FP-FN
		for (int FP = 0; FP < cost.length; FP++) {
			for (int FN = 0; FN < cost.length; FN++) {
				setCostMatrix(CM, cost[FP], cost[FN]);
				Evaluation eval = new Evaluation(data);
				eval.crossValidateModel(cla, data, 5, new Random(1));
				ThresholdCurve tc = new ThresholdCurve();
				Instances result = tc.getCurve(eval.predictions());
				AUC = ThresholdCurve.getROCArea(result);
				System.out.println(FP + " " + FN + " " + AUC);
				if (AUC > bestAUC) {
					bestAUC = AUC;
					bestFP = cost[FP];
					bestFN = cost[FN];
				}
				eval = null; tc = null; result = null;
				garbage = Runtime.getRuntime();
				garbage.gc();
			}
			System.out.println("FP = " + bestFP + "\tFN = " + bestFN + "\tAUC = " + bestAUC);
		}
	}
	
	private static void findOptimWindow(CostMatrix cost, CostSensitiveClassifier cla, Instances data,
			int firstYear, int lastYear, double FP, double FN, String filename, int n_exps) throws Exception {
		/** Finds the optimal time length for which each algorithm must be trained
		 * at each stage separately and jointly. 
		 * Args:
		 * 	(CostMatrix) cost: Unmodified cost matrix
		 * 	(CostSensitiveClassifier) cla: meta-classifier to analyze
		 * 	(Instances) data: Set of data for training/testing
		 * 	(int) firstYear: Year starting the analysis
		 * 	(int) lastYear: Last year to analyze
		 * 	(double)	FP: False Positive ratio to introduce in `cost`
		 * 	(double) FN: False Negative ratio  to introduce in `cost`
		 * 	(String) filename: File exporting results to.
		 * 	(int) n_exps: Number of experiments (# random seeds to try).
		 */
		
		Dataset.rmFileExists(filename);
		BufferedWriter file = new BufferedWriter(new FileWriter(filename, true));
		
		setCostMatrix(cost, FP, FN);
		int stage[] = {-1, 1, 2, 4}; // -1-all, 1-localized, 2-regional, 4-distant
		double bestAUC[] = {0.0, 0.0, 0.0, 0.0};
		double bestWindow[] = {0, 0, 0, 0};
		double[] seeds = linspace(1, 500, n_exps);
		file.write("Seed\tStage\tWindow\tAUC\n");
		
		for (int idx = 0; idx < seeds.length; idx++) {
			long seed = (long) seeds[idx];
			for (int s = 0; s < stage.length; s++) {
				for (int w = 1; w <= 5; w++) {
					double AUC = 0.0;
					for (int y = firstYear; y <= (lastYear + 1 - w); y++) {
						Instances newData = new Instances(data);
						newData = Dataset.filterCumulativeYears(data, y, y + w - 1);
						if (s != 0) {newData = Dataset.filterByStage(newData, stage[s]); }
					
						Evaluation eval = new Evaluation(data);
						eval.crossValidateModel(cla, newData, 5, new Random(seed));
						ThresholdCurve tc = new ThresholdCurve();
						Instances result = tc.getCurve(eval.predictions());
						AUC += ThresholdCurve.getROCArea(result);
						eval = null; tc = null; result = null;
					}
					AUC /= (lastYear - firstYear + 2 - w);
					if (AUC > bestAUC[s]) {
						bestAUC[s] = AUC;
						bestWindow[s] = w;
					}
				}
				file.write(seed + "\t" + Dataset.getStageName(s) + "\t" + bestWindow[s] + "\t" + bestAUC[s] + "\n");
				System.out.println("Seed: "+ seed + " Stage: "+ Dataset.getStageName(s) +" Window: "+bestWindow[s]+" mean AUC: "+bestAUC[s]);
			}
		}
		file.close();
	}
	
	public static void main(String[] args) throws Exception{
		// Read dataset
		Instances data = Dataset.getData();
		data = Dataset.filterCumulativeYears(data, 2004, 2009);
		System.out.println("Num instances: " + data.numInstances());
		
		// A) Naive Bayes
		NaiveBayes bayes = new NaiveBayes();
		CostSensitiveClassifier meta = new CostSensitiveClassifier();
		CostMatrix CM = new CostMatrix(2);
		meta.setCostMatrix(CM);
		meta.setClassifier(bayes);
		// meta.buildClassifier(data);
		System.out.println("--------- Naive Bayes ---------");
		// findOptimCostMatrix(CM, meta, data, 25, "costMatrix_bayes.txt");
		// findOptimCostMatrix(CM, meta, data);
		// Introduce optimCostMatrix manually if uncomment next line
		// findOptimWindow(CM, meta, data, 2004, 2009, 8, 3);
		findOptimWindow(CM, meta, data, 2004, 2009, 3, 1, "window_bayes.txt", 25);
		
		/*
		// B) Logistic Regression
		Logistic logic = new Logistic();
		CostSensitiveClassifier meta = new CostSensitiveClassifier();
		CostMatrix CM = new CostMatrix(2);
		meta.setCostMatrix(CM);
		meta.setClassifier(logic);
		meta.buildClassifier(data);
		System.out.println("--------- Logistic ---------");
		// findOptimCostMatrix(CM, meta, data);
		// Introduce optimCostMatrix manually if uncomment next line
		findOptimWindow(CM, meta, data, 2004, 2009, 11, 17);
		*/
		/*
		// C) ADTree
		ADTree tree = new ADTree();
		CostSensitiveClassifier meta = new CostSensitiveClassifier();
		CostMatrix CM = new CostMatrix(2);
		meta.setCostMatrix(CM);
		meta.setClassifier(tree);
		meta.buildClassifier(data);
		System.out.println("--------- ADTree ---------");
		// findOptimCostMatrix(meta, data);
		// Introduce optimCostMatrix manually if uncomment next line
		findOptimWindow(CM, meta, data, 2004, 2009, 0.5, 17);
		*/
	}
	
}
