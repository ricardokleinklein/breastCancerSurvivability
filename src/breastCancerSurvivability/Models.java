package breastCancerSurvivability;

import java.util.ArrayList;
import java.util.Random;
import java.io.FileWriter;
import java.io.BufferedWriter;

import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.adtree.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.AbstractClassifier.*;

import breastCancerSurvivability.Dataset;
import breastCancerSurvivability.Hyperparams;


public class Models {
	
	private static double mean(double[] array) {
		int size = array.length;
		double m = 0.0;
		int realSize = 0;
		for (int i = 0; i < size; i++) {
			if (!Double.isNaN(array[i])) {
				m += array[i];
				realSize += 1;
			}
		}
		return m / realSize;
	}

	private static double getAUC(Classifier cla, Instances data) throws Exception{
		EvaluationUtils eval = new EvaluationUtils();
		ArrayList<Prediction> pred = eval.getTestPredictions(cla, data);
		ThresholdCurve TC = new ThresholdCurve();
		Instances result = TC.getCurve(pred);
		return ThresholdCurve.getROCArea(result);
	}
	
	private static void predictionAllYears(CostSensitiveClassifier cla, 
			Instances data, String filename, int nFolds) throws Exception {
		/** Prediction of AUC for every stage for either the joint model and
		 * stage-specific ones.
		 */
		Dataset.rmFileExists(filename);
		BufferedWriter file = new BufferedWriter(new FileWriter(filename, true));
		double[] AUC_joint = new double[nFolds];
		double[] AUC_stage = new double[nFolds];
		
		// Split dataset in train and test folds
		Instances[] trainFold = Dataset.getTrainCVFolds(data, nFolds);
		Instances[] testFold = Dataset.getTestCVFolds(data, nFolds);
		Instances trainData = new Instances(data, 0);
		Instances testData = new Instances(data, 0);
		
		file.write("Stage\tJoint\tStage-specific\n");
		int[] stage = {-1, 1, 2, 4};
		for (int s = 0; s < stage.length; s++) {
			for (int f = 0; f < nFolds; f++) {
				trainData = trainFold[f];
				testData = testFold[f];
				if (s != 0) { testData = Dataset.filterByStage(testData, stage[s]); }
				
				System.out.println("Training on fold k = " + f);
				cla.buildClassifier(trainData);
				AUC_joint[f] = getAUC(cla, testData);
				
				if (s != 0) { trainData = Dataset.filterByStage(trainData, stage[s]); }
				cla.buildClassifier(trainData);
				AUC_stage[f] = getAUC(cla, testData);
			}
			double meanAUC_joint = mean(AUC_joint);
			double meanAUC_stage = mean(AUC_stage);
			file.write(Dataset.getStageName(s) + "\t" + meanAUC_joint + "\t" + meanAUC_stage + "\n");
		}
		file.close();
	}
	
	private static void predictionPerYear(CostSensitiveClassifier cla,
			Instances data, String filename, int nFolds, int[] window,
			int firstYear, int lastYear) throws Exception {
		/** Prediction of AUC for every stage for either the joint model and
		 * stage-specific ones per year, thus getting the evolution.
		 */
		Dataset.rmFileExists(filename);
		BufferedWriter file = new BufferedWriter(new FileWriter(filename, true));
		int[] stage = {1, 2, 4};
		// Data initialization
		Instances filteredYears = new Instances(data, 0);
		file.write("Stage\tYear\tWindow\tJointAUC\tStageAUC\n");
		
		for (int s = 0; s < stage.length; s++) {
			int W = window[s];
			for (int y = firstYear; y <= (lastYear + 1 - W); y++) {
				filteredYears = Dataset.filterCumulativeYears(data, y, y + W - 1);
				Instances[] trainFolds = Dataset.getTrainCVFolds(filteredYears, nFolds);
				Instances[] testFolds = Dataset.getTestCVFolds(filteredYears, nFolds);
				double[] jointAUC = new double[nFolds];
				double[] stageAUC = new double[nFolds];
				
				for (int f = 0; f < nFolds; f++) {
					Instances trainTS1 = trainFolds[f];
					Instances testTS1 = testFolds[f];
					
					Instances trainTS2 = Dataset.filterByStage(trainFolds[f], stage[s]);
					Instances testTS2 = Dataset.filterByStage(testFolds[f], stage[s]);
					
					// Subset with same number of instances for fair comparison
					trainTS1 = Dataset.getNInstances(trainTS1, trainTS2.numInstances());
					testTS1 = Dataset.getNInstances(testTS1, testTS2.numInstances());
					
					// System.out.println("Joint instances: " + testTS1.numInstances() + " Stage instances: " + testTS2.numInstances());
					
					cla.buildClassifier(trainTS1);
					jointAUC[f] = getAUC(cla, testTS1);
					
					cla.buildClassifier(trainTS2);
					stageAUC[f] = getAUC(cla, testTS2);
					
					// System.out.println("Joint - " + jointAUC[f] + " stage - " + stageAUC[f]);
				}
				double meanJointAUC = mean(jointAUC);
				double meanStageAUC = mean(stageAUC);
				file.write(Dataset.getStageName(s) + "\t" + y + "\t" 
				+ W + "\t" + meanJointAUC + "\t" + meanStageAUC + "\n");
				System.out.println(Dataset.getStageName(s) + " - Year: " + y +
						" - Joint: " + meanJointAUC + " - Stage: " + meanStageAUC);
			}
		}
		file.close();
	}
	
	private static void ageing(CostSensitiveClassifier cla, Instances data,
			String filename, int[] window, int firstYear, int lastYear) throws Exception {
		/** Train a model over a window of years and evaluate its performance
		 * in previous as well as in following years.
		 */
		Dataset.rmFileExists(filename);
		BufferedWriter file = new BufferedWriter(new FileWriter(filename, true));
		int[] stage = {-1, 1, 2, 4};
		
		for (int s = 0; s < stage.length; s++) {
			int W = window[s];
			for (int y = firstYear; y < (lastYear - W + 1); y += 5) {
				Instances trainData = Dataset.filterCumulativeYears(data, y, y + W - 1);
				if (s != 0) {trainData = Dataset.filterByStage(trainData, stage[s]);}
				cla.buildClassifier(trainData);
				System.out.print("Stage: " + Dataset.getStageName(s) +  " Year: " + y);
				file.write(Dataset.getStageName(s) + "\t" + y + "\t");
				for (int testYear = y + W; testYear < (lastYear + 1); testYear++) {
					Instances testData = Dataset.filterByYear(data, testYear);
					if (s != 0) {testData = Dataset.filterByStage(testData, stage[s]);}
					double AUC = getAUC(cla, testData);
					file.write(AUC + "\t");
				}
				System.out.print("\n");
				file.write("\n");
			}
		}
		file.close();
	}
	
	public static void main(String[] args) throws Exception {
		// Read dataset
		Instances data = Dataset.getData();
		int nFolds = 10;
		Dataset.stratify(data, nFolds);
		// Prepare learning algorithms
		
		// A) Naive Bayes
		String filename = "bayesPrediction.txt";
		NaiveBayes bayes = new NaiveBayes();
		CostSensitiveClassifier meta = new CostSensitiveClassifier();
		CostMatrix CM = new CostMatrix(2);
		meta.setCostMatrix(CM);
		meta.setClassifier(bayes);
		Hyperparams.setCostMatrix(CM, 9, 2);
		// predictionAllYears(meta, data, filename, nFolds);
		int[] window = {3, 3, 4, 4};
		// String filename2 = "bayesPrediction_yearly.txt";
		// predictionPerYear(meta, data, filename2, nFolds, window, 2004, 2009);
		// ageing(meta, data, "bayesAging.txt", window, 1973, 2009);
		
		/*
		// B) Logistic
		String filename = "logicPrediction.txt";
		Logistic log = new Logistic();
		CostSensitiveClassifier meta = new CostSensitiveClassifier();
		CostMatrix CM = new CostMatrix(2);
		meta.setCostMatrix(CM);
		meta.setClassifier(log);
		Hyperparams.setCostMatrix(CM, 11, 17);
		// predictionAllYears(meta, data, filename, nFolds);
		int[] window = {4, 5, 5, 5};
		// String filename2 = "logicPrediction_yearly.txt";
		// predictionPerYear(meta, data, filename2, nFolds, window, 2004, 2009);
		aging(meta, data, "logicAging.txt", window, 1973, 2009);
		*/
		/*
		// C) ADTree
		String filename = "treePrediction.txt";
		ADTree tree = new ADTree();
		CostSensitiveClassifier meta = new CostSensitiveClassifier();
		CostMatrix CM = new CostMatrix(2);
		meta.setCostMatrix(CM);
		meta.setClassifier(tree);
		Hyperparams.setCostMatrix(CM, 0.5, 17);
		// predictionAllYears(meta, data, filename, nFolds);
		int[] window = {4, 5, 3, 5};
		// String filename2 = "treePrediction_yearly.txt";
		// predictionPerYear(meta, data, filename2, nFolds, window, 2004, 2009);
		aging(meta, data, "treeAging.txt", window, 1973, 2009);
		*/
	}
}
