package breastCancerSurvivability;


import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.File;
import java.util.Random;

public class Dataset {
	
	public static void rmFileExists(String filename) {
		File f = new File(filename);
		if (f.exists() && !f.isDirectory()) { f.delete(); }
	}
	
	public static String getStageName(int stage) {
		if (stage < 0) { stage = 0; }
		else if (stage == 4) { stage = 3; }
		String name[] = {"Joint", "localized", "regional", "distant"};
		return name[stage];
	}
	
	private static Instances getDataSource (String filename) throws Exception {
		DataSource source = new DataSource(filename);
		Instances data = source.getDataSet();
		data.setClassIndex(data.attribute("status").index());
		return data;
	}
	
	private static Instances filterInSitu(Instances data) throws Exception {
		/** Removes instances with `in-situ` stage and missing stage.
		 * Args:
		 *  (Instances) data: Original dataset.
		 *  
		 * Returns:
		 * (Instances) stageData: original dataset w/o filtered samples.
		 */
		int stageCol = data.attribute("SEER-historic-stage-A").index();
		int inSituIdx = data.attribute(stageCol).indexOfValue("0") + 1;
		
		RemoveWithValues rmInSitu = new RemoveWithValues();
		rmInSitu.setAttributeIndex(Integer.toString(stageCol + 1));
		rmInSitu.setMatchMissingValues(true);
		rmInSitu.setInvertSelection(false);
		rmInSitu.setNominalIndices(Integer.toString(inSituIdx));
		
		Instances stageData = new Instances(data, 0);
		rmInSitu.setInputFormat(data);
		stageData = Filter.useFilter(data, rmInSitu);
		
		return stageData;
	}

	private static Instances filterMostRecent(Instances data) throws Exception {
		/** Removes data from the last years, since they have not `status`.
		 * Args:
		 *  (Instances) data: Dataset to filter.
		 *  
		 * Returns:
		 * (Instances) solidData: Dataset w/o filtered (last 5 years) incidences.
		 */
		int yearCol = data.attribute("Year-of-diagnosis").index();
		int year[] = {2010, 2011, 2012, 2013, 2014};
		Instances solidData = new Instances(data);

		RemoveWithValues recentYears = new RemoveWithValues();
		recentYears.setAttributeIndex(Integer.toString(yearCol + 1));
		recentYears.setInvertSelection(false);
		
		int num_recent_years = year.length;
		
		for (int i = 0; i < num_recent_years; i++) {
			int yearIdx = data.attribute(yearCol).indexOfValue(Integer.toString(year[i])) + 1;
			recentYears.setNominalIndices(Integer.toString(yearIdx));
			recentYears.setInputFormat(data);
			solidData = Filter.useFilter(solidData, recentYears);
		}
		return solidData;
	}

	public static Instances filterByStage(Instances data, int stage) throws Exception{
		int stageCol = data.attribute("SEER-historic-stage-A").index();
		String S = Integer.toString(stage);
		int StageIdx = data.attribute(stageCol).indexOfValue(S) + 1;	
		
		RemoveWithValues filterStage = new RemoveWithValues();
		filterStage.setAttributeIndex(Integer.toString(stageCol + 1));
		filterStage.setInvertSelection(true);
		filterStage.setNominalIndices(Integer.toString(StageIdx));
		
		Instances stageData = new Instances(data, 0);
		filterStage.setInputFormat(data);
		stageData = Filter.useFilter(data, filterStage);
		
		return stageData;
	}
	
	public static Instances filterByYear(Instances data, int year) throws Exception{
		int yearCol = data.attribute("Year-of-diagnosis").index();
		int yearIdx = data.attribute(yearCol).indexOfValue(Integer.toString(year)) + 1;
		
		RemoveWithValues byYear = new RemoveWithValues();
		byYear.setAttributeIndex(Integer.toString(yearCol + 1));
		byYear.setInvertSelection(true);
		byYear.setNominalIndices(Integer.toString(yearIdx));
		
		Instances yearData = new Instances(data, 0);
		byYear.setInputFormat(data);
		yearData = Filter.useFilter(data, byYear);

		return yearData;
	}
	
	public static Instances getNInstances(Instances data, int N) throws Exception {
		/* Return the first `N` instances from `data`.*/
		Instances subset = new Instances(data, 0, N);
		return subset;
	}
	
	public static Instances filterCumulativeYears(Instances data, int firstYear, int lastYear) 
			throws Exception{
		int originYear = 1973;
		int finalYear = 2010;
		int yearCol = data.attribute("Year-of-diagnosis").index();
		Instances windowData = new Instances(data);
		
		RemoveWithValues rmYear = new RemoveWithValues();
		rmYear.setAttributeIndex(Integer.toString(yearCol + 1));
		rmYear.setInvertSelection(false);
		
		for (int year = originYear; year <= finalYear; year++) {
			if (year < firstYear || year > lastYear) {
				int yearIdx = data.attribute(yearCol).indexOfValue(Integer.toString(year)) + 1;
				rmYear.setNominalIndices(Integer.toString(yearIdx));
				rmYear.setInputFormat(data);
				windowData = Filter.useFilter(windowData, rmYear);
			}
		}
		return windowData;
	}
	
	public static void stratify(Instances data, int nFolds) throws Exception {
		data.stratify(nFolds);
	}

	public static Instances[] getTrainCVFolds(Instances data, int numFolds) throws Exception {
		Instances trainFolds[] = new Instances[numFolds];
		for (int f = 0; f < numFolds; f++) {
			trainFolds[f] = data.trainCV(numFolds, f, new Random(1));
		}
		return trainFolds;
	}
	
	public static Instances[] getTestCVFolds(Instances data, int numFolds) throws Exception {
		Instances testFolds[] = new Instances[numFolds];
		for (int f = 0; f < numFolds; f++) {
			testFolds[f] = data.testCV(numFolds, f);
		}
		return testFolds;
	}

	public static Instances getData() throws Exception{
		String path = "/Users/ricardokleinlein/Desktop/";
		String file = path + "BREAST3W.arff";
		Instances data = getDataSource(file);
		// Dataset removes in-situ incidences & missing stage
		data = filterInSitu(data);
		// Dataset removes incidences later than 2010
		data = filterMostRecent(data);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		String path = "/Users/ricardokleinlein/Desktop/";
		String file = path + "BREAST3W.arff";
		
		Instances data = getDataSource(file);
		// Our dataset is always w/o in-situ incidences nor missing stage
		data = filterInSitu(data);
		// Also cases previous to 2010
		data = filterMostRecent(data);
		System.out.println(data.numInstances());
	}
}
