# breastCancerSurvivability project

This repository contains the required programme to replicate my MSc. thesis, *"Stage-specific breast cancer survivability models over time"*, 
in which we explore how stage-specific models are the preferable option when implementing a machine learning-based solution to the problem of predicting breast cancer survivability.

## Use of this package

The *Eclipse* environment is required for a smooth running of the algorithms. Previously to importing the project, you should be sure that *Weka* API is well integrated in your environment. 

Once you are absolutely sure *Weka* is up and running, proceed to import this repository through *Eclipse*. If you don't know how to do it, please follow [these instructions](http://imtilab.blogspot.com/2016/10/how-to-pushupload-eclipse-project-to.html) to learn how to create, commit and push your project to *Github*.

You should be able to run the experiments with no trouble whatsoever. Please notice that for the sake of debugging and clearance during the development of this project, code is not optimal, and therefore manual modifications on what algorithms to run and hyperparameters must be done in advance.

## Why stage-specific?

When diagnosed with breast cancer, all patients are given an indicator of hazard of their disease. There are several methods, but the one used in our study (we use the [SEER dataset](https://seer.cancer.gov/data/)) considers four stages:

![Table of breast cancer stages classification by SEER dataset.](pics/table_stages.png)

The survivability on each of these stages is different, and it is a major concern to accurately predict the rate of survivability on each of them so physicians can assess better the treatment and life expectancy of new patients in the future. Traditional machine learning models used to this aim consisted of an unified model in which the summary-stage of the disease was taking into account just like any other feature (or attribute) of a patient.

![Unified models in breast cancer survivability prediction](pics/jointmodel.png)

Our perspective, following (Kate, R.J. and Nadig, R.[https://www.ncbi.nlm.nih.gov/pubmed/27919388] builds a separate model for each summary-stage, solving that way the problem of overstimation that arises in evaluating unified models.

![Summary-stage variable breast cancer survivability prediciton models](pics/stage.png)

## Main result

First of all, an analysis on the information gain of the attributes we consider in the dataset, we find that summary-stage is the most relevant attribute in discriminating the instance's label (*survived/not survived*). Nonetheless, when performing this same analysis n each sumary stage separately, the year of diagnosis appears as the most important attribute.

![Information gain in joint model and stage-specific](pics/ig.png)

This suggests that our hypothesis must remain true not only for the whole of the dataset (data are available from 1973 to 2015), but also in a yearly basis, due to the intrinsic importante of the summary-stage. Our results can be summarized in the following table:

![Summar-stage performance on AUC for 3 different machine learning algorithms](pics/prediction.png)

It can be seen that building summary-stage-specific models never worsen the performance of the prediction. On the contrary, it always improves or remains the same.