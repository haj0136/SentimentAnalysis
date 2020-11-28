using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using System;
using System.IO;

namespace SentimentAnalysisProject.Experiments
{
    public class Experiment2
    {
        public static void StartExperiment()
        {
            //TextPreprocessing();
            NGramsExperiment();
        }

        private static void TextPreprocessing()
        {
            Console.WriteLine("---------Stop Words Removal-------------------");
            var mlHelper = new MachineLearningHelper(seed: 1, removeStopWords: false);
            
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true},
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

            };
            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(Paths.YelpDataPath);
            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kDataPath, hasHeader: true);
            IDataView largeDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon1MTrainPath, hasHeader: true);
            IDataView testDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon400KTestPath, hasHeader: true);
            //mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateSdca(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
            //mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
            Console.WriteLine();
            Console.WriteLine("---------Punctuations removal------------------");
            featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = false
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true},
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

            };
            //mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateSdca(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
            //mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
            Console.WriteLine();
            Console.WriteLine("-------Stop Words Removal + Punctuations Removal-----------------");
            featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                KeepPunctuations = false
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true},
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

            };
            //mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateSdca(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
            //mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
            Console.WriteLine();
            
            Console.WriteLine("---------Lemmatization------------------");
            //IDataView dataViewLemma = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(Paths.YelpLemmatizedDataPath);
            //IDataView dataViewLemma = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kLemmatizedDataPath, hasHeader: true);
            IDataView largeDataViewLemm = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.AmazonTrainLemmatizedPath, hasHeader: true);
            IDataView testDataViewLemm = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.AmazonTestLemmatizedPath, hasHeader: true);

            featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = null
            };
            //mlHelper.TrainAndEvaluateSdca(dataViewLemma, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateSdca(largeDataViewLemm, null, featureOptions, crossValidation: false, testData: testDataViewLemm);
            //mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLemma, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(largeDataViewLemm, null, featureOptions, crossValidation: false, testData: testDataViewLemm);
            Console.WriteLine();
            Console.WriteLine("---------Lemmatization + Stop Words Removal------------------");
            featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
            };
            //mlHelper.TrainAndEvaluateSdca(dataViewLemma, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateSdca(largeDataViewLemm, null, featureOptions, crossValidation: false, testData: testDataViewLemm);
            //mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLemma, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(largeDataViewLemm, null, featureOptions, crossValidation: false, testData: testDataViewLemm);
        }

        private static void NGramsExperiment()
        {
            
            var mlHelper = new MachineLearningHelper(seed: 1, removeStopWords: false);
            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(Paths.YelpDataPath, hasHeader: false);
            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kDataPath, hasHeader: true);
            IDataView largeDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon1MTrainPath, hasHeader: true);
            IDataView testDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon400KTestPath, hasHeader: true);
            TextWriter oldOut = Console.Out;
            using (var fs = new FileStream("data/reportAmazonPreprocess.txt", FileMode.OpenOrCreate, FileAccess.Write))
            using (var writer = new StreamWriter(fs))
            {
                Console.SetOut(writer);
                for (int i = 1; i < 4; i++)
                {
                    foreach (bool boolVal in new[] { false }) // boolVal = Use all lengths of n-grams?
                    {
                        if (boolVal && i == 1)
                        {
                            continue;
                        }
                        foreach (NgramExtractingEstimator.WeightingCriteria weightingCriterion in Enum.GetValues(typeof(NgramExtractingEstimator.WeightingCriteria)))
                        {
                            Console.WriteLine($"N-gram Length = {i}");
                            Console.WriteLine($"Used all lengths = {boolVal}");
                            Console.WriteLine($"Used weighting criteria = {weightingCriterion}");
                            Console.SetOut(oldOut);
                            Console.WriteLine($"N-gram Length = {i}");
                            Console.WriteLine($"Used all lengths = {boolVal}");
                            Console.WriteLine($"Used weighting criteria = {weightingCriterion}");
                            Console.SetOut(writer);
                            var featureOptions = new TextFeaturizingEstimator.Options
                            {
                                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = i, UseAllLengths = boolVal, Weighting = weightingCriterion },
                                KeepPunctuations = false
                                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

                            };
                            //mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
                            mlHelper.TrainAndEvaluateSdca(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
                            //mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
                            mlHelper.TrainAndEvaluateAveragedPerceptron(largeDataView, null, featureOptions, crossValidation: false, testData: testDataView);
                            Console.WriteLine("-----------------------------------------------------------------");
                            Console.WriteLine();
                            writer.Flush();
                            fs.Flush();
                        }
                    }
                }
                Console.SetOut(oldOut);
            }
        }
    }
}
