using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using System;
using System.IO;

namespace SentimentAnalysisProject.Experiments
{
    public class Experiment2
    {
        private static readonly string YelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt"); // label index 1, sentimentText index 0
        private static readonly string YelpLemmatizedDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelpLemmatized.txt"); // label index 1, sentimentText index 0
        private static readonly string ImdbLargeDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "labeledTrainData.tsv"); // label index 1, sentimentText index 2
        private static readonly string ImdbLargeLemmatizedDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "ImdbLargeLemmatized.tsv"); // label index 1, sentimentText index 2

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
            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpDataPath);
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(ImdbLargeDataPath, hasHeader: true);
            mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
            Console.WriteLine();
            Console.WriteLine("---------No Punctuations removal------------------");
            featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = false
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true},
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

            };
            mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
            Console.WriteLine();
            Console.WriteLine("-------Stop Words Removal + No Punctuations Removal-----------------");
            featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                KeepPunctuations = false
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true},
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

            };
            mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
            Console.WriteLine();
            Console.WriteLine("---------Lemmatization------------------");
            //IDataView dataViewLemma = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpLemmatizedDataPath);
            IDataView dataViewLemma = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(ImdbLargeLemmatizedDataPath, hasHeader: true);

            featureOptions = new TextFeaturizingEstimator.Options
            {
            };
            mlHelper.TrainAndEvaluateSdca(dataViewLemma, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLemma, null, featureOptions, crossValidation: true);
            Console.WriteLine();
            Console.WriteLine("---------Lemmatization + Stop Words Removal------------------");
            featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
            };
            mlHelper.TrainAndEvaluateSdca(dataViewLemma, null, featureOptions, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLemma, null, featureOptions, crossValidation: true);
        }

        private static void NGramsExperiment()
        {
            
            var mlHelper = new MachineLearningHelper(seed: 1, removeStopWords: false);
            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpDataPath, hasHeader: false);
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(ImdbLargeDataPath, hasHeader: true);
            TextWriter oldOut = Console.Out;
            using (var fs = new FileStream("data/reportImdbPreprocess1.txt", FileMode.OpenOrCreate, FileAccess.Write))
            using (var writer = new StreamWriter(fs))
            {
                Console.SetOut(writer);
                for (int i = 1; i < 2; i++)
                {
                    foreach (bool boolVal in new[] { false, true })
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
                                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = i, UseAllLengths = boolVal, Weighting = weightingCriterion },
                                KeepPunctuations = true
                                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

                            };
                            //IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpDataPath);
                            mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, crossValidation: true);
                            mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, crossValidation: true);
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
