using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;

namespace SentimentAnalysisProject.Experiments
{
    public class Experiment3
    {
        public static void StartExperiment()
        {
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-us");
            Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("en-US");
            var mlHelper = new MachineLearningHelper(seed: 1, removeStopWords: false);

            //SmallDatasetSDCA(mlHelper);
            //ImdbDatasetSDCA(mlHelper);
            //AmazonDatasetSDCA(mlHelper);

            //SmallDatasetAp(mlHelper);
            //ImdbDatasetAp(mlHelper);
            AmazonDatasetAp(mlHelper);

            //Console.ReadKey();
        }

        private static void SmallDatasetSDCA(MachineLearningHelper mlHelper)
        {
            Console.WriteLine("------------YELP DATASET-------------------");
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(Paths.YelpDataPath);
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = true,
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 1, UseAllLengths = false, Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf },
            };
            for (float i = 0.005f; i <= 0.006; i += 0.005f)
            {
                var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",
                    Shuffle = false,
                    NumberOfThreads = 1,
                    //L1Regularization = 0f,
                    ConvergenceTolerance = i,
                    //BiasLearningRate = i,
                    //MaximumNumberOfIterations = 3000
                    //L2Regularization = i
                };
                Console.WriteLine($"Hyper parameter = {i}");
                var model = mlHelper.TrainAndEvaluateSdca(dataView, null, featureOptions, sdcaOptions, crossValidation: true);
                Console.WriteLine("-------------------------------------------");
                Console.WriteLine();
            }
        }

        private static void ImdbDatasetSDCA(MachineLearningHelper mlHelper)
        {
            Console.WriteLine("------------IMDB LARGE DATASET-------------------");
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 1, UseAllLengths = false, Weighting = NgramExtractingEstimator.WeightingCriteria.Idf },
                KeepPunctuations = false
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },

            };
            IDataView dataViewLarge = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kDataPath, hasHeader: true);
            for (float i = 0.005f; i <= 0.06f; i += 0.005f)
            {
                var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",
                    Shuffle = false,
                    NumberOfThreads = 1,
                    //L1Regularization = 0f,
                    ConvergenceTolerance = i,
                    //BiasLearningRate = i,
                    //MaximumNumberOfIterations = 3000
                    //L2Regularization = i
                };
                Console.WriteLine($"Hyper parameter = {i}");
                mlHelper.TrainAndEvaluateSdca(dataViewLarge, null, featureOptions, sdcaOptions, crossValidation: true);
                //mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLarge, null, featureOptions, crossValidation: true);
                Console.WriteLine("-------------------------------------------");
                Console.WriteLine();
            }
        }

        private static void AmazonDatasetSDCA(MachineLearningHelper mlHelper)
        {
            Console.WriteLine("------------Amazon DATASET-------------------");
            IDataView trainDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon1MTrainPath, hasHeader: true);
            IDataView testDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon400KTestPath, hasHeader: true);
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = false,
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 1, UseAllLengths = false, Weighting = NgramExtractingEstimator.WeightingCriteria.Idf }
            };
            for (float i = 0.01f; i <= 0.15f; i += 0.005f)
            {
                var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",
                    Shuffle = false,
                    NumberOfThreads = 1,
                    ConvergenceTolerance = i
                };
                Console.WriteLine($"Hyper parameter = {i}");
                var model = mlHelper.TrainAndEvaluateSdca(trainDataView, null, featureOptions, sdcaOptions, crossValidation: false, testData: testDataView);
                Console.WriteLine("-------------------------------------------");
                Console.WriteLine();
            }
        }

        private static void SmallDatasetAp(MachineLearningHelper mlHelper)
        {
            Console.WriteLine("------------YELP DATASET-------------------");
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(Paths.YelpDataPath);
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = true,
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 1, UseAllLengths = false, Weighting = NgramExtractingEstimator.WeightingCriteria.Idf },
            };

            using (var fs = new FileStream("data/Ex3ReportYelpExploss.csv", FileMode.OpenOrCreate, FileAccess.Write))
            using (var writer = new StreamWriter(fs))
            {
                writer.WriteLine("index,lossFunction,learningRate,NOiter,accuracy");
                int index = 0;

                foreach (float learningRate in new[] { 1f, 0.5f, 0.2f, 0.1f, 0.05f })
                {
                    foreach (int iterationsCount in new[] { 1, 10, 50, 100, 200, 400 })
                    {
                        var apOptions = new AveragedPerceptronTrainer.Options
                        {
                            LossFunction = new ExpLoss(),
                            LearningRate = learningRate,
                            //LazyUpdate = false,
                            //RecencyGain = 0.1f,
                            DecreaseLearningRate = false,
                            NumberOfIterations = iterationsCount,
                            L2Regularization = 0,
                            LabelColumnName = "Label",
                            FeatureColumnName = "Features"
                        };
                        Console.WriteLine($"Loss function = Exp loss");
                        Console.WriteLine($"Learning rate = {learningRate}");
                        Console.WriteLine($"Number of iterations = {iterationsCount}");
                        double? score = 0;
                        try
                        {
                            mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, apOptions,
                                ref score, crossValidation: true);
                        }
                        catch (InvalidOperationException e)
                        {
                            Console.WriteLine(e);
                        }
                        Console.WriteLine("-------------------------------------------");
                        Console.WriteLine();
                        writer.WriteLine($"{index},Exp loss,{learningRate},{iterationsCount},{score:0.##}");
                        writer.Flush(); ;
                        index++;
                    }
                }
            }
        }

        private static void ImdbDatasetAp(MachineLearningHelper mlHelper)
        {
            Console.WriteLine("------------IMDB 50k DATASET-------------------");
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kDataPath, hasHeader: true);
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English },
                KeepPunctuations = false,
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 1, UseAllLengths = false, Weighting = NgramExtractingEstimator.WeightingCriteria.Idf },
            };

            using (var fs = new FileStream("data/Ex3ReportImdbExploss.csv", FileMode.OpenOrCreate, FileAccess.Write))
            using (var writer = new StreamWriter(fs))
            {
                writer.WriteLine("index,lossFunction,learningRate,NOiter,accuracy");
                int index = 0;

                foreach (float learningRate in new[] { 1f, 0.5f, 0.2f, 0.1f, 0.05f }) // full test 1f, 0.5f, 0.2f, 0.1f, 0.05f
                {
                    foreach (int iterationsCount in new[] { 1, 10, 50, 100, 200, 400 })
                    {
                        var sw = new Stopwatch();

                        sw.Start();
                        var apOptions = new AveragedPerceptronTrainer.Options
                        {
                            LossFunction = new ExpLoss(),
                            LearningRate = learningRate,
                            //LazyUpdate = false,
                            //RecencyGain = 0.1f,
                            DecreaseLearningRate = false,
                            NumberOfIterations = iterationsCount,
                            L2Regularization = 0,
                            LabelColumnName = "Label",
                            FeatureColumnName = "Features"
                        };
                        Console.WriteLine($"Loss function = Exp loss");
                        Console.WriteLine($"Learning rate = {learningRate}");
                        Console.WriteLine($"Number of iterations = {iterationsCount}");
                        double? score = 0;
                        try
                        {
                            mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, featureOptions, apOptions,
                                ref score, crossValidation: true);
                        }
                        catch (InvalidOperationException e)
                        {
                            Console.WriteLine(e);
                        }
                        sw.Stop();
                        Console.WriteLine("Elapsed={0}",sw.Elapsed);
                        Console.WriteLine("-------------------------------------------");
                        Console.WriteLine();
                        writer.WriteLine($"{index},Exp loss,{learningRate},{iterationsCount},{score:0.##}");
                        writer.Flush(); ;
                        index++;
                    }
                }
            }
        }

        private static void AmazonDatasetAp(MachineLearningHelper mlHelper)
        {
            Console.WriteLine("------------Amazon DATASET-------------------");
            IDataView trainDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon1MTrainPath, hasHeader: true);
            IDataView testDataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon400KTestPath, hasHeader: true);
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepPunctuations = false,
                WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = false, Weighting = NgramExtractingEstimator.WeightingCriteria.Tf },
            };

            using (var fs = new FileStream("data/Ex3ReportAmazonHingeLoss3.csv", FileMode.OpenOrCreate, FileAccess.Write))
            using (var writer = new StreamWriter(fs))
            {
                writer.WriteLine("index,lossFunction,learningRate,NOiter,accuracy");
                int index = 0;

                foreach (float learningRate in new[] { 1f, 0.5f, 0.2f, 0.1f, 0.05f}) // full test 1f, 0.5f, 0.2f, 0.1f, 0.05f
                {
                    foreach (int iterationsCount in new[] { 1, 10, 50, 100, 200, 400 })
                    {
                        var sw = new Stopwatch();

                        sw.Start();
                        var apOptions = new AveragedPerceptronTrainer.Options
                        {
                            LossFunction = new HingeLoss(),
                            LearningRate = learningRate,
                            DecreaseLearningRate = false,
                            NumberOfIterations = iterationsCount,
                            L2Regularization = 0,
                            LabelColumnName = "Label",
                            FeatureColumnName = "Features"
                        };
                        Console.WriteLine($"Loss function = Hinge loss");
                        Console.WriteLine($"Learning rate = {learningRate}");
                        Console.WriteLine($"Number of iterations = {iterationsCount}");
                        double? score = 0;
                        try
                        {
                            mlHelper.TrainAndEvaluateAveragedPerceptron(trainDataView, null, featureOptions, apOptions,
                                ref score, crossValidation: false, testData: testDataView);
                        }
                        catch (InvalidOperationException e)
                        {
                            Console.WriteLine(e);
                        }
                        sw.Stop();
                        Console.WriteLine("Elapsed={0}",sw.Elapsed);
                        Console.WriteLine("-------------------------------------------");
                        Console.WriteLine();
                        writer.WriteLine($"{index},Hinge loss,{learningRate},{iterationsCount},{score:0.##}");
                        writer.Flush(); ;
                        index++;
                    }
                }
            }
        }
    }
}
