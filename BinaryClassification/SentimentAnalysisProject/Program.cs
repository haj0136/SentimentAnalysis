using LemmaSharp;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SentimentAnalysisProject.Experiments;
using System.Diagnostics;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysisProject
{
    class Program
    {
        private static readonly string ImdbDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labeled.txt"); // label index 1, sentimentText index 0
        private static readonly string YelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt"); // label index 1, sentimentText index 0
        private static readonly string SdcaModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "SdcaModel.zip");
        private static readonly string ApModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "ApModel.zip");

        static void Main(string[] args)
        {
            var sw = new Stopwatch();

            sw.Start();
            //Experiment1.StartExperiment();
            //Experiment2.StartExperiment();
            Experiment3.StartExperiment();
            //SmallDataset();
            //LargeDataset();

            //Lemmatization(Paths.Amazon1MTrainPath, "AmazonTrainLemmatized.tsv");
            //Lemmatization("Data/AmazonValidationSet100k.tsv", "AmazonValidationLemmatized.tsv");

            sw.Stop();

            Console.WriteLine("Elapsed={0}",sw.Elapsed);
        }

        private static void SmallDataset()
        {
            var mlHelper = new MachineLearningHelper(seed: 1, removeStopWords: false);

            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpDataPath);
            var test = mlHelper.MlContext.Data.CreateEnumerable<SentimentData>(dataView, false).ToList();
            Console.WriteLine($"Data size is: {test.Count} rows");
            Utils.ShowDataViewInConsole(mlHelper.MlContext, dataView);
            //var model = mlHelper.TrainAndEvaluateSdca(dataView, SdcaModelPath, crossValidation: false);
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                KeepDiacritics = true,
                CaseMode = TextNormalizingEstimator.CaseMode.Lower
            };

            var model = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, ApModelPath, featureOptions, crossValidation: true);

            //mlHelper.Predict<SentimentData>(model, "This is a very rude movie");
            //mlHelper.Predict<SentimentData>(model, "I like this movie");
            //mlHelper.PredictWithModelLoadedFromFile(data.TestSet);


            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };
            mlHelper.PredictWithModelLoadedFromFile(sentiments, SdcaModelPath);

        }

        private static void LargeDataset()
        {
            var mlHelper = new MachineLearningHelper(seed: 1, removeStopWords: false);

            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.ImdbLargeDataPath, hasHeader: true);
            var test = mlHelper.MlContext.Data.CreateEnumerable<SentimentDataImdbLarge>(dataView, false).ToList();
            Console.WriteLine($"Data size is: {test.Count} rows");
            Utils.ShowDataViewInConsole(mlHelper.MlContext, dataView);
            var model = mlHelper.TrainAndEvaluateSdca(dataView, SdcaModelPath, crossValidation: false);

            //var model = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, ApModelPath, crossValidation: false);

            //mlHelper.Predict<SentimentDataImdbLarge>(model, "This is a very rude movie");
            //mlHelper.Predict<SentimentDataImdbLarge>(model, "I like this movie");

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This is a very rude movie"
                },
                new SentimentData
                {
                    SentimentText = "I like this movie"
                }
            };
            mlHelper.PredictWithModelLoadedFromFile(sentiments, SdcaModelPath);


        }

        private static void Lemmatization(string pathToFile, string newFileName)
        {
            bool displayToConsole = false;
            string exampleSentence = "cats running ran cactus cactuses cacti community communities was";
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Example sentence lemmatized");
            Console.WriteLine("        WORD ==> LEMMA");
            ILemmatizer lmtz = new LemmatizerPrebuiltCompact(LemmaSharp.LanguagePrebuilt.English);

            using (StreamReader sr = File.OpenText(pathToFile))
            using (StreamWriter sw = File.CreateText(newFileName))
            {
                string s = string.Empty;
                int counter = 0;
                while ((s = sr.ReadLine()) != null)
                {
                    //if (counter > 20)
                    //    break;

                    string[] exampleWords = s.Split(
                        new char[] { ' ', ',', '.', ')', '(' }, StringSplitOptions.RemoveEmptyEntries);
                    var sb = new StringBuilder();
                    foreach (string word in exampleWords)
                    {
                        string lemma = LemmatizeOne(lmtz, word, displayToConsole);
                        sb.Append(lemma);
                        sb.Append(" ");
                    }

                    sb.Length -= 1;
                    if (displayToConsole)
                    {
                        Console.WriteLine("-----------------------------");
                        Console.WriteLine("Original");
                        Console.WriteLine(s);
                        Console.WriteLine("Lemmatized");
                        Console.WriteLine(sb.ToString());
                    }
                    sw.WriteLine(sb.ToString());
                    counter++;
                }
            }

            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Lemmatization done.");
        }

        private static string LemmatizeOne(LemmaSharp.ILemmatizer lmtz, string word, bool display)
        {
            string wordLower = word.ToLower();
            string lemma = lmtz.Lemmatize(wordLower);
            if (display)
            {
                Console.ForegroundColor = wordLower.Equals(lemma) ? ConsoleColor.White : ConsoleColor.Red;
                Console.WriteLine("{0,12} ==> {1}", word, lemma);
            }

            return lemma;
        }
    }
}
