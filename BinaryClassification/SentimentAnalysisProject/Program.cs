using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.CodeDom;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;

namespace SentimentAnalysisProject
{
    class Program
    {
        private static readonly string ImdbDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labeled.txt"); // label index 1, sentimentText index 0
        private static readonly string YelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt"); // label index 1, sentimentText index 0
        private static readonly string ImdbLargeDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "labeledTrainData.tsv"); // label index 1, sentimentText index 2
        private static readonly string SdcaModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "SdcaModel.zip");
        private static readonly string ApModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "ApModel.zip");

        static void Main(string[] args)
        {
            SmallDataset();
            //LargeDataset();
        }

        private static void SmallDataset()
        {
            var mlHelper = new MachineLearningHelper(seed:1, removeStopWords:true);

            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpDataPath);
            var test = mlHelper.MlContext.Data.CreateEnumerable<SentimentData>(dataView, false).ToList();
            Console.WriteLine($"Data size is: {test.Count} rows");
            Utils.ShowDataViewInConsole(mlHelper.MlContext ,dataView);
            var model = mlHelper.TrainAndEvaluateSdca(dataView, SdcaModelPath, crossValidation: false);

            //var model = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, ApModelPath, crossValidation: false);

            //mlHelper.Predict(model, "This is a very rude movie");
            //mlHelper.Predict(model, "I like this movie");
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
            mlHelper.PredictWithModelLoadedFromFile(sentiments,SdcaModelPath);
        }

        private static void LargeDataset()
        {
            var mlHelper = new MachineLearningHelper(seed:1, removeStopWords:false);

            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(ImdbLargeDataPath, hasHeader:true);
            var test = mlHelper.MlContext.Data.CreateEnumerable<SentimentDataImdbLarge>(dataView, false).ToList();
            Console.WriteLine($"Data size is: {test.Count} rows");
            Utils.ShowDataViewInConsole(mlHelper.MlContext ,dataView);
            //var model = mlHelper.TrainAndEvaluateSdca(dataView, SdcaModelPath, crossValidation: false);

            var model = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, ApModelPath, crossValidation: false);

            mlHelper.Predict(model, "This is a very rude movie");
            mlHelper.Predict(model, "I like this movie");
            //mlHelper.PredictWithModelLoadedFromFile(data.TestSet);


        }
    }
}
