using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SentimentAnalysisProject
{
    class Program
    {
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv"); // label index 0 , sentimentText index 1
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        private static readonly string _imdbDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labelled.txt"); // label index 1, sentimentText index 0
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        private static TextLoader _textLoader;

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Bool, 1),
                        new TextLoader.Column("SentimentText", DataKind.Text, 0),
                        new TextLoader.Column("Probability", DataKind.R4, 2), 
                    }
            }
            );

            IDataView dataView = _textLoader.Read(_imdbDataPath);
            var data = mlContext.Clustering.TrainTestSplit(dataView, 0.2);

            var model = Train(mlContext, data.trainSet);

            Evaluate(mlContext, model, data.testSet);
            Predict(mlContext, model, "This is a very rude movie");
            //PredictWithModelLoadedFromFile(mlContext);

        } // Main

        public static ITransformer Train(MLContext mlContext, IDataView dataView)
        {
            //IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(learningRate:0.5, numTrees:108));


            Console.WriteLine("Create and Train the Model");
            var model = pipeline.Fit(dataView);

            Console.WriteLine("End of training");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView dataView)
        {
            //IDataView dataView = _textLoader.Read(_testDataPath);
            Console.WriteLine("Evaluating Model accuracy with Test data");
            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("End of model evaluation");

            //var preview = predictions.Preview();
            //foreach (var row in preview.RowView.Take(20))
            //{
            //    foreach (var kv in row.Values)
            //    {
            //        Console.WriteLine(kv.Key + ": " + kv.Value);
            //    }

            //    Console.WriteLine();
            //}

            SaveModelAsFile(mlContext, model);
        }

        public static void Predict(MLContext mlContext, ITransformer model, string text)
        {
            var predictionEngine = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);

            var sampleStatement = new SentimentData
            {
                SentimentText = text
            };

            var resultPrediction = predictionEngine.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("Prediction Test of model with a single sample and test data-set");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(!Convert.ToBoolean(resultPrediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }
        }

        public static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Best movie in DC Extended Universe so far...great acting performances, great visual effects, great screenplay and a great score....in a word: FANTASTIC...."
                    //SentimentText = "This is a very rude movie"
                },
                new SentimentData
                {
                    SentimentText = "He is the best."
                }
            };

            ITransformer loadedModel;
            using (var fs = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(fs);
            }

            // Create prediction engine
            //var sentimentStreamingDataView = mlContext.CreateStreamingDataView(sentiments);

            IDataView sentimentStreamingDataView = _textLoader.Read(_testDataPath);

            var predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            var predictedResults = predictions.AsEnumerable<SentimentPrediction>(mlContext, reuseRowObject: false);


            var stringSentiments = predictions.GetColumn<string>(mlContext, "SentimentText");

            Console.WriteLine();

            Console.WriteLine("====Prediction Test of loaded model with a multiple samples====");
            Console.WriteLine();


            var sentimentsAndPredictions =
                stringSentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
                Console.WriteLine();
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
