using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;

namespace SentimentAnalysisProject
{
    class Program
    {
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv"); // label index 0 , sentimentText index 1
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        private static readonly string _imdbDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labeled.txt"); // label index 1, sentimentText index 0
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_imdbDataPath);
            var data = mlContext.Data.TrainTestSplit(dataView, 0.2);

            var model = TrainAndEvaluate(mlContext, data.TrainSet, data.TestSet);

            //Predict(mlContext, model, "This is a very rude movie");
            PredictWithModelLoadedFromFile(mlContext, data.TestSet);

        } // Main

        public static ITransformer TrainAndEvaluate(MLContext mlContext, IDataView trainSet, IDataView testSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", inputColumnName: nameof(SentimentData.SentimentText));
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = pipeline.Append(trainer);


            Console.WriteLine("Create and TrainAndEvaluate the Model");
            var model = trainingPipeline.Fit(trainSet);

            Console.WriteLine("End of training");
            Console.WriteLine();

            //IDataView trainSet = _textLoader.Read(_testDataPath);
            Console.WriteLine("Evaluating Model accuracy with Test data");
            var predictions = model.Transform(testSet);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine();
            Utils.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
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

            mlContext.Model.Save(model, trainSet.Schema, _modelPath);
            return model;
        }

        public static void Predict(MLContext mlContext, ITransformer model, string text)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var sampleStatement = new SentimentData
            {
                SentimentText = text
            };

            var resultPrediction = predictionEngine.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("Prediction Test of model with a single sample and test data-set");
            Console.WriteLine();
            Console.WriteLine($"Text: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability of being toxic: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void PredictWithModelLoadedFromFile(MLContext mlContext, IDataView testData)
        {
            ITransformer loadedModel;
            DataViewSchema dataViewSchema;
            using (var fs = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(fs, out dataViewSchema);
            }

            // Create prediction engine
            //var sentimentStreamingDataView = mlContext.CreateStreamingDataView(sentiments);

            //IDataView sentimentStreamingDataView = mlContext.Data.LoadFromTextFile<SentimentData>(_testDataPath);

            var predictions = loadedModel.Transform(testData);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            Console.WriteLine("====Prediction Test of loaded model with a multiple samples====");
            Console.WriteLine();

            Utils.ShowDataViewInConsole(mlContext, predictions, numberOfRows: 4);

            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
