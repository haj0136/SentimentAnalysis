using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SentimentAnalysisProject
{
    public class MachineLearningHelper
    {
        public MLContext MlContext { get; }
        private readonly bool _randomTry;
        private readonly bool _removeStopWords;

        /// <summary>
        /// If seed is null, algorithm will work with random chance and results will be different.
        /// </summary>
        /// <param name="seed"></param>
        /// <param name="removeStopWords"></param>
        public MachineLearningHelper(int? seed = null, bool removeStopWords = false)
        {
            MlContext = new MLContext(seed);
            if (seed == null)
            {
                _randomTry = true;
            }
            _removeStopWords = removeStopWords;
        }

        public IDataView LoadData(string path)
        {
            return MlContext.Data.LoadFromTextFile<SentimentData>(path);
        }

        public ITransformer TrainAndEvaluateSdca(IDataView data, string modelPath, bool crossValidation = false)
        {
            var splitData = MlContext.Data.TrainTestSplit(data, 0.2);
            var trainSet = splitData.TrainSet;
            var testSet = splitData.TestSet;

            var featureOptions = new TextFeaturizingEstimator.Options
            {
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true },
                CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },
            };
            if (_removeStopWords)
            {
                featureOptions.StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English };
            }

            var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                Shuffle = false,
                NumberOfThreads = 1,
            };

            var pipeline = MlContext.Transforms.Text.FeaturizeText("Features", options: featureOptions, nameof(SentimentData.SentimentText));
            var trainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            if (!_randomTry)
            {
                trainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(sdcaOptions);
            }
            var trainingPipeline = pipeline.Append(trainer);

            ITransformer model = null;

            if (crossValidation)
            {
                Console.WriteLine("Cross validation");
                var cvMetrics = MlContext.BinaryClassification.CrossValidate(data, trainingPipeline, numberOfFolds: 5);
                Console.WriteLine("End of Cross validation");
                Console.WriteLine();
                Console.WriteLine("Evaluating Model accuracy with Test data");

                Console.WriteLine();
                foreach (var cvr in cvMetrics)
                {
                    Utils.PrintBinaryClassificationMetrics(trainer.ToString(), cvr.Metrics);
                }

                model = cvMetrics.Aggregate((m1, m2) => m1.Metrics.F1Score > m2.Metrics.F1Score ? m1 : m2).Model;
                var accuracies = cvMetrics.Select(r => r.Metrics.Accuracy);
                Console.WriteLine(accuracies.Average());
                Console.WriteLine("End of model evaluation");
            }
            else
            {
                Console.WriteLine("Create and TrainAndEvaluateSdca the Model");
                model = trainingPipeline.Fit(trainSet);

                Console.WriteLine("End of training");
                Console.WriteLine();

                Console.WriteLine("Evaluating Model accuracy with Test data");
                var predictions = model.Transform(testSet);
                var metrics = MlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

                Console.WriteLine();
                Utils.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                Console.WriteLine("End of model evaluation");
            }

            MlContext.Model.Save(model, trainSet.Schema, modelPath);
            return model;
        }

        public ITransformer TrainAndEvaluateAveragedPerceptron(IDataView data, string modelPath, bool crossValidation = false)
        {
            var splitData = MlContext.Data.TrainTestSplit(data, 0.2);
            var trainSet = splitData.TrainSet;
            var testSet = splitData.TestSet;

            var featureOptions = new TextFeaturizingEstimator.Options
            {
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true },
                CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },
            };
            if (_removeStopWords)
            {
                featureOptions.StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English };
            }

            var pipeline = MlContext.Transforms.Text.FeaturizeText("Features", options:featureOptions, nameof(SentimentData.SentimentText));
            var options = new AveragedPerceptronTrainer.Options
            {
                LossFunction = new ExpLoss(),
                LearningRate = 0.1f,
                LazyUpdate = false,
                RecencyGain = 0.1f,
                NumberOfIterations = 10,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            };
            var trainer = MlContext.BinaryClassification.Trainers.AveragedPerceptron(options);
            var trainingPipeline = pipeline.Append(trainer);

            ITransformer model = null;

            if (crossValidation)
            {
                Console.WriteLine("Cross validation");
                var cvMetrics = MlContext.BinaryClassification.CrossValidateNonCalibrated(data, trainingPipeline, numberOfFolds: 5);
                Console.WriteLine("End of Cross validation");
                Console.WriteLine();
                Console.WriteLine("Evaluating Model accuracy with Test data");

                Console.WriteLine();
                foreach (var cvr in cvMetrics)
                {
                    Utils.PrintBinaryClassificationMetrics(trainer.ToString(), cvr.Metrics);
                }

                model = cvMetrics.Aggregate((m1, m2) => m1.Metrics.F1Score > m2.Metrics.F1Score ? m1 : m2).Model;
                var accuracies = cvMetrics.Select(r => r.Metrics.Accuracy);
                Console.WriteLine($"Average accuracy: {(accuracies.Average() * 100):F}%");
                Console.WriteLine("End of model evaluation");
            }
            else
            {
                Console.WriteLine("Train and Evaluate the Model");
                model = trainingPipeline.Fit(trainSet);

                Console.WriteLine("End of training");
                Console.WriteLine();

                Console.WriteLine("Evaluating Model accuracy with Test data");
                var predictions = model.Transform(testSet);
                var metrics = MlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

                Console.WriteLine();
                Utils.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                Console.WriteLine("End of model evaluation");
            }

            MlContext.Model.Save(model, trainSet.Schema, modelPath);
            return model;
        }


        public void Predict(ITransformer model, string text)
        {
            var predictionEngine = MlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var sampleStatement = new SentimentData
            {
                SentimentText = text
            };

            var resultPrediction = predictionEngine.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("Prediction Test of model with a single sample and test data-set");
            Console.WriteLine();
            Console.WriteLine($"Text: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Negative" : "Positive")} | Probability of being toxic: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public void PredictWithModelLoadedFromFile(IDataView testData, string modelPath)
        {
            ITransformer loadedModel;
            DataViewSchema dataViewSchema;
            using (var fs = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = MlContext.Model.Load(fs, out dataViewSchema);
            }

            // Create prediction engine
            //var sentimentStreamingDataView = MlContext.CreateStreamingDataView(sentiments);

            //IDataView sentimentStreamingDataView = MlContext.Data.LoadFromTextFile<SentimentData>(_testDataPath);

            var predictions = loadedModel.Transform(testData);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            Console.WriteLine("====Prediction Test of loaded model with a multiple samples====");
            Console.WriteLine();

            Utils.ShowDataViewInConsole(MlContext, predictions, numberOfRows: 4);

            Console.WriteLine("=============== End of predictions ===============");
        }

        public void PredictWithModelLoadedFromFile(IEnumerable<SentimentData> sentiments, string modelPath)
        {
            ITransformer loadedModel;
            DataViewSchema dataViewSchema;
            using (var fs = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = MlContext.Model.Load(fs, out dataViewSchema);
            }

            IDataView batchComments = MlContext.Data.LoadFromEnumerable(sentiments);

            var predictions = loadedModel.Transform(batchComments);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            Console.WriteLine("====Prediction Test of loaded model with a multiple samples====");
            Console.WriteLine();

            IEnumerable<SentimentPrediction> predictedResults = MlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            var results = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));

            foreach (var result  in results)
{
            Console.WriteLine($"Sentiment: {result.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(result.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {result.prediction.Probability} ");

}

            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
