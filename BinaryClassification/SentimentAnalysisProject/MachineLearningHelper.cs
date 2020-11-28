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
        /// Constant for cross validation
        /// </summary>
        private const int NumberOfFolds = 5;

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

        public ITransformer TrainAndEvaluateSdca(IDataView data, string modelPath, bool crossValidation = false, bool crossValidationDetailOutput = false, IDataView testData = null)
        {
            var featureOptions = new TextFeaturizingEstimator.Options
            {
                //WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true},
                //CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 5, UseAllLengths = false },
            };

            if (_removeStopWords)
            {
                featureOptions.StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English };
            }
            return TrainAndEvaluateSdca(data, modelPath, featureOptions, crossValidation, crossValidationDetailOutput, testData);
        }

        public ITransformer TrainAndEvaluateSdca(IDataView data, string modelPath, TextFeaturizingEstimator.Options featureOptions, bool crossValidation = false, bool crossValidationDetailOutput = false, IDataView testData = null)
        {
            var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                Shuffle = false,
                NumberOfThreads = 1,
            };
            if (_randomTry)
            {
                sdcaOptions = null;
            }

            return TrainAndEvaluateSdca(data, modelPath, featureOptions, sdcaOptions, crossValidation, crossValidationDetailOutput, testData);
        }

        /// <summary>
        /// This method won't remove stop words by default, even if it was set by constructor
        /// </summary>
        /// <param name="data"></param>
        /// <param name="modelPath"></param>
        /// <param name="featureOptions"></param>
        /// <param name="sdcaOptions"></param>
        /// <param name="crossValidation"></param>
        /// <param name="crossValidationDetailOutput"></param>
        /// <param name="testData"></param>
        /// <returns></returns>
        public ITransformer TrainAndEvaluateSdca(IDataView data, string modelPath, TextFeaturizingEstimator.Options featureOptions, SdcaLogisticRegressionBinaryTrainer.Options sdcaOptions, bool crossValidation = false, bool crossValidationDetailOutput = false, IDataView testData = null)
        {
            IDataView trainSet;
            IDataView testSet;

            if (testData == null)
            {
                var splitData = MlContext.Data.TrainTestSplit(data, 0.2);
                trainSet = splitData.TrainSet;
                testSet = splitData.TestSet;
            }
            else
            {
                trainSet = data;
                testSet = testData;
            }


            var pipeline = MlContext.Transforms.Text.FeaturizeText("Features", options: featureOptions, nameof(SentimentData.SentimentText));

            SdcaLogisticRegressionBinaryTrainer trainer;
            if (sdcaOptions != null)
            {
                trainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(sdcaOptions);
            }
            else
            {
                trainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            }
            var trainingPipeline = pipeline.Append(trainer);

            ITransformer model = null;

            if (crossValidation)
            {
                Console.WriteLine("Cross validation");
                var cvMetrics = MlContext.BinaryClassification.CrossValidate(data, trainingPipeline, NumberOfFolds);
                Console.WriteLine("End of Cross validation");
                Console.WriteLine("Evaluating Model accuracy with Test data");

                Console.WriteLine();
                if (crossValidationDetailOutput)
                {
                    foreach (var cvr in cvMetrics)
                    {
                        Utils.PrintBinaryClassificationMetrics(trainer.ToString(), cvr.Metrics);
                    }
                }

                model = cvMetrics.Aggregate((m1, m2) => m1.Metrics.F1Score > m2.Metrics.F1Score ? m1 : m2).Model;
                var accuracies = cvMetrics.Select(r => r.Metrics.Accuracy);
                Console.WriteLine($"SDCA Average accuracy: {(accuracies.Average() * 100):F}%");
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

            if (modelPath != null)
            {
                MlContext.Model.Save(model, trainSet.Schema, modelPath);
                Console.WriteLine("Model saved");
            }
            return model;
        }

        public ITransformer TrainAndEvaluateAveragedPerceptron(IDataView data, string modelPath, bool crossValidation = false, bool crossValidationDetailOutput = false, IDataView testData = null)
        {
            var featureOptions = new TextFeaturizingEstimator.Options
            {
            };
            if (_removeStopWords)
            {
                featureOptions.StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = TextFeaturizingEstimator.Language.English };
            }
            return TrainAndEvaluateAveragedPerceptron(data, modelPath, featureOptions, crossValidation, crossValidationDetailOutput, testData);
        }

        public ITransformer TrainAndEvaluateAveragedPerceptron(IDataView data, string modelPath, TextFeaturizingEstimator.Options featureOptions, bool crossValidation = false, bool crossValidationDetailOutput = false, IDataView testData = null)
        {
            double? a = null;
            return TrainAndEvaluateAveragedPerceptron(data, modelPath, featureOptions, null, ref a, crossValidation, crossValidationDetailOutput, testData);
        }

        /// <summary>
        /// This method won't remove stop words by default, even if it was set by class constructor
        /// </summary>
        /// <param name="data"></param>
        /// <param name="modelPath"></param>
        /// <param name="featureOptions"></param>
        /// <param name="options"></param>
        /// <param name="accuracy">REF parameter to get model accuracy, pass null if no score is needed</param>
        /// <param name="crossValidation"></param>
        /// <param name="crossValidationDetailOutput"></param>
        /// <param name="testData"></param>
        /// <returns></returns>
        public ITransformer TrainAndEvaluateAveragedPerceptron(IDataView data, string modelPath, TextFeaturizingEstimator.Options featureOptions, AveragedPerceptronTrainer.Options options,  ref double? accuracy, bool crossValidation = false, bool crossValidationDetailOutput = false, IDataView testData = null)
        {
            IDataView trainSet;
            IDataView testSet;
            if (testData == null)
            {
                var splitData = MlContext.Data.TrainTestSplit(data, 0.2);
                trainSet = splitData.TrainSet;
                testSet = splitData.TestSet;
            }
            else
            {
                trainSet = data;
                testSet = testData;
            }

            var pipeline = MlContext.Transforms.Text.FeaturizeText("Features", options: featureOptions, nameof(SentimentData.SentimentText));

            AveragedPerceptronTrainer trainer = null;

            if (options != null)
            {
                trainer = MlContext.BinaryClassification.Trainers.AveragedPerceptron(options);
            } else
            {
                trainer = MlContext.BinaryClassification.Trainers.AveragedPerceptron();
            }
            var trainingPipeline = pipeline.Append(trainer);

            ITransformer model = null;

            if (crossValidation)
            {
                Console.WriteLine("Cross validation");
                var cvMetrics = MlContext.BinaryClassification.CrossValidateNonCalibrated(data, trainingPipeline, numberOfFolds: NumberOfFolds);
                Console.WriteLine("End of Cross validation");
                Console.WriteLine();
                Console.WriteLine("Evaluating Model accuracy with Test data");

                if (crossValidationDetailOutput)
                {
                    Console.WriteLine();
                    foreach (var cvr in cvMetrics)
                    {
                        Utils.PrintBinaryClassificationMetrics(trainer.ToString(), cvr.Metrics);
                    }
                }

                model = cvMetrics.Aggregate((m1, m2) => m1.Metrics.F1Score > m2.Metrics.F1Score ? m1 : m2).Model;
                var accuracies = cvMetrics.Select(r => r.Metrics.Accuracy);
                Console.WriteLine($"Averaged perceptron Average accuracy: {(accuracies.Average() * 100):F}%");
                if (accuracy != null)
                {
                    accuracy = accuracies.Average() * 100;
                }
                Console.WriteLine("End of model evaluation");
            }
            else
            {
                Console.WriteLine("Train and Evaluate the Model AP");
                model = trainingPipeline.Fit(trainSet);

                Console.WriteLine("End of training");
                Console.WriteLine();

                Console.WriteLine("Evaluating Model accuracy with Test data");
                var predictions = model.Transform(testSet);
                var metrics = MlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

                if (accuracy != null)
                {
                    accuracy = metrics.Accuracy * 100;
                }

                Console.WriteLine();
                Utils.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                Console.WriteLine("End of model evaluation");
                Console.WriteLine();
            }

            if (modelPath != null)
            {
                MlContext.Model.Save(model, trainSet.Schema, modelPath);
                Console.WriteLine("Model saved");
            }
            return model;
        }


        public void Predict<TSentimentData>(ITransformer model, string text) where TSentimentData : class, ISentimentData, new()
        {
            var predictionEngine = MlContext.Model.CreatePredictionEngine<TSentimentData, SentimentPrediction>(model);

            var sampleStatement = new TSentimentData
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

            foreach (var (sentiment, prediction) in results)
            {
                Console.WriteLine($"Sentiment: {sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
