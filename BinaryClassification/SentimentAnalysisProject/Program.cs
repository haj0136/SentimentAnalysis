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
        private static readonly string _imdbDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_labeled.txt"); // label index 1, sentimentText index 0
        private static readonly string _sdcaModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "SdcaModel.zip");
        private static readonly string _ApModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "ApModel.zip");

        static void Main(string[] args)
        {
            var mlHelper = new MachineLearningHelper(1);

            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(_imdbDataPath);
            Utils.ShowDataViewInConsole(mlHelper.MlContext ,dataView);
            var model = mlHelper.TrainAndEvaluateSdca(dataView, _sdcaModelPath, crossValidation: false);
            //var model = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, _ApModelPath, crossValidation: false);

            mlHelper.Predict(model, "This is a very rude movie");
            mlHelper.Predict(model, "I like this movie");
            //mlHelper.PredictWithModelLoadedFromFile(data.TestSet);

            /*
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This is a very rude movie"
                },
                new SentimentData
                {
                    SentimentText = "I love this movie."
                }
            };
            mlHelper.PredictWithModelLoadedFromFile(sentiments,_sdcaModelPath);
            */
        }
    }
}
