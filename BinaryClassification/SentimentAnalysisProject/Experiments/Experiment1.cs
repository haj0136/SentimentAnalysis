using System;
using Microsoft.ML;

namespace SentimentAnalysisProject.Experiments
{
    public class Experiment1
    {
        public static void StartExperiment()
        {
            var mlHelper = new MachineLearningHelper(seed:1, removeStopWords:false);
            Console.WriteLine("------------YELP DATASET-------------------");
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(Paths.YelpDataPath);
            var model = mlHelper.TrainAndEvaluateSdca(dataView, null, crossValidation: true);
            var model2 = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, crossValidation: true);

            Console.WriteLine("------------IMDB LARGE DATASET 25k-------------------");
            IDataView dataViewLarge = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb25kDataPath, hasHeader:true);
            mlHelper.TrainAndEvaluateSdca(dataViewLarge, null, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLarge, null, crossValidation: true);

            Console.WriteLine("------------IMDB LARGE DATASET 50k-------------------");
            dataViewLarge = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kDataPath, hasHeader:true);
            mlHelper.TrainAndEvaluateSdca(dataViewLarge, null, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLarge, null, crossValidation: true);
        }
    }
}
