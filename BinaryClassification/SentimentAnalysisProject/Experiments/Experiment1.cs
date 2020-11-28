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

            Console.WriteLine("------------IMDB Medium DATASET 50k-------------------");
            IDataView dataViewLarge = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(Paths.Imdb50kDataPath, hasHeader:true);
            mlHelper.TrainAndEvaluateSdca(dataViewLarge, null, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLarge, null, crossValidation: true);
            
            Console.WriteLine("------------Amazon LARGE DATASET 1,4M-------------------");
            dataViewLarge = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon1MTrainPath, hasHeader:true);
            IDataView testDataView =
                mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataAmazonLarge>(Paths.Amazon400KTestPath,
                    hasHeader: true);
            mlHelper.TrainAndEvaluateSdca(dataViewLarge, null, crossValidation: false, testData: testDataView);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLarge, null, crossValidation: false, testData:testDataView);
        }
    }
}
