using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace SentimentAnalysisProject.Experiments
{
    public class Experiment1
    {
        private static readonly string YelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt"); // label index 1, sentimentText index 0
        private static readonly string ImdbLargeDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "labeledTrainData.tsv"); // label index 1, sentimentText index 2

        public static void StartExperiment()
        {
            var mlHelper = new MachineLearningHelper(seed:1, removeStopWords:false);
            Console.WriteLine("------------YELP DATASET-------------------");
            IDataView dataView = mlHelper.MlContext.Data.LoadFromTextFile<SentimentData>(YelpDataPath);
            var model = mlHelper.TrainAndEvaluateSdca(dataView, null, crossValidation: true);
            var model2 = mlHelper.TrainAndEvaluateAveragedPerceptron(dataView, null, crossValidation: true);

            Console.WriteLine("------------IMDB LARGE DATASET-------------------");
            IDataView dataViewLarge = mlHelper.MlContext.Data.LoadFromTextFile<SentimentDataImdbLarge>(ImdbLargeDataPath, hasHeader:true);
            mlHelper.TrainAndEvaluateSdca(dataViewLarge, null, crossValidation: true);
            mlHelper.TrainAndEvaluateAveragedPerceptron(dataViewLarge, null, crossValidation: true);
        }
    }
}
