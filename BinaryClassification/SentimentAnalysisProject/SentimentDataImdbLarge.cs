using Microsoft.ML.Data;

namespace SentimentAnalysisProject
{
    public class SentimentDataImdbLarge
    {
        [LoadColumn(2)] public string SentimentText;
        [LoadColumn(1)] public bool Label;
    }
}
