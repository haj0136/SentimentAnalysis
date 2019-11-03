using Microsoft.ML.Data;

namespace SentimentAnalysisProject
{
    public class SentimentDataImdbLarge : ISentimentData
    {
        [LoadColumn(2)] public string SentimentText;
        [LoadColumn(1)] public bool Label;

        string ISentimentData.SentimentText {  get => SentimentText; set => SentimentText = value; }
    }
}
