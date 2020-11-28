using Microsoft.ML.Data;

namespace SentimentAnalysisProject
{
    public class SentimentDataAmazonLarge : ISentimentData
    {
        [LoadColumn(1)] public string SentimentText;
        [LoadColumn(0)] public bool Label;

        string ISentimentData.SentimentText {  get => SentimentText; set => SentimentText = value; }
    }
}
