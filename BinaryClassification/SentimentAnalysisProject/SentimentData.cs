using Microsoft.ML.Data;

namespace SentimentAnalysisProject
{
    public class SentimentData : ISentimentData
    {
        [LoadColumn(0)] public string SentimentText;
        [LoadColumn(1)] public bool Label;

        string ISentimentData.SentimentText { get => SentimentText; set => SentimentText = value; }
    }
}
