using Microsoft.ML.Data;

namespace SentimentAnalysisProject
{
    public class SentimentData
    {
        [LoadColumn(0)] public string SentimentText;
        [LoadColumn(1)] public bool Label;
    }
}
