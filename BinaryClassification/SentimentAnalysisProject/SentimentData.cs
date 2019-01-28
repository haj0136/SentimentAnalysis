using Microsoft.ML.Data;

namespace SentimentAnalysisProject
{
    public class SentimentData
    {
        [Column(ordinal: "0", name: "Label")] public float Sentiment;
        [Column(ordinal: "1")] public string SentimentText;
        public float Probability;
    }
}
