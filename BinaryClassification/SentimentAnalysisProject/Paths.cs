using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentAnalysisProject
{
    public static class Paths
    {
        public static readonly string YelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt"); // label index 1, sentimentText index 0
        public static readonly string YelpLemmatizedDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelpLemmatized.txt"); // label index 1, sentimentText index 0
        public static readonly string ImdbLargeDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "labeledTrainData.tsv"); // label index 1, sentimentText index 2
        public static readonly string Imdb25kDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_25k.tsv"); // label index 1, sentimentText index 2
        public static readonly string Imdb25kLemmatizedDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Imdb25KLemmatized.tsv");
        public static readonly string Imdb50kDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "imdb_50k.tsv"); // label index 1, sentimentText index 2
        public static readonly string Imdb50kLemmatizedDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Imdb50KLemmatized.tsv");
    }
}
