//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace SentimentAnalysis1ML.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("Text"), LoadColumn(0)]
        public string Text { get; set; }


        [ColumnName("Label"), LoadColumn(1)]
        public bool Label { get; set; }


    }
}
