using Microsoft.ML;
using System;
using System.Windows;

namespace EmotionRecognitionMLNet
{
    /// <summary>
    /// Interaction logic for EvaluationMetrics.xaml
    /// </summary>
    public partial class EvaluationMetrics : Window
    {
        public EvaluationMetrics(IDataView testSetDataView, Type TFaceData)
        {
            InitializeComponent();

            // Test model and display results.
            MachineLearning.TestModel(testSetDataView, TFaceData, out var metrics);
            ConfusionMatrixTextBlock.Text = "";
            ConfusionMatrixTextBlock.Text = metrics.ConfusionMatrix.GetFormattedConfusionTable();
            MicroAccuracyTextBlock.Text = metrics.MicroAccuracy.ToString("#.###");
            MacroAccuracyTextBlock.Text = metrics.MacroAccuracy.ToString("#.###");
            LogLossTextBlock.Text = metrics.LogLoss.ToString("#.###");
            LogLossReductionTextBlock.Text = metrics.LogLossReduction.ToString("#.###");
        }

        private void CloseWindowButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}
