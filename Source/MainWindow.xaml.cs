using Microsoft.ML;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace EmotionRecognitionMLNet
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            currentFaceDataMenuItem = FaceData1MenuItem;
        }

        private OpenFileDialog openedFileDialog;
        private FileInfo currentImageFileInfo;
        private IDataView testSetDataView;

        private Type selectedFaceDataType = typeof(FaceData1);
        private MenuItem currentFaceDataMenuItem = null;

        /// <summary>
        /// Callback for when the LoadImage button is clicked. Gives the user a file browser to allow them to select an image file. If one is selected,
        /// the <see cref="currentImageFileInfo"/> and the <see cref="InputImage"/> fields are updated with the selection.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LoadImageClicked(object sender, RoutedEventArgs e)
        {
            openedFileDialog = new OpenFileDialog()
            {
                Title = "Open Image File",
                FileName = "input.jpg",
                Filter = "Image Files(*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                InitialDirectory = AppDomain.CurrentDomain.BaseDirectory
            };

            var fileDialogResult = openedFileDialog.ShowDialog(this);
            if (fileDialogResult.HasValue && fileDialogResult.Value)
            {
                currentImageFileInfo = new FileInfo(openedFileDialog.FileName);
                InputImage.Source = new BitmapImage(new Uri(openedFileDialog.FileName, UriKind.Absolute));
            }
        }

        /// <summary>
        /// Callback for when the Train Model button is clicked. Disables all button and launches a new thread which asks extracts features (if not present) and then trains the model using
        /// <see cref="MachineLearning.TrainModel{TFaceData}"/> (if not present), prompting the user if an existing version is found. The face data type passed in is dependent on <see cref="selectedFaceDataType"/>.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private async void TrainModelButton_Click(object sender, RoutedEventArgs e)
        {
            TrainModelButton.IsEnabled = false;
            PredictEmotionButton.IsEnabled = false;
            LoadImageToPredict.IsEnabled = false;
            EvaluateModelButton.IsEnabled = false;
            RunBenchmarksMenuItem.IsEnabled = false;
            var trainButtonOriginalLabel = (string)TrainModelButton.Content;
            TrainModelButton.Content = $"{trainButtonOriginalLabel} (Training...)";

            await Task.Run(() =>
            {
                var existingFile = MachineLearning.GetFile(MachineLearning.GetFacialFeaturesFileName(selectedFaceDataType), false);

                bool recalculatedFacialFeatures = false;

                if (existingFile != null && IsExistingFileValid(existingFile))
                {
                    if (MessageBox.Show("Existing facial landmarks file exists, extract again (will take extra time)?",
                            "Extracted Facial Feature File Found", MessageBoxButton.YesNo, MessageBoxImage.Question) ==
                        MessageBoxResult.Yes)
                    {
                        MachineLearning.ExtractFeatures(selectedFaceDataType);
                        recalculatedFacialFeatures = true;
                    }
                }
                else
                {
                    MachineLearning.ExtractFeatures(selectedFaceDataType);
                    recalculatedFacialFeatures = true;
                }

                if (recalculatedFacialFeatures)
                {
                    TrainModelAccordingToCurrentFaceMode();
                }
                else if (MachineLearning.GetFile(MachineLearning.GetModelZipFileName(selectedFaceDataType), false) !=
                         null)
                {
                    if (MessageBox.Show("Trained model found, use the one stored on disk?", "Trained Model Found",
                        MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.No)
                    {
                        TrainModelAccordingToCurrentFaceMode();
                    }
                }
                else
                {
                    TrainModelAccordingToCurrentFaceMode();
                }
            });

            TrainModelButton.Content = $"{trainButtonOriginalLabel} (Trained)";
            TrainModelButton.IsEnabled = true;
            PredictEmotionButton.IsEnabled = true;
            LoadImageToPredict.IsEnabled = true;
            EvaluateModelButton.IsEnabled = true;
            RunBenchmarksMenuItem.IsEnabled = true;
        }

        /// <summary>
        /// A check to determine if the existing file contains enough information for it to be considered valid.
        /// </summary>
        /// <param name="existingFile"></param>
        /// <returns></returns>
        private bool IsExistingFileValid(FileInfo existingFile)
        {
            return existingFile.Length > 1024;
        }

        /// <summary>
        /// Calls <see cref="MachineLearning.TrainModel{TFaceData}"/> with the TFaceData set according to <see cref="selectedFaceDataType"/> as a workaround
        /// as type variables cannot be cast to generic types. Shows a message box informing the user once its done training.
        /// </summary>
        private void TrainModelAccordingToCurrentFaceMode()
        {
            // Because you can't pass FaceDataType into a generic without serious workarounds due to the use of an out paramater.
            if (selectedFaceDataType == typeof(FaceData1))
            {
                MachineLearning.TrainModel<FaceData1>(out testSetDataView, 100000);
            }
            else if (selectedFaceDataType == typeof(FaceData2))
            {
                MachineLearning.TrainModel<FaceData2>(out testSetDataView, 1000000);
            }
            else if (selectedFaceDataType == typeof(FaceData3))
            {
                MachineLearning.TrainModel<FaceData3>(out testSetDataView,100000);
            }

            MessageBox.Show("Training Complete", "Training is complete.", MessageBoxButton.OK,
                MessageBoxImage.Information);
        }

        /// <summary>
        /// Callback for when the <see cref="PredictEmotionButton"/> is clicked. This launches a new task which calls <see cref="MachineLearning.PredictEmotion"/> with the
        /// <see cref="currentImageFileInfo"/> and <see cref="selectedFaceDataType"/>, which results are then used to update <see cref="PredictedEmotionMainLabel"/> and
        /// <see cref="PredictedEmotionsListBox"/>.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private async void PredictEmotionButton_Click(object sender, RoutedEventArgs e)
        {
            var originalLabel = PredictEmotionButton.Content;
            PredictedEmotionMainLabel.Content = "Predicting...";
            PredictedEmotionsListBox.Items.Clear();
            PredictEmotionButton.Content = (string)PredictEmotionButton.Content + " (Predicting...)";
            PredictEmotionButton.IsEnabled = false;
            FaceOutput predictedEmotion = null;
            Dictionary<string, float> predictedEmotionsWithAllLabels = null;

            MachineLearning.DrawPointsOfLandmarks(currentImageFileInfo);

            await Task.Run(() =>
            {
                MachineLearning.PredictEmotion(currentImageFileInfo, selectedFaceDataType, out var prediction, out var predictionWithLabels);
                predictedEmotion = prediction;
                predictedEmotionsWithAllLabels = predictionWithLabels;
            });

            PredictedEmotionMainLabel.Content = $"{FirstLetterToUpper(predictedEmotion.PredictedEmotion)}" +
                                                $" ({FloatToPercent(predictedEmotionsWithAllLabels[predictedEmotion.PredictedEmotion])})";

            PredictedEmotionsListBox.Items.Clear();
            predictedEmotionsWithAllLabels = predictedEmotionsWithAllLabels.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
            foreach (var predictedEmotionLabelAndScore in predictedEmotionsWithAllLabels)
            {
                PredictedEmotionsListBox.Items.Add(
                    $"{FirstLetterToUpper(predictedEmotionLabelAndScore.Key)}: {FloatToPercent(predictedEmotionLabelAndScore.Value)}");
            }

            PredictedEmotionsListBox.Items.RemoveAt(0);

            PredictEmotionButton.Content = originalLabel;
            PredictEmotionButton.IsEnabled = true;
        }

        string FirstLetterToUpper(string str)
        {
            return str[0].ToString().ToUpper() + str.Substring(1, str.Length - 1);
        }

        string FloatToPercent(float num)
        {
            return num.ToString("P");
        }

        /// <summary>
        /// Callback for when <see cref="EvaluateModelButton"/> is clicked. Displays the <see cref="EvaluationMetrics"/> window.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void EvaluateModelButton_Click(object sender, RoutedEventArgs e)
        {
            var evaluationMetricsWindow = new EvaluationMetrics(testSetDataView, selectedFaceDataType) { Owner = this };
            evaluationMetricsWindow.Show();
        }

        /// <summary>
        /// Callback for when the <see cref="RunBenchmarksMenuItem"/> is clicked. TODO
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private async void RunBenchmarksMenuItem_Click(object sender, RoutedEventArgs e)
        {
            var originalHeader = (string)RunBenchmarksMenuItem.Header;
            RunBenchmarksMenuItem.Header = $"{originalHeader} (Running...)";
            SetEnabledStateOfAllUserControls(false);

            await Task.Run(() =>
            {
                Benchmarker.RunMaximumIterationBenchmarks(typeof(FaceData1));
                Benchmarker.RunMaximumIterationBenchmarks(typeof(FaceData2));
                Benchmarker.RunMaximumIterationBenchmarks(typeof(FaceData3));
            });

            MessageBox.Show("Finished Benchmarks!", "Finished Benchmarks", MessageBoxButton.OK,
                MessageBoxImage.Information);

            RunBenchmarksMenuItem.Header = originalHeader;
            SetEnabledStateOfAllUserControls(true);
        }

        /// <summary>
        /// Callback for when <see cref="FaceData1MenuItem"/> is clicked. Updates <see cref="currentFaceDataMenuItem"/>, calls <see cref="ChangeStateOfMenuItemsAndFaceDataType"/> and <see cref="ResetButtons"/>.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void FaceData1MenuItem_Click(object sender, RoutedEventArgs e)
        {
            if (currentFaceDataMenuItem == FaceData1MenuItem)
            {
                return;
            }

            ChangeStateOfMenuItemsAndFaceDataType(typeof(FaceData1), FaceData1MenuItem);
            ResetButtons();
        }

        /// <summary>
        /// Callback for when <see cref="FaceData2MenuItem"/> is clicked. Updates <see cref="currentFaceDataMenuItem"/>, calls <see cref="ChangeStateOfMenuItemsAndFaceDataType"/> and <see cref="ResetButtons"/>.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void FaceData2MenuItem_Click(object sender, RoutedEventArgs e)
        {
            if (currentFaceDataMenuItem == FaceData2MenuItem)
            {
                return;
            }

            ChangeStateOfMenuItemsAndFaceDataType(typeof(FaceData2), FaceData2MenuItem);
            ResetButtons();
        }

        /// <summary>
        /// Callback for when <see cref="FaceData3MenuItem"/> is clicked. Updates <see cref="currentFaceDataMenuItem"/>, calls <see cref="ChangeStateOfMenuItemsAndFaceDataType"/> and <see cref="ResetButtons"/>.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void FaceData3MenuItem_Click(object sender, RoutedEventArgs e)
        {
            if (currentFaceDataMenuItem == FaceData3MenuItem)
            {
                return;
            }

            ChangeStateOfMenuItemsAndFaceDataType(typeof(FaceData3), FaceData3MenuItem);
            ResetButtons();
        }

        /// <summary>
        /// Updates the menu items inside of the <see cref="FaceDataModeMenuItem"/>. Makes sure that only one item can be checked, unchecking the previous one and
        /// updates <see cref="selectedFaceDataType"/>.
        /// </summary>
        /// <param name="tFaceData"></param>
        /// <param name="clickOnMenuItem"></param>
        /// <seealso cref="FaceData1MenuItem_Click"/>
        /// <seealso cref="FaceData2MenuItem"/>
        /// <seealso cref="FaceData3MenuItem"/>
        private void ChangeStateOfMenuItemsAndFaceDataType(Type tFaceData, MenuItem clickOnMenuItem)
        {
            currentFaceDataMenuItem.IsChecked = false;
            clickOnMenuItem.IsChecked = true;
            currentFaceDataMenuItem = clickOnMenuItem;
            selectedFaceDataType = tFaceData;
        }

        private void SetEnabledStateOfAllUserControls(bool state)
        {
            TrainModelButton.IsEnabled = state;
            LoadImageToPredict.IsEnabled = state;
            PredictEmotionButton.IsEnabled = state;
            EvaluateModelButton.IsEnabled = state;
            RunBenchmarksMenuItem.IsEnabled = state;
            FaceDataModeMenuItem.IsEnabled = state;
        }

        /// <summary>
        /// Resets the buttons to what they were when the application is first launched, excluding <see cref="FaceDataModeMenuItem"/> and its children.
        /// </summary>
        private void ResetButtons()
        {
            PredictEmotionButton.IsEnabled = false;
            EvaluateModelButton.IsEnabled = false;
            LoadImageToPredict.IsEnabled = false;
            testSetDataView = null;
            TrainModelButton.Content = "Train Model";
        }
    }
}
