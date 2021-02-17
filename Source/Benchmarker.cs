using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Text;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Navigation;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace EmotionRecognitionMLNet
{
    public static class Benchmarker
    {
        /// <summary>
        /// Holds statistics for a particular run which are stored using <see cref="StatValue"/> and <see cref="StatConfusionMatrix"/>.
        /// </summary>
        /// <seealso cref="StatValue"/>
        /// <seealso cref="StatConfusionMatrix"/>
        class BenchmarkStatistics
        {
            public StatValue MacroAccuracy { get; private set; }
            public StatValue MicroAccuracy { get; private set; }
            public StatValue LogLoss { get; private set; }
            public StatValue LogLossReduction { get; private set; }
            public StatValue TimeTakenMiliSeconds { get; private set; }
            public StatConfusionMatrix StatConfusionMatrix { get; private set; }

            public BenchmarkStatistics(int numOfFeatureClasses)
            {
                MacroAccuracy = new StatValue();
                MicroAccuracy = new StatValue();
                LogLoss = new StatValue();
                LogLossReduction = new StatValue();
                TimeTakenMiliSeconds = new StatValue();
                StatConfusionMatrix = new StatConfusionMatrix(numOfFeatureClasses);
            }

            /// <summary>
            /// Add a new set of data.
            /// </summary>
            /// <param name="mcm"></param>
            /// <param name="timeTaken"></param>
            public void AddData(MulticlassClassificationMetrics mcm, double timeTaken)
            {
                MacroAccuracy.AddValue(mcm.MacroAccuracy);
                MicroAccuracy.AddValue(mcm.MicroAccuracy);
                LogLoss.AddValue(mcm.LogLoss);
                LogLossReduction.AddValue(mcm.LogLossReduction);
                TimeTakenMiliSeconds.AddValue(timeTaken);
                StatConfusionMatrix.AddConfusionMatrix(mcm.ConfusionMatrix);
            }
        }

        /// <summary>
        /// Extract face features by calling <see cref="MachineLearning.ExtractFeatures"/> for the <paramref name="faceDataType"/>, if the <paramref name="checkIfPresent"/> is true.
        /// </summary>
        /// <param name="faceDataType"></param>
        /// <param name="checkIfPresent"></param>
        /// <seealso cref="MachineLearning.ExtractFeatures"/>
        public static void ExtractFacialFeatures(Type faceDataType, bool checkIfPresent=true)
        {
            if (checkIfPresent)
            {
                var foundFile = MachineLearning.GetFile(MachineLearning.GetFacialFeaturesFileName(faceDataType), false);

                if (foundFile != null)
                    return;
            }
            MachineLearning.ExtractFeatures(faceDataType);
        }

        /// <summary>
        /// Run the benchmark on the designated <paramref name="faceDataType"/> saving the results in a csv. 
        /// </summary>
        /// <param name="faceDataType"></param>
        public static void RunMaximumIterationBenchmarks(Type faceDataType)
        {
            ExtractFacialFeatures(faceDataType, true);

            const int maxRepeatsExclusive = 5;
            const int maxIterationMultipliersInclusive = 6;
            const int factorMultiplier = 10; // The 10 part of 10^n
            Dictionary<int, BenchmarkStatistics> bsIterations = new Dictionary<int, BenchmarkStatistics>(maxIterationMultipliersInclusive * maxRepeatsExclusive);
            for (int repeats = 0; repeats < maxRepeatsExclusive; ++repeats)
            {
                for (int iterationMultiplier = 1; iterationMultiplier <= maxIterationMultipliersInclusive; ++iterationMultiplier)
                {
                    IDataView testSet = null;
                    int iterations = (int)Math.Pow(factorMultiplier, iterationMultiplier);

                    Stopwatch sw = new Stopwatch();
                    sw.Start();

                    if (faceDataType == typeof(FaceData1))
                        MachineLearning.TrainModel<FaceData1>(out testSet, iterations, false);
                    else if (faceDataType == typeof(FaceData2))
                        MachineLearning.TrainModel<FaceData2>(out testSet, iterations, false);
                    else if (faceDataType == typeof(FaceData3))
                        MachineLearning.TrainModel<FaceData3>(out testSet, iterations, false);
                    else
                        throw new ArgumentException("No corresponding logic for the specified faceDataType",
                            nameof(faceDataType));

                    sw.Stop();

                    MachineLearning.TestModel(testSet, faceDataType, out var testMetrics);

                    if (!bsIterations.ContainsKey(iterationMultiplier))
                        bsIterations.Add(iterationMultiplier, new BenchmarkStatistics(testMetrics.ConfusionMatrix.NumberOfClasses));

                    bsIterations[iterationMultiplier].AddData(testMetrics, (double)sw.ElapsedMilliseconds);
                }
            }

            using (var writer = new StreamWriter($"Iterations Benchmark {faceDataType.ToString()}.csv", false))
            {
                // GENERAL STATS

                writer.Write("Log Loss\n");
                writer.Write($"Iterations, Median, Max, Min\n");
                foreach (var keyValuePair in bsIterations)
                {
                    writer.Write($"{Math.Pow(factorMultiplier, keyValuePair.Key)}, {keyValuePair.Value.LogLoss.Median}, {keyValuePair.Value.LogLoss.Max}, {keyValuePair.Value.LogLoss.Min},\n");
                }
                writer.Write("\n");
                
                writer.Write("Log Loss Reduction\n");
                writer.Write($"Iterations, Median, Max, Min\n");
                foreach (var keyValuePair in bsIterations)
                {
                    writer.Write($"{Math.Pow(factorMultiplier, keyValuePair.Key)}, {keyValuePair.Value.LogLossReduction.Median}, {keyValuePair.Value.LogLossReduction.Max}, {keyValuePair.Value.LogLossReduction.Min},\n");
                }
                writer.Write("\n");

                writer.Write("Micro Accuracy\n");
                writer.Write($"Iterations, Median, Max, Min\n");
                foreach (var keyValuePair in bsIterations)
                {
                    writer.Write($"{Math.Pow(factorMultiplier, keyValuePair.Key)}, {keyValuePair.Value.MicroAccuracy.Median}, {keyValuePair.Value.MicroAccuracy.Max}, {keyValuePair.Value.MicroAccuracy.Min},\n");
                }
                writer.Write("\n");

                writer.Write("Macro Accuracy\n");
                writer.Write($"Iterations, Median, Max, Min\n");
                foreach (var keyValuePair in bsIterations)
                {
                    writer.Write($"{Math.Pow(factorMultiplier, keyValuePair.Key)}, {keyValuePair.Value.MacroAccuracy.Median}, {keyValuePair.Value.MacroAccuracy.Max}, {keyValuePair.Value.MacroAccuracy.Min},\n");
                }
                writer.Write("\n");

                writer.Write("Time Taken\n");
                writer.Write($"Iterations, Median, Max, Min\n");
                foreach (var keyValuePair in bsIterations)
                {
                    writer.Write($"{Math.Pow(factorMultiplier, keyValuePair.Key)}, {keyValuePair.Value.TimeTakenMiliSeconds.Median}, {keyValuePair.Value.TimeTakenMiliSeconds.Max}, {keyValuePair.Value.TimeTakenMiliSeconds.Min},\n");
                }
                writer.Write("\n");

                // CONFUSION MATRICES
                foreach (var keyValuePair in bsIterations)
                {
                    writer.Write("\n");
                    writer.Write(keyValuePair.Value.StatConfusionMatrix.GetString($"Iterations {Math.Pow(factorMultiplier, keyValuePair.Key)}"));
                    writer.Write("\n");
                }

                writer.Flush();
            }
        }
    }
}
