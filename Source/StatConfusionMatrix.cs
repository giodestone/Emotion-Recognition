using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace EmotionRecognitionMLNet
{
    /// <summary>
    /// Confusion matrix, except holds min, median, and max values using <see cref="StatValue"/>.
    /// </summary>
    class StatConfusionMatrix
    {
        public Dictionary<int, StatValue> PerClassPrescision;
        public Dictionary<int, StatValue> PerClassRecall;
        public StatValue[,] RecallMatrix; // row: Predicted Class, col: Actual Class
        public Dictionary<int, string> IdToEmotionName;

        /// <summary>
        /// Creates a stat confusion matrix. Requires <see cref="MachineLearning.IDtoEmotion"/> to be set up.
        /// </summary>
        /// <param name="numOfClasses"></param>
        public StatConfusionMatrix(int numOfClasses)
        {
            // Get names of emotions.
            IdToEmotionName = MachineLearning.IDtoEmotion;

            // Setup the rest of the values.
            PerClassPrescision = new Dictionary<int, StatValue>(numOfClasses);
            PerClassRecall = new Dictionary<int, StatValue>(numOfClasses);
            RecallMatrix = new StatValue[numOfClasses, numOfClasses];

            for (int i = 0; i < numOfClasses; ++i)
            {
                PerClassRecall[i] = new StatValue();
                PerClassPrescision[i] = new StatValue();

                for (int j = 0; j < numOfClasses; ++j)
                {
                    RecallMatrix[i, j] = new StatValue();
                }
            }
        }

        /// <summary>
        /// Add a confusion matrix to the stats.
        /// </summary>
        /// <param name="cm"></param>
        public void AddConfusionMatrix(ConfusionMatrix cm)
        {
            for (int i = 0; i < cm.NumberOfClasses; ++i)
            {
                PerClassRecall[i].AddValue(cm.PerClassRecall[i]);
                PerClassPrescision[i].AddValue(cm.PerClassPrecision[i]);
                for (int j = 0; j < cm.NumberOfClasses; ++j)
                {
                    RecallMatrix[i, j].AddValue(cm.GetCountForClassPair(i, j));
                }
            }
        }

        /// <summary>
        /// Get a string formatted in csv format.
        /// </summary>
        /// <param name="matrixTitle"></param>
        /// <returns></returns>
        public string GetString(string matrixTitle)
        {
            using (StringWriter writer = new StringWriter())
            {
                // MAX
                writer.Write($"Max Confusion Matrix {matrixTitle}.");
                writer.Write(",");
                foreach (var key in IdToEmotionName.Keys)
                {
                    writer.Write($"{key},");
                }

                writer.Write("Recall,\n");

                var iters = Math.Sqrt(RecallMatrix.Length);

                for (var i = 0; i < iters; i++)
                {
                    writer.Write($"{i}. {IdToEmotionName[i]},");
                    for (var j = 0; j < iters; ++j)
                    {
                        writer.Write($"{RecallMatrix[i, j].Max},");
                    }
                    writer.Write(PerClassRecall[i].Max);
                    writer.Write("\n");
                }

                writer.Write("Precision,");
                for (int i = 0; i < iters; ++i)
                {
                    writer.Write($"{PerClassPrescision[i].Max},");
                }

                writer.Write("\n\n");

                // MEDIAN
                writer.Write($"Median Confusion Matrix {matrixTitle}.");
                writer.Write(",");
                foreach (var key in IdToEmotionName.Keys)
                {
                    writer.Write($"{key},");
                }

                writer.Write("Recall,\n");

                for (var i = 0; i < iters; i++)
                {
                    writer.Write($"{i}. {IdToEmotionName[i]},");
                    for (var j = 0; j < iters; ++j)
                    {
                        writer.Write($"{RecallMatrix[i, j].Median},");
                    }
                    writer.Write(PerClassRecall[i].Median);
                    writer.Write("\n");
                }

                writer.Write("Precision,");
                for (int i = 0; i < iters; ++i)
                {
                    writer.Write($"{PerClassPrescision[i].Median},");
                }

                writer.Write("\n\n");

                // MIN
                writer.Write($"Min Confusion Matrix {matrixTitle}.");
                writer.Write(",");
                foreach (var key in IdToEmotionName.Keys)
                {
                    writer.Write($"{key},");
                }

                writer.Write("Recall,\n");

                for (var i = 0; i < iters; i++)
                {
                    writer.Write($"{i}. {IdToEmotionName[i]},");
                    for (var j = 0; j < iters; ++j)
                    {
                        writer.Write($"{RecallMatrix[i, j].Min},");
                    }
                    writer.Write(PerClassRecall[i].Min);
                    writer.Write("\n");
                }

                writer.Write("Precision,");
                for (int i = 0; i < iters; ++i)
                {
                    writer.Write($"{PerClassPrescision[i].Min},");
                }

                writer.Write("\n\n");


                writer.Flush();
                return writer.ToString();
            }
        }
    }
}
