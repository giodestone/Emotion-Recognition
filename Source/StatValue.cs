using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EmotionRecognitionMLNet
{
    /// <summary>
    /// A class for storting some values and retriving their min, median, and max.
    /// </summary>
    public class StatValue
    {
        private List<double> values;
        private int valuesSizeWhenCalculatedValues;

        private double max;
        public double Max
        {
            get
            {
                CalculateValues();
                return max;
            }
            private set => max = value;
        }

        private double median;
        public double Median
        {
            get
            {
                CalculateValues();
                return median;
            }
            private set => median = value;
        }

        private double min;
        public double Min
        {
            get
            {
                CalculateValues();
                return min;
            }
            private set => min = value;
        }

        public StatValue()
        {
            values = new List<double>();
            valuesSizeWhenCalculatedValues = 0;
            Max = double.NaN;
            Median = double.NaN;
            Min = double.NaN;
        }

        public void AddValue(double value)
        {
            values.Add(value);
        }

        private void CalculateValues()
        {
            if (values.Count == valuesSizeWhenCalculatedValues)
                return;

            valuesSizeWhenCalculatedValues = values.Count;
            Max = values.Max();
            Min = values.Min();

            values = values.OrderByDescending(d => d).ToList();

            if (values.Count % 2 == 0)
            {
                var a = values[values.Count / 2 - 1];
                var b = values[values.Count / 2];
                Median = (a + b) / 2.0;
            }
            else
                Median = values[values.Count / 2];
        }
    }
}
