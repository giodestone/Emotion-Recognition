using Microsoft.ML.Data;

/// <summary>
/// Data structure for storing the face output.
/// </summary>
public class FaceOutput
{
    [ColumnName("PredictedLabel")]
    public string PredictedEmotion;

    [ColumnName("Score")]
    public float[] Scores;
}