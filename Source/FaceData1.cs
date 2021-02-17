using Microsoft.ML.Data;

/// <summary>
/// Data structure for storing data points about the face.
/// </summary>
public class FaceData1
{
    [LoadColumn(0)]
    public string Emotion { get; set; }

    [LoadColumn(1)]
    public float LeftEyebrow { get; set; }

    [LoadColumn(2)]
    public float RightEyebrow { get; set; }

    [LoadColumn(3)]
    public float LeftLip { get; set; }

    [LoadColumn(4)]
    public float RightLip { get; set; }

    [LoadColumn(5)]
    public float LipHeight { get; set; }

    [LoadColumn(6)]
    public float LipWidth { get; set; }
}
