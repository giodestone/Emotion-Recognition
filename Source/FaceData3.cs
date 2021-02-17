using Microsoft.ML.Data;

/// <summary>
/// A data structure representing the features of a face.
/// </summary>
class FaceData3
{
    [LoadColumn(0)]
    public string Emotion { get; set; }

    [LoadColumn(1)]
    public float LeftEyebrowDistance { get; set; }

    [LoadColumn(2)]
    public float RightEyebrowDistance { get; set; }

    [LoadColumn(3)]
    public float LeftEyeWidth { get; set; }

    [LoadColumn(4)]
    public float RightEyeWidth { get; set; }

    [LoadColumn(5)]
    public float LeftEyeHeight { get; set; }

    [LoadColumn(6)]
    public float RightEyeHeight { get; set; }

    [LoadColumn(7)]
    public float OuterLipWidth { get; set; }

    [LoadColumn(8)]
    public float InnerLipWidth { get; set; }

    [LoadColumn(9)]
    public float OuterLipHeight { get; set; }

    [LoadColumn(10)]
    public float InnerLipHeight { get; set; }

    [LoadColumn(11)]
    public float LeftLipEdgeAngle { get; set; }

    [LoadColumn(12)]
    public float RightLipEdgeAngle { get; set; }
}

