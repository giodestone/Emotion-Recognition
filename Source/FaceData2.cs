using Microsoft.ML.Data;

/// <summary>
/// A class representing all the features of a face, just passed in.
/// </summary>
public class FaceData2
{
    [LoadColumn(0)]
    public string Emotion { get; set; }

    [LoadColumn(1, 68)]
    [VectorType(68)]
    public float[] RawCoordiantesX { get; set; } = new float[68];

    [LoadColumn(69, 136)]
    [VectorType(68)]
    public float[] RawCoordiantesY { get; set; } = new float[68];

    [LoadColumn(137, 204)]
    [VectorType(68)]
    public float[] AngleBetweenFeatures { get; set; } = new float[68];

    [LoadColumn(205, 272)]
    [VectorType(68)]
    public float[] LengthBetweenFeatures { get; set; } = new float[68];
}