using DlibDotNet;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Windows;
using Dlib = DlibDotNet.Dlib;
using Point = DlibDotNet.Point;

static class MachineLearning
{
    private const string ShapePredictorFileName = "shape_predictor_68_face_landmarks.dat";

    private const string ModelZipFileExtension = ".zip";
    private const string DataViewFileExtension = ".txt";

    private static MLContext mlContext = null;
    private static ITransformer model = null;

    /// <summary>
    /// Get the name of the emotion, which is usually mapped to an id.
    /// </summary>
    public static Dictionary<int, string> IDtoEmotion { get; private set; }


    /// <summary>
    /// Extract features from images, and store them in a csv file at the .exe directory.
    /// </summary>
    /// <param name="faceData">Type of face data that will be stored.</param>
    /// <seealso cref="FaceData1"/>
    /// <seealso cref="FaceData2"/>
    /// <seealso cref="FaceData3"/>
    public static void ExtractFeatures(Type faceData)
    {
        // Setup CSV file
        string header = $"Emotion, Mode: {faceData.ToString()}\n";
        File.WriteAllText(GetFacialFeaturesFileName(faceData), header);

        StreamWriter csvFile = new StreamWriter(GetFacialFeaturesFileName(faceData), true);

        using (var fd = Dlib.GetFrontalFaceDetector())
        using (var sp = ShapePredictor.Deserialize(GetFile(ShapePredictorFileName).FullName))
        {
            var dirInfos = GetDataSetDirectoryInfos();

            if (dirInfos.Count == 0)
            {
                MessageBox.Show("Unable to find any image data sets. Program exiting.", "Error Finding Datasets",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                throw new FileNotFoundException("Unable to find any image data sets!");
            }

            List<FileInfo> imageFiles = new List<FileInfo>();
            foreach (var dir in dirInfos)
            {
                foreach (var directoryInfo in dir.GetDirectories())
                {
                    imageFiles.AddRange(directoryInfo.GetFiles("*.png"));
                }
            }

            foreach (var imageFile in imageFiles)
            {
                if (faceData == typeof(FaceData1))
                {
                    var faceData1 = GetFaceData1FromImage(imageFile, sp, fd);
                    if (faceData1 == null)
                    {
                        continue;
                    }

                    // Write to CSV
                    csvFile.WriteLine(faceData1.Emotion + "," + faceData1.LeftEyebrow + "," + faceData1.RightEyebrow +
                                      "," + faceData1.LeftLip + "," + faceData1.RightLip +
                                      "," + faceData1.LipHeight + "," + faceData1.LipWidth);
                }
                else if (faceData == typeof(FaceData2))
                {
                    var faceData2 = GetFaceData2FromImage(imageFile, sp, fd);
                    if (faceData2 == null)
                    {
                        continue;
                    }

                    csvFile.Write($"{faceData2.Emotion},");

                    foreach (var rawCoordsX in faceData2.RawCoordiantesX)
                    {
                        csvFile.Write($"{rawCoordsX},");
                    }

                    foreach (var rawCoordsY in faceData2.RawCoordiantesY)
                    {
                        csvFile.Write($"{rawCoordsY},");
                    }

                    foreach (var angleOfFeature in faceData2.AngleBetweenFeatures)
                    {
                        csvFile.Write($"{angleOfFeature},");
                    }

                    foreach (var lengthOfFeature in faceData2.LengthBetweenFeatures)
                    {
                        csvFile.Write($"{lengthOfFeature},");
                    }

                    csvFile.Write("\n");
                }
                else if (faceData == typeof(FaceData3))
                {
                    var faceData3 = GetFaceData3FromImage(imageFile, sp, fd);
                    if (faceData3 == null)
                    {
                        continue;
                    }

                    csvFile.WriteLine(
                        $"{faceData3.Emotion},{faceData3.LeftEyebrowDistance},{faceData3.RightEyebrowDistance},{faceData3.LeftEyeWidth},{faceData3.RightEyeWidth},{faceData3.LeftEyeHeight},{faceData3.RightEyeHeight},{faceData3.OuterLipWidth},{faceData3.InnerLipWidth},{faceData3.OuterLipHeight},{faceData3.InnerLipHeight},{faceData3.LeftLipEdgeAngle},{faceData3.RightLipEdgeAngle}");
                }
                else
                {
                    throw new ArgumentException("Invalid TFaceData.", "faceData");
                }
            }
            // Close file
            csvFile.Close();
        }
    }

    private static List<DirectoryInfo> GetDataSetDirectoryInfos()
    {
        // Get all images
        List<DirectoryInfo> dirInfos = new List<DirectoryInfo>();

        try
        {
            var mugImgDirInfo = GetDirectory("MUG Images");
            dirInfos.Add(mugImgDirInfo);
        }
        catch
        {
            // ignored
        }

        try
        {
            var googleImgDirInfo = GetDirectory("Google Set");
            dirInfos.Add(googleImgDirInfo);
        }
        catch
        {
            // ignored
        }

        try
        {
            var cohnKanadeeImgDirInfo = GetDirectory("Cohn-Kanade Images");
            dirInfos.Add(cohnKanadeeImgDirInfo);
        }
        catch
        {
            // ignored
        }

        try
        {
            var cohnKanadeeImgDirInfo = GetDirectory("CK+");
            dirInfos.Add(cohnKanadeeImgDirInfo);
        }
        catch
        {
            // ignored
        }

        return dirInfos;
    }

    public static void DrawPointsOfLandmarks(FileInfo image)
    {
        using (var fd = Dlib.GetFrontalFaceDetector())
        using (var sp = ShapePredictor.Deserialize(GetFile(ShapePredictorFileName).FullName))
        {
            using (var img = Dlib.LoadImage<RgbPixel>(image.FullName))
            {
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    var shape = sp.Detect(img, face);
                    // draw the landmark points on the image
                    for (var i = 0; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);
                        if (i == 0)
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 255), thickness: 8);
                        }
                        else if (i == 21 || i == 22 || i == 39 || i == 42 || i == 33 || i == 51 || i == 57 ||
                                 i == 48 ||
                                 i == 54)
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 255), thickness: 4);
                        }
                        else if (i == 18 || i == 19 || i == 20 || i == 21) // left eye
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 0), 6);
                        }
                        else if (i == 22 || i == 23 || i == 24 || i == 25) // right eye
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 128, 0), 6);
                        }
                        else if (i == 48 || i == 49 || i == 50) // left lip
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), 2);
                        }
                        else if (i == 52 || i == 53 || i == 54) // right lip
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 128), 2);
                        }
                        else
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 0, 0), thickness: 4);
                        }
                    }
                    Dlib.SavePng(img, "output.jpg");
                }
            }
        }
    }

    /// <summary>
    /// Train the model based on the FaceData. Features need to be extracted first using <see cref="ExtractFeatures"/>.
    /// </summary>
    /// <typeparam name="TFaceData">Type of the FaceData being passed in. Used this way because a Type variable cannot be used as a type paramater.</typeparam>
    /// <param name="testSetDataView">The test split generated for later testing.</param>
    public static void TrainModel<TFaceData>(out IDataView testSetDataView, int maximumNumberOfIterations = 10000, bool saveModelAndDataViews = true)
    {
        // Create ML Context
        mlContext = new MLContext();

        // Create interface to data.
        IDataView inputData = mlContext.Data.LoadFromTextFile<TFaceData>(GetFacialFeaturesFileName(typeof(TFaceData)), hasHeader: true, separatorChar: ',');

        var trainTestSplit = mlContext.Data.TrainTestSplit(inputData, testFraction: 0.2);
        var dataView = trainTestSplit.TrainSet;
        testSetDataView = trainTestSplit.TestSet;

        var featureVectorName = "FaceFeatures";
        var labelColumnName = "Label";

        EstimatorChain<KeyToValueMappingTransformer> pipeline = null;

        var o = new Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer.Options()
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = featureVectorName,
            MaximumNumberOfIterations = maximumNumberOfIterations
        };

        if (typeof(TFaceData) == typeof(FaceData1))
        {
            pipeline = mlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: "Emotion", outputColumnName: labelColumnName) // Tell that Emotion map to PredictionEmotions as the label aka Emotion ==> Predicted Emotion.
            .Append(mlContext.Transforms.Concatenate(
                featureVectorName,
                "LeftEyebrow",
                "RightEyebrow",
                "LeftLip",
                "RightLip",
                "LipHeight",
                "LipWidth")) // Map the feature vector/array/list with these values.
            .AppendCacheCheckpoint(mlContext) // Tell the algorithm to cache stuff.
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(o)) // This line actually determines the input, output neurons and layers in between and some stuff like epochs, learning rate etc.
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }
        else if (typeof(TFaceData) == typeof(FaceData2))
        {
            pipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Emotion", outputColumnName: labelColumnName) // Tell that Emotion map to PredictionEmotions as the label aka Emotion ==> Predicted Emotion.
                .Append(mlContext.Transforms.Concatenate(
                    featureVectorName,
                    "RawCoordiantesX",
                    "RawCoordiantesY",
                    "AngleBetweenFeatures",
                    "LengthBetweenFeatures")) // Map the feature vector/array/list with these values.
                .AppendCacheCheckpoint(mlContext) // Tell the algorithm to cache stuff.
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(o)) // This line actually determines the input, output neurons and layers in between and some stuff like epochs, learning rate etc.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        }
        else if (typeof(TFaceData) == typeof(FaceData3))
        {
            pipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Emotion", outputColumnName: labelColumnName) // Tell that Emotion map to PredictionEmotions as the label aka Emotion ==> Predicted Emotion.
                .Append(mlContext.Transforms.Concatenate(
                    featureVectorName,
                    "LeftEyebrowDistance",
                    "RightEyebrowDistance",
                    "LeftEyeWidth",
                    "RightEyeWidth",
                    "LeftEyeHeight",
                    "RightEyeHeight",
                    "OuterLipWidth",
                    "InnerLipWidth",
                    "OuterLipHeight",
                    "InnerLipHeight",
                    "LeftLipEdgeAngle",
                    "RightLipEdgeAngle")) // Map the feature vector/array/list with these values.
                .AppendCacheCheckpoint(mlContext) // Tell the algorithm to cache stuff.
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(o)) // This line actually determines the input, output neurons and layers in between and some stuff like epochs, learning rate etc.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        }
        else
        {
            throw new ArgumentException("Invalid TFaceData, should be a FaceData.", "TFaceData");
        }

        ///* 5 fold cross validation end */

        //// this passes the data and makes a model. this determines how long to train for etc.
        model = pipeline.Fit(dataView);

        SetupEmotionsToId(inputData.Schema);

        if (!saveModelAndDataViews)
            return;

        // Save the model for future use
        using (var fileStream = new FileStream(GetModelZipFileName(typeof(TFaceData)), FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            mlContext.Model.Save(model, dataView.Schema, fileStream);
        }

        using (var testSetFileStream =
            new FileStream(GetTestSetDataViewFileName(typeof(TFaceData)), FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            mlContext.Data.SaveAsText(trainTestSplit.TestSet, testSetFileStream);
        }

        using (var trainSetFileStream =
            new FileStream(GetTrainSetDatViewFileName(typeof(TFaceData)), FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            mlContext.Data.SaveAsText(trainTestSplit.TrainSet, trainSetFileStream);
        }
    }

    /// <summary>
    /// Sets up the <see cref="IDtoEmotion"/> field.
    /// </summary>
    /// <param name="inputDataViewSchema">The schema when the initial data view is loaded. This will be inside of <see cref="TrainModel{TFaceData}"/>.</param>
    private static void SetupEmotionsToId(DataViewSchema inputDataViewSchema)
    {
        // Get names of emotions.

        var slotNames = new VBuffer<ReadOnlyMemory<char>>();
        model.GetOutputSchema(inputDataViewSchema).GetColumnOrNull("Label")?.GetKeyValues(ref slotNames);
        IDtoEmotion = new Dictionary<int, string>();
        int num = 0;
        foreach (var denseValue in slotNames.DenseValues())
        {
            IDtoEmotion.Add(num++, denseValue.ToString());
        }
    }

    /// <summary>
    /// Test the created model using the <paramref name="testSetDataView"/>, which was generated by the <see cref="TrainModel{TFaceData}"/> function.
    /// Loads the corresponding model if not in memory, although checks must be performed before otherwise an exception will be thrown.
    /// </summary>
    /// <param name="testSetDataView">Test data set, as generated by the <see cref="TrainModel{TFaceData}"/>. Can be null - then is loaded from disk, checks must be present before calling to see if they exist.</param>
    /// <param name="TFaceData">Struct used to hold the extracted features.</param>
    /// <param name="testMetrics">Test results.</param>
    public static void TestModel(IDataView testSetDataView, Type TFaceData, out MulticlassClassificationMetrics testMetrics)
    {
        if (mlContext == null)
        {
            mlContext = new MLContext();
        }

        if (model == null)
        {
            model = mlContext.Model.Load(GetModelZipFileName(TFaceData), out var dataView);
        }

        if (testSetDataView == null)
        {
            testSetDataView = mlContext.Data.LoadFromTextFile(GetTestSetDataViewFileName(TFaceData));
        }

        testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testSetDataView));
    }

    /// <summary>
    /// Predict the emotion of an image.
    /// </summary>
    /// <param name="imageFileInfo"><see cref="FileInfo"/> of the image file.</param>
    /// <param name="TFaceData">Type of face data that the parameters should be used.</param>
    /// <param name="predictedEmotion">The emotion that was predicted.</param>
    /// <param name="predictedEmotionWithAllLabels">All the other emotions with their scores appended after.</param>
    public static void PredictEmotion(FileInfo imageFileInfo, Type TFaceData, out FaceOutput predictedEmotion, out Dictionary<string, float> predictedEmotionWithAllLabels)
    {
        if (mlContext == null)
        {
            mlContext = new MLContext();
        }

        if (model == null)
        {
            model = mlContext.Model.Load(GetModelZipFileName(TFaceData), out var dataView);
        }

        predictedEmotion = null;
        predictedEmotionWithAllLabels = null;

        // Not using generics because different function calls are required anyway.
        if (TFaceData == typeof(FaceData1))
        {
            using (var predictor = mlContext.Model.CreatePredictionEngine<FaceData1, FaceOutput>(model))
            {
                using (var fd = Dlib.GetFrontalFaceDetector())
                using (var sp = ShapePredictor.Deserialize(GetFile(ShapePredictorFileName).FullName))
                {
                    var faceDataFromImage = GetFaceData1FromImage(imageFileInfo, sp, fd, false);
                    faceDataFromImage.Emotion = ""; // Get rid of label, as this is what we want to know.

                    predictedEmotion = predictor.Predict(faceDataFromImage);
                }

                // Prediction with all labels.
                predictedEmotionWithAllLabels = new Dictionary<string, float>();
                var slotNames = new VBuffer<ReadOnlyMemory<char>>();
                predictor.OutputSchema.GetColumnOrNull("Label")?.GetKeyValues(ref slotNames);
                var names = new string[slotNames.Length];
                var num = 0;
                foreach (var denseValue in slotNames.DenseValues())
                {
                    predictedEmotionWithAllLabels.Add(denseValue.ToString(), predictedEmotion.Scores[num++]);
                }

                Console.WriteLine("Predicted Emotion: " + predictedEmotion.PredictedEmotion);
                Console.WriteLine($"Scores: {string.Join(" ", predictedEmotion.Scores)}");
            }
        }
        else if (TFaceData == typeof(FaceData2))
        {
            using (var predictor = mlContext.Model.CreatePredictionEngine<FaceData2, FaceOutput>(model))
            {
                using (var fd = Dlib.GetFrontalFaceDetector())
                using (var sp = ShapePredictor.Deserialize(GetFile(ShapePredictorFileName).FullName))
                {
                    var faceDataFromImage = GetFaceData2FromImage(imageFileInfo, sp, fd, false);
                    faceDataFromImage.Emotion = ""; // Get rid of label, as this is what we want to know.

                    predictedEmotion = predictor.Predict(faceDataFromImage);
                }

                // Prediction with all labels.
                predictedEmotionWithAllLabels = new Dictionary<string, float>();
                var slotNames = new VBuffer<ReadOnlyMemory<char>>();
                predictor.OutputSchema.GetColumnOrNull("Label")?.GetKeyValues(ref slotNames);
                var names = new string[slotNames.Length];
                var num = 0;
                foreach (var denseValue in slotNames.DenseValues())
                {
                    predictedEmotionWithAllLabels.Add(denseValue.ToString(), predictedEmotion.Scores[num++]);
                }

                Console.WriteLine("Predicted Emotion: " + predictedEmotion.PredictedEmotion);
                Console.WriteLine($"Scores: {string.Join(" ", predictedEmotion.Scores)}");
            }
        }
        else if (TFaceData == typeof(FaceData3))
        {
            using (var predictor = mlContext.Model.CreatePredictionEngine<FaceData3, FaceOutput>(model))
            {
                using (var fd = Dlib.GetFrontalFaceDetector())
                using (var sp = ShapePredictor.Deserialize(GetFile(ShapePredictorFileName).FullName))
                {
                    var faceDataFromImage = GetFaceData3FromImage(imageFileInfo, sp, fd, false);
                    faceDataFromImage.Emotion = ""; // Get rid of label, as this is what we want to know.

                    predictedEmotion = predictor.Predict(faceDataFromImage);
                }

                // Prediction with all labels.
                predictedEmotionWithAllLabels = new Dictionary<string, float>();
                var slotNames = new VBuffer<ReadOnlyMemory<char>>();
                predictor.OutputSchema.GetColumnOrNull("Label")?.GetKeyValues(ref slotNames);
                var names = new string[slotNames.Length];
                var num = 0;
                foreach (var denseValue in slotNames.DenseValues())
                {
                    predictedEmotionWithAllLabels.Add(denseValue.ToString(), predictedEmotion.Scores[num++]);
                }

                Console.WriteLine("Predicted Emotion: " + predictedEmotion.PredictedEmotion);
                Console.WriteLine($"Scores: {string.Join(" ", predictedEmotion.Scores)}");
            }
        }
    }

    static double GetTotalFeatureLength(Point refPoint, Point[] eyebrowPoints, Point refPoint2)
    {
        var tempPoints = eyebrowPoints.Clone() as Point[];
        double total = 0.0;
        for (int i = 0; i < tempPoints.Length; ++i)
        {
            tempPoints[i] -= refPoint;
            total += tempPoints[i].Length;
        }

        return total / (refPoint2 - refPoint).Length;
    }

    static string GetLabel(FileInfo img)
    {
        if (img.Directory.Parent.Name == "Google Set" || img.Directory.Parent?.Name == "Cohn-Kanade Images" || img.Directory.Parent?.Name == "CK+")
        {
            return GetLabelGoogleSetOrCohnKanade(img.Directory.Name);
        }
        else
        {
            return GetLabelMUG(img.Name);
        }
    }

    /// <summary>
    /// Get the emotion from an images file name.
    /// </summary>
    /// <param name="fileName"></param>
    /// <returns></returns>
    static string GetLabelMUG(string fileName)
    {
        if (fileName.Contains("an"))
        {
            return "angry";
        }
        else if (fileName.Contains("di"))
        {
            return "disgusted";
        }
        else if (fileName.Contains("fe"))
        {
            return "fear";
        }
        else if (fileName.Contains("ha"))
        {
            return "happy";
        }
        else if (fileName.Contains("sa"))
        {
            return "sad";
        }
        else if (fileName.Contains("ne"))
        {
            return "neutral";
        }
        else if (fileName.Contains("su"))
        {
            return "surprise";
        }

        throw new Exception("This image was incorrectly assigned, or maybe you're using the function with the wrong dataset," +
        "it is only for classifying images for the MUG dataset!");
    }

    /// <summary>
    /// Get the emotion of the emotion based on the parent directory of the image.
    /// </summary>
    /// <param name="parentDirectoryName"></param>
    /// <returns></returns>
    static string GetLabelGoogleSetOrCohnKanade(string parentDirectoryName)
    {
        if (parentDirectoryName.Contains("anger"))
        {
            return "angry";
        }
        else if (parentDirectoryName.Contains("disgust"))
        {
            return "disgusted";
        }
        else if (parentDirectoryName.Contains("fear"))
        {
            return "fear";
        }
        else if (parentDirectoryName.Contains("happy") || parentDirectoryName.Contains("joy") || parentDirectoryName.Contains("happiness"))
        {
            return "happy";
        }
        else if (parentDirectoryName.Contains("sadness"))
        {
            return "sad";
        }
        else if (parentDirectoryName.Contains("neutral"))
        {
            return "neutral";
        }
        else if (parentDirectoryName.Contains("surprise"))
        {
            return "surprise";
        }

        throw new Exception("This image was incorrectly assigned, or maybe you're using the function with the wrong dataset," +
                            "it is only for classifying images for the Google Set or Cohn-Kanade image set!");
    }

    /// <summary>
    /// Extract features from an image and store it in <see cref="FaceData3"/>.
    /// </summary>
    /// <param name="imageFileInfo">File info of the image.</param>
    /// <param name="sp"></param>
    /// <param name="fd"></param>
    /// <param name="getLabel">Whether to get the label or not. False if not using for prediction.</param>
    /// <returns></returns>
    /// <seealso cref="GetFaceDataPoints3"/>
    static FaceData3 GetFaceData3FromImage(FileInfo imageFileInfo, ShapePredictor sp, FrontalFaceDetector fd,
        bool getLabel = true)
    {
        using (var img = Dlib.LoadImage<RgbPixel>(imageFileInfo.FullName))
        {
            var faces = fd.Operator(img);
            foreach (var face in faces)
            {
                var shape = sp.Detect(img, face);
                try
                {
                    return GetFaceDataPoints3(ref shape,
                        getLabel
                            ? GetLabel(imageFileInfo)
                            : "Not getting label, see argument this function was called with.");
                }
                catch
                {
                    return null;
                }
            }
        }

        Debug.WriteLine($"Unable to get facial feature from {imageFileInfo.Name} as no faces were found!");
        return null;
    }

    /// <summary>
    /// Extract features from an image and store it in <see cref="FaceData2"/>.
    /// </summary>
    /// <param name="imageFileInfo">File info of the image.</param>
    /// <param name="sp"></param>
    /// <param name="fd"></param>
    /// <param name="getLabel">>Whether to get the label or not. False if not using for prediction.</param>
    /// <returns></returns>
    /// <seealso cref="GetFaceDataPoints2"/>
    static FaceData2 GetFaceData2FromImage(FileInfo imageFileInfo, ShapePredictor sp, FrontalFaceDetector fd,
        bool getLabel = true)
    {
        using (var img = Dlib.LoadImage<RgbPixel>(imageFileInfo.FullName))
        {
            var faces = fd.Operator(img);
            foreach (var face in faces)
            {
                var shape = sp.Detect(img, face);

                try
                {
                    return GetFaceDataPoints2(ref shape,
                        getLabel
                            ? GetLabel(imageFileInfo)
                            : "Not getting label, see argument this function was called with.");
                }
                catch
                {
                    return null;
                }
            }
        }

        Debug.WriteLine($"Unable to get facial feature from {imageFileInfo.Name} as no faces were found!");
        return null;
    }

    /// <summary>
    /// Extract features from an image and store it in <see cref="FaceData1"/>.
    /// </summary>
    /// <param name="imageFileInfo">File info of the image.</param>
    /// <param name="sp"></param>
    /// <param name="fd"></param>
    /// <param name="getLabel">>Whether to get the label or not. False if not using for prediction.</param>
    /// <returns></returns>
    /// <seealso cref="GetFaceDataPoints1"/>
    static FaceData1 GetFaceData1FromImage(FileInfo imageFileInfo, ShapePredictor sp, FrontalFaceDetector fd, bool getLabel = true)
    {
        // load input image
        using (var img = Dlib.LoadImage<RgbPixel>(imageFileInfo.FullName))
        {
            var faces = fd.Operator(img);
            foreach (var face in faces)
            {
                var shape = sp.Detect(img, face);

                try
                {
                    return GetFaceDataPoints1(ref shape,
                        getLabel
                            ? GetLabel(imageFileInfo)
                            : "Not getting label, see argument this function was called with.");
                }
                catch
                {
                    return null;
                }
            }
        }
        Debug.WriteLine($"Unable to get facial feature from {imageFileInfo.Name} as no faces were found!");
        return null;
    }

    /// <summary>
    /// Extract the features from a shape and place it into the <see cref="FaceData1"/>.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="label">The emotion/label of the face.</param>
    /// <returns></returns>
    static FaceData1 GetFaceDataPoints1(ref FullObjectDetection shape, string label)
    {
        var leftEye = GetTotalFeatureLength(shape.GetPart(39),
            new Point[]
            {
                shape.GetPart(18), shape.GetPart(19), shape.GetPart(20)
            },
            shape.GetPart(21));
        var rightEye = GetTotalFeatureLength(shape.GetPart(42),
            new Point[]
            {
                shape.GetPart(23), shape.GetPart(24), shape.GetPart(25)
            },
            shape.GetPart(22));

        var leftLip = GetTotalFeatureLength(shape.GetPart(33),
            new Point[] { shape.GetPart(48), shape.GetPart(49), shape.GetPart(50) },
            shape.GetPart(51));
        var rightLip = GetTotalFeatureLength(shape.GetPart(33),
            new Point[] { shape.GetPart(52), shape.GetPart(53), shape.GetPart(54) },
            shape.GetPart(51));

        var lipHeight = (shape.GetPart(57) - shape.GetPart(51)).Length /
                        (shape.GetPart(33) - shape.GetPart(51)).Length;
        var lipWidth = (shape.GetPart(54) - shape.GetPart(48)).Length /
                       (shape.GetPart(33) - shape.GetPart(51)).Length;

        return new FaceData1()
        {
            Emotion = label,
            LeftEyebrow = (float)leftEye,
            LeftLip = (float)leftLip,
            LipWidth = (float)lipWidth,
            RightEyebrow = (float)rightEye,
            RightLip = (float)rightLip,
            LipHeight = (float)lipHeight
        };
    }

    /// <summary>
    /// Extract the features from a shape and place it into the <see cref="FaceData2"/>.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="label">The emotion/label of the face.</param>
    /// <returns></returns>
    static FaceData2 GetFaceDataPoints2(ref FullObjectDetection shape, string label)
    {
        //http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/#more-565

        float avgx = 0f, avgy = 0f;
        float[] x = new float[shape.Parts];
        float[] y = new float[shape.Parts];

        Point[] distToCentres = new Point[shape.Parts];
        for (uint i = 0; i < shape.Parts; ++i)
        {
            avgx += shape.GetPart(i).X;
            x[i] = shape.GetPart(i).X;
            avgy += shape.GetPart(i).Y;
            y[i] = shape.GetPart(i).Y;
        }
        avgx /= shape.Parts;
        avgy /= shape.Parts;

        for (var i = 0; i < distToCentres.Length; i++)
        {
            distToCentres[i] = new Point(Convert.ToInt32(x[i] - avgx), Convert.ToInt32(y[i] - avgy));
        }

        FaceData2 fd = new FaceData2();

        // Get angle
        var middlePoint = shape.GetPart(33);
        var topNasalPoint = shape.GetPart(27);

        for (uint i = 0; i < shape.Parts; ++i)
        {
            fd.Emotion = label;
            fd.RawCoordiantesX[i] = x[i];
            fd.RawCoordiantesY[i] = y[i];
            var distance = (new Point((int)avgx, (int)avgy) - new Point((int)x[i], (int)y[i])).LengthSquared;
            fd.LengthBetweenFeatures[i] = (float)distance;
            fd.AngleBetweenFeatures[i] = Convert.ToSingle(Math.Atan2(y[i], x[i]) * 360 / (2 * Math.PI));
        }

        return fd;
    }

    /// <summary>
    /// Extract the features from a shape and place it into the <see cref="FaceData3"/>.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="label">The emotion/label of the face.</param>
    /// <returns></returns>
    static FaceData3 GetFaceDataPoints3(ref FullObjectDetection shape, string label)
    {
        // Get average point
        float avgx = 0f, avgy = 0f;
        for (uint i = 0; i < shape.Parts; ++i)
        {
            avgx += shape.GetPart(i).X;
            avgy += shape.GetPart(i).Y;
        }
        avgx /= shape.Parts;
        avgy /= shape.Parts;

        // Get normalization Distance
        var middle = new Point((int)avgx, (int)avgy);
        var normalization = (float)(shape.GetPart(27) - middle).LengthSquared;

        FaceData3 fd3 = new FaceData3();
        fd3.Emotion = label;
        fd3.LeftEyebrowDistance = (GetDistance(ref shape, middle, 17, normalization) +
                                   GetDistance(ref shape, middle, 18, normalization) +
                                   GetDistance(ref shape, middle, 18, normalization) +
                                   GetDistance(ref shape, middle, 19, normalization) +
                                   GetDistance(ref shape, middle, 20, normalization) +
                                   GetDistance(ref shape, middle, 21, normalization)) / 5;
        fd3.RightEyebrowDistance = (GetDistance(ref shape, middle, 22, normalization) +
                                    GetDistance(ref shape, middle, 23, normalization) +
                                    GetDistance(ref shape, middle, 24, normalization) +
                                    GetDistance(ref shape, middle, 25, normalization) +
                                    GetDistance(ref shape, middle, 25, normalization)) / 5;
        fd3.LeftEyeWidth = GetDistanceBetween(ref shape, 36, 39, normalization);
        fd3.RightEyeWidth = GetDistanceBetween(ref shape, 42, 45, normalization);
        fd3.LeftEyeHeight = GetDistanceBetween(ref shape, 40, 38, normalization);
        fd3.RightEyeHeight = GetDistanceBetween(ref shape, 46, 44, normalization);
        fd3.OuterLipWidth = GetDistanceBetween(ref shape, 48, 54, normalization);
        fd3.InnerLipWidth = GetDistanceBetween(ref shape, 60, 64, normalization);
        fd3.OuterLipHeight = GetDistanceBetween(ref shape, 52, 58, normalization);
        fd3.InnerLipHeight = GetDistanceBetween(ref shape, 63, 67, normalization);

        fd3.LeftLipEdgeAngle = GetAngleBetween(ref shape, 48, middle);
        fd3.RightLipEdgeAngle = GetAngleBetween(ref shape, 54, middle);

        return fd3;
    }

    /// <summary>
    /// Get distance between a point on a face by using the point index and middle of face, and divide it by the normalization value.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="middle"></param>
    /// <param name="pointIndex"></param>
    /// <param name="normalization"></param>
    /// <returns></returns>
    static float GetDistance(ref FullObjectDetection shape, Point middle, uint pointIndex, float normalization)
    {
        return (float)(shape.GetPart(pointIndex) - middle).LengthSquared / normalization;
    }

    /// <summary>
    /// Get distance between two points on a face, using their index.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="pointIndexA"></param>
    /// <param name="pointIndexB"></param>
    /// <param name="normalization"></param>
    /// <returns></returns>
    static float GetDistanceBetween(ref FullObjectDetection shape, uint pointIndexA, uint pointIndexB,
        float normalization)
    {
        return (float)(shape.GetPart(pointIndexB) - shape.GetPart(pointIndexA)).LengthSquared / normalization;
    }

    /// <summary>
    /// Get the angle between an index and a point in radians.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="pointIndex"></param>
    /// <param name="point"></param>
    /// <returns></returns>
    static float GetAngleBetween(ref FullObjectDetection shape, uint pointIndex, Point point)
    {
        var pointI = shape.GetPart(pointIndex);

        var vectorBetweenPoints = pointI - point;
        var vectorBetweenPointsNormalized = new Vector2((float)(vectorBetweenPoints.X / vectorBetweenPoints.Length), (float)(vectorBetweenPoints.Y / vectorBetweenPoints.Length));

        return (float)Math.Atan2(vectorBetweenPointsNormalized.Y, vectorBetweenPointsNormalized.X) /** (180f / (float)Math.PI)*/;
    }

    /// <summary>
    /// Get file around the current executables' directory.
    /// </summary>
    /// <param name="fileName">File name with extension.</param>
    /// <returns>FileInfo of the found file.</returns>
    public static FileInfo GetFile(string fileName, bool throwExceptionIfNotFound = true)
    {
        DirectoryInfo currentDir = new DirectoryInfo(Environment.CurrentDirectory);
        int attempts = 0;
        const int maxUpDirectories = 5;
        do
        {
            var foundFileInfo = currentDir.GetFiles().ToList().Find(file => file.Name == fileName);
            if (foundFileInfo == null)
            {
                attempts++;
                currentDir = currentDir.Parent;
            }
            else
            {
                return foundFileInfo;
            }

        } while (attempts < maxUpDirectories);

        // To help fix issue.
        if (fileName == ShapePredictorFileName)
        {
            MessageBox.Show(
                $"You need to unzip the file {ShapePredictorFileName.Remove(ShapePredictorFileName.Length - 4)}.zip before continuing. The program will now exit.", "Error Finding Shape Predictor", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        if (throwExceptionIfNotFound)
        {
            throw new FileNotFoundException(
                $"File {fileName} was not found inside around the executale files' directory. Try placing it no more than {maxUpDirectories} directories above where the executable is located.");
        }
        else
        {
            return null;
        }
    }

    /// <summary>
    /// Get a directory in or above where the executale file is located.
    /// </summary>
    /// <param name="directoryName">Name of the directory.</param>
    /// <returns><see cref="DirectoryInfo"/> of the found directory.</returns>
    public static DirectoryInfo GetDirectory(string directoryName)
    {
        DirectoryInfo currentDir = new DirectoryInfo(Environment.CurrentDirectory);
        int attempts = 0;
        const int maxUpDirectories = 4;
        do
        {
            var foundDirInfo = currentDir.GetDirectories().ToList().Find(dir => dir.Name == directoryName);
            if (foundDirInfo == null)
            {
                attempts++;
                currentDir = currentDir.Parent;
            }
            else
            {
                return foundDirInfo;
            }

        } while (attempts < maxUpDirectories);

        throw new FileNotFoundException(
            $"Directory {directoryName} was not found around the executable files' directory. Try placing it no more than {maxUpDirectories} directories above where the executable is located.");

    }

    /// <summary>
    /// Get the start of the feature extraction file name depending on type, appends extension.
    /// </summary>
    /// <param name="TFaceData"></param>
    /// <returns></returns>
    public static string GetFacialFeaturesFileName(Type TFaceData)
    {
        const string originalFacialFeaturesFileName = "FeatureVector";
        return originalFacialFeaturesFileName + TFaceData.ToString() + ".csv";
    }

    /// <summary>
    /// Get the name of the zip according to the type passed in, with extension.
    /// </summary>
    /// <param name="TFaceData"></param>
    /// <returns></returns>
    public static string GetModelZipFileName(Type TFaceData)
    {
        const string modelZipFileName = "model";
        return modelZipFileName + TFaceData.ToString() + ModelZipFileExtension;
    }

    /// <summary>
    /// Get the filename of the test data view depending on the type of face data, with extension.
    /// </summary>
    /// <param name="TFaceData"></param>
    /// <returns></returns>
    private static string GetTestSetDataViewFileName(Type TFaceData)
    {
        const string testSetDataViewFileName = "testsetdataview";
        return testSetDataViewFileName + TFaceData.ToString() + DataViewFileExtension;
    }

    /// <summary>
    /// Get the name fo the train set file name depending on the type passed in, with extension.
    /// </summary>
    /// <param name="TFaceData"></param>
    /// <returns></returns>
    private static string GetTrainSetDatViewFileName(Type TFaceData)
    {
        const string trainSetDataViewFileName = "trainsetdataview";
        return trainSetDataViewFileName + TFaceData.ToString() + DataViewFileExtension;
    }
}
