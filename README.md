# Emotion Recognition
A C#/ML.NET application which uses machine learning and facial recognition to identify the emotion of a face. Implemented in C# using ML.NET and Dlib for feature extraction. Made as coursework for a module in university.

![UI](https://raw.githubusercontent.com/giodestone/Emotion-Recognition/main/Images/Image1.jpg)

## Running
[Download](https://github.com/giodestone/Emotion-Recognition/releases)

The Cohn-Kanade set must be downloaded and placed in the same folder as the `.exe`. It can be requested from [here](http://www.pitt.edu/~emotion/ck-spread.htm) or downloaded from [here](https://github.com/spenceryee/CS229). Other data sets works, such as [MUG](https://mug.ee.auth.gr/fed/). Generally any face data set with people facing forwards will work.

## Using Source
The `shape_predictor_68_face_landmarks.zip` inside of `Source/` must be extracted into the same destination when received.

## Implementation
*The [full report](https://github.com/giodestone/Emotion-Recognition/raw/main/Report.pdf) is available for more information, including: how feature extraction works, and the improvements described in detail.*

The user interface is implemented using Windows Forms.

The facial features (distance between eyebrows and nose, lip distance between nose etc.) are extracted from the Cohn-Kanade face data set using Dlib. These features are then normalized versus the size of the nose and stored in a feature vector.

The model trained using the SCDA Maximum Entropy trainer, as it determines importance of a feature towards a label, individually.

![Confusion Matrix](https://raw.githubusercontent.com/giodestone/Emotion-Recognition/main/Images/ConfusionMatrix.jpg)

The current implementation is relatively simple and only recognizes faces from a single data set, resulting in a relatively poor facial recognition performance. This could be improved by:

### Potential Improvements
* Flip and mirror images.
* More training epochs.
* Optimise feature extraction stage to reduce time taken.
* More features to help in differentiating anger and sadness.
* More samples.
* Webcam support.
* Different trainer.