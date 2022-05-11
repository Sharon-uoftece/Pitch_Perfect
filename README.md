# Pitch-Perfect
A ML model that recognizes pitch of music notes played from various instruments
---

## Intro
Imagine you are in a coffee shop, and suddenly you hear a song that hits your sweet spot. This happens in many people’s daily lives, including ours. Therefore, we are motivated to make music identification easier for music enthusiasts like ourselves.

Our goal is to create a tool that can quickly and precisely recognize any note played by any instrument, we call it the “Perfect Pitch AI”. Our team believes that our model has many different potential applications. People may utilize our project by inputting several hundreds of snapshots of a song, and they will receive the overall note distribution of the song, and since it’s hardly ever possible for two different songs to have the same sequence of notes, this data may be further used for training, for example, a song-recognition model. Google’s song search by humming is another idea. 

Machine learning is a reasonable approach because this can be seen as a classification problem. Different notes have different sounds that can generate different sound-wave diagrams which then can be used to train/test our model. It’s similar to identifying gestures, and although these “musical gestures” are more complex and are harder to visualize, it’s the same process of letting the neural network see multiple snapshots of different notes and generalize the overall distribution and curve of different wave diagrams.

Our model will be classifying 88 pitches from pitch 21 to pitch 108 of the MIDI standard from isolated samples of notes. 

---

## Background 
Music consists of 12 distinct pitch classes(A A# B C C# D D# E F F# G G#) that are separated by their frequencies. 

For example, the frequency of an A4 note is 440 Hz. However, there are many A notes, such as A5 note at 880 Hz and A3 note at 220 Hz. They sound similar and their waveforms look similar. 

---

## Related Work
One related application is tuning software. Tuning software analyzes the waveforms of the sound. They sample the instrument over a window of time and perform a fourier transformation on the waveform to get a frequency amplitude function of the samples. Since there may be noise surrounding the note, the frequency amplitude plot won’t be just a single frequency. Instruments don’t produce a single frequency but a multitude of frequencies (aka harmonics). The fourier transformation is transformed again through the Harmonic Product Spectrum, which estimates the fundamental frequency (the peak with the lowest frequency) or the note itself. 

Another application is called chordify.net. It finds the chords and the beats at which they occur at. They use deep learning for both beat recognition and chord recognition, where chords are defined by their fundamental pitch. They used spectrograms of songs labelled with chords to train their models.

---

## Data Processing
### Data Metrics
The data set that we use is called NSynth, a high quality set of isolated, 4-second music notes samples annotated and recorded by Google’s machine learning team Magenta. The original data set contains 1006 instruments, categorized by instrument family. The pitch distribution of the original data set is already imbalanced. As a result, we chose to take 1800 random samples from pitch 21 to pitch 108, to ensure a balanced distribution. 
NSynth and the used set’s instrument distribution is imbalanced. The team did not take special care in balancing the instrument family because we hoped the model would be able to learn a pitch embedding disregarding the instrument itself. The length at which the note was held also varied with time, but was stated to end at around 3 seconds generally. 

### Preprocessing
We took a spectrogram of each sample. A spectrogram measures the strength of each frequency at point and time, making it similar to a heap map. Images were generated using Librosa, by converting the .wav files into spectrograms using Short-Term Fourier Transform (STFT). Amplitudes were then converted to decibel scale. Images are resized to 224 x 224 pixels to match AlexNet.

### Data Splits
NSynth’s validation set was used directly for validation (12732 samples). 
NSynth’s test set was used directly for testing (4293 samples).

---

## Architecture
The primary model accepts images of dimensions 3x224x224. The image is passed through alexnet.features for feature extraction, then passed through to fully-connected layers (fig X). It was trained on an initial batch size of 188, learning rate of 1e-5, and 18 epochs. We multiplied the batch size by 1.1 times for every epoch loop after epoch 5. This acts like a various learning rate where at the beginning we want noisy gradients to jump out of local minimums and at the end we want gradients to be more accurate so that we can converge to the minimum.

---

## Baseline Model
Choosing a reasonable baseline model is an important step in a machine learning problem. After we have done a lot of research on choosing the best and most suitable baseline model for our project and comparing the pros and cons of each doable baseline model. Our group shortlisted ANN,CNN and random forest classifier as baseline model choices. We decided to choose a random forest classifier as our final baseline model since it works well with a large amount of data and our project can be interpreted as a classification problem.

 Random forest classifier is a method of classification. It can construct different trees at training time, and output the regression of individual trees. After adjusting the number of estimators to 200, which is the number of total trees, the max-depth to 6, which is the maximum number of branches in each tree, the minimum sample split to 10, which can create arbitrary small leaves. The baseline model can achieve a training accuracy of 42.73%, which is a solid number for the baseline model. 

We have also created a CNN baseline model without optimization and parameters tuning. This model can only achieve an overall accuracy of 37.8% at its best. Therefore, our group concluded that a random forest classifier is suitable for our project since it uses an ensemble method  to construct multiple decision making trees.

---

## Quantitative Result
Our model achieves 92.4% training accuracy, 82.8% validation accuracy and 80.4% testing accuracy. The model took over 10 hours to train for 18 epochs, and google-colab kept disconnecting before the model was fully trained. Therefore, the team could not obtain the whole training curve for the model. However, the validation loss dropped to a minimum at epoch 18 while the accuracy was the highest, and thus the team decided to use the model that was trained for 18 epochs. Since the purpose of the model is to identify different pitches, the team balanced the data set in terms of pitch. However, we recognized that the data that we use varied in many areas which made the prediction much more challenging. For example, the data varied in instruments, it also varied in instrument families. For example, keyboards can be acoustic, electronic or synthetic. The data also varied in length. Some samples contain three second of silence and only one second of actual sound. Therefore, the team decided to try testing with some tolerance. The tolerance would be three pitches, which means the model predicted pitch 70 and the actual pitch is 67, we considered the model to be correct. It turned out the testing accuracy with tolerance increased significantly to 88.7% which shows that the model is predicting on the right track and the predictions are reasonable and logical. 

---

## Qualitative Result
Model performed worse on lower pitch notes. We believe that this is due to the overlapping of tones, as pitches become lower, the strings required to play them become thicker, and many overtones can become dominant in the sound and overwhelm the fundamental tone.

Another result is that the model performed worse on certain instruments. This is due to the imbalance of the data set in terms of instrument and instrument family. For example, the training set contained only under 10000 flute samples while it contained over 60000 bass samples.

---

## Evaluate Model on New Data
### Applying Model to real life samples
To evaluate the model, we first recorded audio samples of three different pitches, using an online virtual keyboard. Each audio sample lasts for approximately four seconds, same as the samples in the NSynth data set. Then we feed these audio samples into a program that we have written for transforming .wav files to spectrograms. After we have obtained the newly generated spectrograms, we fed them into our model to see if our model can correctly predict all the labels. The correct labels for each pitch could be found using the MIDI chart. 

### Applying Model to another large dataset
Dataset Description
TinySOL dataset was used, containing 2478 samples recorded in 1990’s of various instruments. Most instrument classes are a part of NSynth however, specific instruments aren't specified in NSynth. Please note that the instruments such as Accordion are not found within the instrument classes of NSynth. And the dataset also includes varying lengths of data (from 3s - 8s), thus making it a good choice for testing on the model’s ability to generalize. 

Bass Tuba
French Horn
Trombone
Trumpet in C
Accordion
Contrabass
Violin
Viola
Violoncello
Bassoon
Clarinet in B-flat
Flute
Oboe
Alto Saxophone

List 1: List of Instruments

Performance on TinySol
Our model obtained a very low accuracy on this dataset of about 1%. This illustrates the limitations of our model. The data is varying in length and also has a different sampling rate. Moreover, the data is very noisy as it was recorded a long time ago. All of which are aspects that our model can improve on in the future. 

