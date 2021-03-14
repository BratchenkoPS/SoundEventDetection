# SED
This repo is a solution to a test where you have to demonstrate ability to solve the task of
sound event detection using [ESC-50 dataset](https://github.com/karolpiczak/ESC-50#download) as a train
data and [DESED dataset](https://github.com/turpaultn/DESED) as a validation data.
## 1) Model choice and general approach
So, there are two main metrics in sound event detection systems used in DESED:
1) Event-based measures with a 200 ms collar on onsets and a 200 ms / 20% of the events length collar 
on offsets
2) Segment-based F1-score on 1 s segments

We going to use the second approach and treat the task as a simple multilabel classification
problem, just splitting both datasets into chunks of 1 second long recordings. For the model we going
to use resnet18 as a strong baseline, replacing first convolutional layer to work with 1 channel
data and fully-connected layer to fit into our number of classes. Melspectrograms will be our main features.
 We also have to map the classes
from ESC-50 into DESED, so we end up with just 5 classes total (you can take a look at those in
**config.yml**). We also going to do some oversampling to increase the amount of data
by cutting files into 1 sec chunks starting from different point (e.g. **0 : 1** sec, **0.1 : 1.1**
 sec, etc)

The pipeline is going to like this:
1) Download and unpack data
2) Load the DESED data, pick the classes we using, split it into 1 second chunk with the respect
 of the multilabel samples (we end up with only 1.78% of multilabel samples)
3) Same for ESC-50 with oversampling to build a train data (we not building any multilabel example due to 
how little there are in the test set and lack of time)
4) Build a custom torch dataset and model
5) Train it and see the results.

## 2) Result
So as the result we can achieve macro f1 score of 0.497. It's not very impressive, considering
that the DESED baseline is around 0.7. But that baseline uses way bigger model and way more data for
the task, so that's kind of expected.

## 3) Improvements
1) Get more data! This is the most important one, because before oversampling we had less training
data than test. Although the point of task was to build a model using *only* ESC-50 dataset
2) Use different model. Looking back i tend to realise that the best way would probably be using 
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) or 
[attention CNN](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)
but those two a kind of hard to digest if it's your first time working with audio.
3) Augmentations: Noise injection, pitch shifting, changing speed, etc. Sadly we didn't had much time to do so

## 4) How to run
```git clone https://github.com/BratchenkoPS/SED``` <br />
```cd SED``` <br />
```python -m venv /venv``` <br />
```pip install -r requirements.txt``` <br />
```python main.py```