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
**config.yml**)

The pipeline is going to like this:
1) Download and unpack data
2) Load the DESED data, pick the classes we using, split it into 1 second chunk with the respect
 of the multilabel samples (we end up with only 1.78% of multilabel samples)
3) Same for ESC-50 to build a train data
4) Build a custom dataset and model
5) Train it and see the results.