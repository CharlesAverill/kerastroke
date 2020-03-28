# KeraStroke

KeraStroke is a [Python package](https://pypi.org/project/kerastroke/#description) that implements generalization-improvement techniques for Keras models in the form of custom Keras Callbacks. These techniques function similarly but have different philosophies and results. The techniques are:
- Stroke: Re-initializaing random weight/bias values.
- Pruning: Reducing model size by setting weight/bias values that are close to 0, to 0.
- NeuroPlast: Re-initializing any weight/bias values that are 0 or close to 0.

Stroke and NeuroPlast are ideas of my own, I drew inspiration from the human brain. 

Stroke is modeled after seizures, which send random electrical signals throughout the brain, sometimes causing damage to synapses. 

NeuroPlast is (unsurprisingly) modeled after the concept of neuroplasticity, when neurons that no longer have a primary function begin to rewire to improve another function. I started working on NeuroPlast after I read the work done by Blakemore and Cooper on horizontal/vertical line receptor neurons in the brains of cats. 

Keep in mind that using any KeraStroke Callback on large models can introduce serious slowdown during training.

If you'd like to see the tests I'm performing with KeraStroke, you can view my testing repository [here](https://github.com/CharlesAverill/stroke-testing).

KeraStroke 2.0.0 marks when I really started putting work into the project. I've made an effort to comment more, clean my code up, and make the package easier to understand overall without sacrificing utility.

# Limitations
KeraStroke is still in the development phase, and not advised for general use yet. Heavy testing has been done on Dense nets, but little testing has been done on CNNs and no testing has been done on RNNs. Currently CNNs are not supported at all. I'm working on this, but will definitely need the help. Please see [the github page](https://pypi.org/project/kerastroke/#description) or [contact me](charlesaverill20@gmail.com) to contribute to the project.

# Stroke
The goal of the Stroke callback is to re-initialize weights/biases that have begun to contribute to overfitting.

Parameters of the callback are:

- value: If set, re-initialized weights will be set to this value, rather than a random one
- low_bound: When weights are re-initialized, this will be their lowest possible value
- high_bound: When weights are re-initialized, this will be their highest possible value
- volatility_ratio: The percentage of weights that will be re-initialized after every epoch
- cutoff: The number of epochs that Stroke will be performed
- decay: Every epoch, v_ratio is multiplied by this number. decay can be greater than 1.0, but v_ratio will
       never exceed 1.0
- indeces: A list of integers specifying the indeces of layers that Stroke will be performed on, rather than
         the model as a whole
- biases: If true, Stroke will also be performed on biases

# Pruning
The goal of the Pruning callback is to nullify weights/biases that are effectively 0.

Parameters of the callback are:

- value: The value that pruned weights will be set to. (this should usually be 0.0)
- min_value: The lowest value a weight can be to be oeprated on
- max_value: The highest value a weight can be to be operated on
- cutoff: The number of epochs that Pruning will be performed
- biases: If true, Pruning will also be performed on biases

# NeuroPlast
The goal of the NeuroPlast callback is to randomly re-initialize weights/biases that are effectively 0.

Parameters of the callback are:

- value: The value that pruned weights will be set to. (this should usually be 0.0)
- min_value: The lowest value a weight/bias can be to be operated on
- max_value: The highest value a weight/bias can be to be operated on
- pruning_min: The lowest value a weight can be to be pruned (this should usually be 0.0)
- pruning_max: The highest value a weight can be to be pruned
- cutoff: The number of epochs that Pruning will be performed

# Usage
KeraStroke Callbacks can be used like any other custom callback. Here's a basic example:

```python
from kerastroke import Stroke
model.fit(X,
          y, 
          epochs=32, 
          callbacks=[Stroke()])
```
