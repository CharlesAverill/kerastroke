# KeraStroke

KeraStroke is a [Python package](https://pypi.org/project/kerastroke/#description) that implements 
"Post-Back-propagation Weight Operations", or "PBWOs"; generalization-improvement techniques for Keras models in the 
form of custom Keras Callbacks. These techniques function 
similarly but have different philosophies and results. The techniques are:
- Stroke: Re-initializaing random weight/bias values.
- Pruning: Reducing model size by setting weight/bias values that are close to 0, to 0.
- NeuroPlast: Re-initializing any weight/bias values that are 0 or close to 0.

Stroke is modeled after seizures, which send random electrical signals throughout the brain, sometimes causing damage 
to synapses. 

NeuroPlast is modeled after the concept of neuroplasticity, when neurons that no longer have a primary function begin 
to rewire to improve another function. I started working on NeuroPlast after I read the work done by Blakemore and 
Cooper on horizontal/vertical line receptor neurons in the brains of cats. 

If you'd like to see the tests I'm performing with KeraStroke, you can view my testing repository 
[here](https://github.com/CharlesAverill/stroke-testing).

KeraStroke 2.0.0 marks when I really started putting work into the project. I've made an effort to comment more, 
clean my code up, and make the package easier to understand overall without sacrificing utility.

# Limitations
KeraStroke is still in the development phase. Heavy testing has been done on 
Dense nets, but little testing has been done on CNNs and no testing has been done on RNNs. As of 2.1.0, CNNs are 
functioning properly in KeraStroke! The issue with previous versions had to do with the way the callback would retrieve
 the weights from the models. The callbacks perform significantly better on DenseNets, but could still find use in CNNs.
 I'm working on this, but will definitely need the help. Please see
 [the github page](https://github.com/CharlesAverill/stroke-testing/) or 
 [contact me](https://mail.google.com/mail/?view=cm&fs=1&to=charlesaverill20@gmail.com) to contribute to the project.

# Stroke
The goal of the Stroke callback is to re-initialize weights/biases that have begun to contribute to overfitting.

Parameters:

 - `set_value`: re-initialized weights will be set to this value, rather than a random one
 - `low_bound`: low bound for weight re-initialization
 - `high_bound`: high bound for weight re-initialization
 - `volatility_ratio`: percentage of weights to be re-initialized
 - `cutoff`: number of epochs to perform PBWOs
 - `decay`: Every epoch, v_ratio is multiplied by this number. decay can be greater than 1.0,
                      but v_ratio will never exceed 1.0
 - `do_weights`: perform stroke on weights
 - `do_biases`: perform stroke on biases

# Pruning
The goal of the Pruning callback is to nullify weights/biases that are effectively 0.

Parameters:

 - `set_value`: The value that pruned weights will be set to
 - `min_value`: The lowest value a weight/bias can be to be oeprated on
 - `max_value`: The highest value a weight/bias can be to be operated on
 - `cutoff`: number of epochs to perform PBWOs
 - `do_weights`: perform pruning on weights
 - `do_biases`: perform pruning on biases

# NeuroPlast
The goal of the NeuroPlast callback is to randomly re-initialize weights/biases that are effectively 0.

Parameters:

 - `set_value`: re-initialized weights will be set to this value, rather than a random one
 - `min_value`: lowest value a weight/bias can be to be operated on
 - `max_value`: highest value a weight/bias can be to be operated on
 - `low_bound`: low bound for weight re-initialization
 - `high_bound`: high bound for weight re-initialization
 - `cutoff`: number of epochs to perform PBWOs
 - `do_weights`: perform neuroplast on weights
 - `do_biases`: perform neuroplast on biases

# Usage
KeraStroke Callbacks can be used like any other custom callback. Here's a basic example:

```python
from kerastroke import Stroke
model.fit(X, y, 
          epochs=32, 
          callbacks=[Stroke()])
```
