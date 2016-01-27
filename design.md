##Design overview

Tugboat is a model-serving system written in Python to support out of the box
serving of a wide variety of machine-learning models.

Tugboat is focused on three core design goals:

+ __Better wrong than late:__ Tugboat aims to serve predictions from a variety of models
with low-latency, gracefully trading off latency and accuracy when it is impossible to
achieve both. Tugboat provides a set of model-oblivious strategies to facilitate this
tradeoff, as well as being able to take advantage of additional model-specific information
where available.
+ __Collaboration is key to success__:
+ __Accuracy requires adaptivity and robustness__:


Within this context, Tugboat is an easy-to-use, low-latency serving system
primarily directed at serving `scikit-learn` and TensorFlow models. However,
through the use of Python's foreign function interface C-bindings, it
can be effectively used to call models from many other languages.



Tugboat has two APIs.
