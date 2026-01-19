## Core Question

Can a neural network learn (through GA or ES) to make use of tools that provided to it to do things like high-precision math, if these tools are interfaced directly with the nodes in its network?

## Background

Deep neural networks, including both transformers and other architectures, can learn to do things like modular arithmetic.  In so doing, they develop specialized neural "circuits" that approximate numeric rings and other such useful functions.

If we know such functions are useful, we ought to be able to provide a network a direct neural interface to tools that compute them directly, making the task much easier for the network to learn and resulting in more reliable, higher-capacity behavior.  For example, we could have an arithmetic module that takes in some neural encoding of input numbers and operators, and produces a similar encoding of the mathematically correct answer.  A network should be able to use this in a very natural way, whenever it is useful to the task at hand.

Such a module would not be differentiable, and so inserting it into the network architecture probably precludes the use of back propagation.  But GA and ES have been shown to be effective even for large neural networks.  So, assume we use those to train the network.


## Open Questions

- What previous research has looked at this issue?
- How should the inputs and outputs of such a module be encoded, and how can we ensure the network will discover this?
- What network architecture(s) should we try?
- Under what circumstances will the network actually learn to use the provided module, and when will it simply ignore it?
- How can we demonstrate all this in a toy-sized problem, and how might we scale it up?



