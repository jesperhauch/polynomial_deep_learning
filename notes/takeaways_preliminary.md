# Single variable
* Feedforward Neural Network with ReLU
    * Model is great at learning in distribution but cannot generalize OOD.
* Polynomial NN 
    * Model with one layer can learn OOD if n_degree (transfer) >= n_degree (polynomial)
    * Model with two layers cannot learn OOD if n_degree (transfer) < n_degree (polynomial) 
* CCP 
    * Model can learn OOD if n_layers >= n_degree

# Two variable
* Generally
    * Did not seem much harder to learn than a single variable
* CCP
    * Model can learn for multiple features
* Polynomial NN
    * Model with one layer can learn OOD if n_degree (transfer) >= n_degree (polynomial)

# Two variable interaction
* Generally
    * Interactions are harder to learn
* CCP
    * Model can learn interactions OOD at 30 epochs (still learning) if n_degree >= n_degree (polynomial)
* Polynomial NN
    * Single layer - Model can learn interactions OOD but learning starts to stagnate 
    * Two layer - Model cannot learn interactions and produce really large losses (maybe we should have ReLUs and then polynomial transfer for final layer?)

# Two variable non-deterministic
* Generally
    * Models generalize better when trained on data that is N(0,1) distributed (for non-deterministic case)
    * Models are both able to generalize

# Out of scope
## Single variable division
* Generally
    * Models cannot learn when generating from N(0,1)
    * Adding ReLU does not help us generalize
* CCP
    * Learning is slow and unstable when n_degree (model) >= n_degree (polynomial). Learns a bit for N(6,1) but cannot generalize
* Polynomial NN
    * Model learns nicely if n_degree (transfer) >= n_degree (polynomial) for N(6,1) but cannot generalize

## Two variable division
* CCP 
    * Model cannot learn at all
    * Model with ReLU cannot learn at all

## Single variable logarithm
* Polynomial NN
    * Model learns 
