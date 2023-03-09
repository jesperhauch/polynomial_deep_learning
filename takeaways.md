* Feedforward Neural Network with ReLU
    * Model is great at learning in distribution but cannot generalize OOD.
* Polynomial NN 
    * Model with one layer can learn OOD if n_degree (activation) >= n_degree (polynomial)
    * Model with x^3 activations and 2 layers cannot learn OOD if n_degree > 3
* CCP 
    * Model can learn OOD if n_layers >= n_degree