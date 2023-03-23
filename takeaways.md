# Test dataloaders
0. Normal(-50, 1)
1. Normal(-50, 5)
2. Normal(-50, 25)
3. Normal(0, 1)
4. Normal(0, 5)
5. Normal(0, 25)
6. Normal(90, 1)
7. Normal(90, 5)
8. Normal(90, 25)

# 5a2 + 6b2
* PolynomialNN
    * Generally
        * Able to generalize OOO after 30 epochs
        * Loss for normal(0,1) is way lower than normal(0,5)
    * Deterministic 
        * Model trained on normal(0,5) performs better in test on R2
    * Non-deterministic
        * Model trained on normal(0,1) performs better in test on R2

* CCP
    * Generally
        Able to generalize OOO after 30 epochs
    * Deterministic
        * Able to generalize OOO after 30 epochs
        * Model trained on normal(0,1) performs better in test on R2
    * Non-deterministic
        * Normal(0,1) model performs better in test on R2 
        * Normal(0,5) model drops in performance for dataloaders: Normal(0,1) and Normal(90,1)

**Overall**
Models are able to generalize for $5a^2+6b^2$ pretty well. Some drops in performance is seen in CCP, whereas PolynomialNN maintains great performance throughout.

# 2a3 * 6b2
* PolynomialNN
    * Deterministic
        * Learning curves for validation look odd for R2. Models peak before plummeting
        * Extremely high MAEs on test set
        * Test R2 is alright for Normal(-50, 25) and Normal(90,25)
    * Non-deterministic
        * Learning curves for validation look odd for R2. Normal(0,1) model peaks and plummets
        * Extremely high MAEs on test set
        * Test R2 is alright for Normal(-50, 5), Normal(-50, 25), and Normal(90, 25)

* CCP
    * Deterministic
        * Learning curves look okay.
        * Extremely high MAEs on test set (might just be because of large target values)
        * Normal(0,5) model performs generally better in R2, where Normal(0,1) model has bad test performance for dataloaders: Normal(-50, 1) and Normal(90,1).
    * Non-deterministic
        * Generally same as deterministic
        * Normal(0,5) model generally has good performance but fails on Normal(0,1)
        * Normal(0,1) model has terrible performance for dataloaders: Normal(-50, 1), Normal(90, 1) and Normal(90, 5)

**Overall**
PolynomialNN is no longer able to generalize OOO and performs worse when more layers are added. It does not work to have a polynomial output functions and no activations.
CCP is mostly good at it across dataloaders, but we see the model struggle more for interactions. Adding ReLUs to CCP leads to worse performance allround.

# 2a3*b2 - 3c
* Generally
    * High MAE, Low MAPE and high R2 generally
* Normal(0,5)
    * Models trained with noise are better at generalizing. Deterministic PolynomialNN is bad at generalizing
* Normal(0,1)
    * Low R2 (sometimes negative), high MAPE over 100%, and high MSE
    * Models are way worse than Normal(0,5)

**Overall**
Models trained on non-deterministic Normal(0,5) data are still generalizing well. CCP on deterministic Normal(0,5) performs great as well.

# 2a3*b3 - 3c
* Normal(0,5)
    * Polynomial NN outperforms  CCP in Normal(90,5)
    * CCP is mostly generalizing well in deterministic case
        * Fails for Normal(0,1), Normal(0,25) (high R2 and high MAPE?)
* Normal(0,1)
    * Training performance is low
    * Models are bad at generalizing.

**Overall**
CCP models trained on Normal(0,5) are able to generalize OOD.

# a\*b\*c+d-e-f-g
* Generally
    * Able to fit OOD.
* Normal(0,5)
    * Great performance in general. Models trained on Normal(0,5) fails for Normal(0,1) in test however
    * CCP models are generally better than PolynomialNN
* Normal(0,1)
    * Non-deterministic training leads to worse performance
    * CCP models are generally better than PolynomialNN
    * Models fail for Normal(90,1)

**Overall**
Models trained on Normal(0,1) perform better than Normal(0,5). CCP models are better than PolynomialNN.

# abcdefg
**Overall**
Models cannot fit the polynomial at all. Training curves are decreasing but validation curves are increasing. R2 is negative and MAE is extremely high.

# a10-b9
Overflow encountered. Degree is too high to ensure numerical stability.