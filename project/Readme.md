# Evaluating adversarial robustness of uncertainty aware models
## About
This repository contains experiments conducted as part of the course project of the EE5111 course, Spring 2020. We first set out to understand the formulation of dropout as an approximation of a Gaussian Process in the paper "[Dropout as a Bayesian Approximation](http://proceedings.mlr.press/v48/gal16.pdf)" by Yarin Gal et al. We also study various choices of uncertainty measures and their significance in detecting adversarial examples, an open challenge in the current machine learning community. MC Dropout, the method proposed by Yarin Gal et al. fails to provide reliable uncertainty estimates. Since variational inference schemes typically used in Bayesian Neural Nets are all prone to underestimating the posterior, we switch gears and study "[Evidential Deep Learning to Quantify Model Uncertainty](https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty.pdf)"  a recent paper that uses principles from the Theory of Evidence for uncertainty aware probability estimates. Please refer to the Midterm and final presentations for more details.

## Code
The main training and evaluation code has been provided in the notebooks *uncertainty_adv.ipynb* and *MC_dropout_adv.ipynb*.  We evalute the following four models against FGSM attacks using the MNIST test set. Note that all the models have the same architecture.
  - 'L2'- conventional LeNet network with only L2 regularisation.
  - 'MC Dropout' - The model used in "Dropout as a Bayesian Approximation"
  - 'MC + Concrete' - MC dropout  with the [Concrete Dropout](https://arxiv.org/abs/1705.07832) layer used instead of normal dropout.
  - 'EDL' - The method provided in the paper based on Evidence theory.

## Dependencies
- opencv >= 3.4.2
- keras >= 2.3.1
- tensorflow >= 1.14 (only tf 1.0 supported)
- numpy >= 1.18.1
- matplotlib >= 3.1.3
- scipy >= 1.4.1
- cleverhans >= 3.0.1



