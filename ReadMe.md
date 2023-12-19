# Code will be release soon (22/12/2023)

Author : 
- **CALLARD Baptiste** (MVA)
- **TOCQUEC Louis** (MVA)

As part of the "Introduction to Probabilistic Graphical Models and Deep Generative Models" course run by P. LATOUCHE, P.A. MATTEI in the MVA Master's programme. We studied the paper "Are Generative Classifiers More Robust to Adversarial Attacks?" (https://arxiv.org/pdf/1802.06552.pdf).

# Paper review 
The article "Are Generative Classifiers More Robust to Adversarial Attacks?" investigates the robustness of deep neural network classifiers against adversarial attacks. The focus is on the comparison between generative classifiers, which model the conditional distribution of labels given inputs, and discriminative classifiers. The authors propose the deep Bayes classifier, which is an improvement over the classical naive Bayes, using conditional deep generative models. They develop methods to detect adversarial examples. Their experimental results suggest that deep Bayes classifiers are more robust than traditional deep discriminative classifiers and that the proposed detection methods effectively counter many recent adversarial attacks.

# Personal implementation
We re-implemented the 7 different models from scratch for the MNIST, FashionMNIST and SVHN datasets. We then implemented a white box attack $l_{\infty}$ and a black box attack zoo. We then tested the generative models versus the discriminative models.

Finally, we presented our results in a document in NEURIPS format.

# Some Visualisation

You can go through our results in the notebook. Also, all the experiments are reproducible. For the training sessions, we used Google Colab to take advantage of the GPUs.

Here are the main results and cool visualisations : 

## Some attacks : 
Black box attack: target zoo in different classes. The image on the left is well classified before the attack. After the attack, it is classified in the target class. We present successful attacks on discriminatory models.

On SVHN
![attacks_eg_2](https://github.com/b-ptiste/generative-model-adv-attack/assets/75781257/abdfeffb-3777-4791-ad33-9e3d04ab6ec5)

On FashionMNIST
![attacks_eg](https://github.com/b-ptiste/generative-model-adv-attack/assets/75781257/38c466b3-ff18-4ac4-9ccb-0fda21df054b)

## Robustness :

- Generative models : GFZ, GBZ, GFY, GBY
- Discriminative models : DBX, DFX, DFZ

We confirm the results of the paper with our implementation. On the other hand, the approach works on small datasets because, by construction, generative models are not good at prediction. So, if the question is about robustness, we may be interested in the drop in performance. However, in practice what we are interested in is performance, for example in terms of accuracy against such and such attacks. In this configuration, generative models will lag behind real-life datasets due to their lack of predictive power.

### White box attack

![summarise](https://github.com/b-ptiste/generative-model-adv-attack/assets/75781257/8654ca8d-8cf9-49a1-8c37-f1a6152b9aa8)

### Black box attack

Fashion Mnist

We can see that discriminative model are completely misled by the attack targeted at class 7.

<p float="left">
  <img src="https://github.com/b-ptiste/generative-model-adv-attack/assets/75781257/c2fc4218-2241-48e8-a4ab-743efce974ff" width="45%" /> 
  <img src="https://github.com/b-ptiste/generative-model-adv-attack/assets/75781257/1d597fb9-ec3c-4705-b702-95500a613af2" width="45%" />
</p>


