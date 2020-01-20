# Normalizing Flows

This are some basic normalizing flows implemented in PyTorch. A very short description and overview of how normalizing flows work is provided next. For more thorough explanations see the following references:

Nice introductory blogpost by Adam Kosiorek

    http://akosiorek.github.io/ml/2018/04/03/norm_flows.html

A good overview paper that can also serve as an introduction

    Kobyzev, Ivan, Simon Prince, and Marcus A. Brubaker. 
    “Normalizing Flows: An Introduction and Review of Current Methods.” 
    ArXiv:1908.09257 [Cs, Stat], 
    December 8, 2019. http://arxiv.org/abs/1908.09257.

Very thorough and extensive overview paper

    Papamakarios, George, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. 
    “Normalizing Flows for Probabilistic Modeling and Inference.” 
    ArXiv:1912.02762 [Cs, Stat], 
    December 5, 2019. http://arxiv.org/abs/1912.02762.

## Basics

Normalizing flows are a powerful method that allow to transform a density distrbution into an arbitrary complex density distribution. An intuitive explanation is to look at normalizing flows as taking away probability mass from one part of a distribution and placing it back in a different position. This operaiton preserves the total mass of one, thus making the output distribution a probability distribution. The flow operations have to be smooth bijective operations. This means that we can take the derivative of the operation (smooth) and that the mapping can be inverted (bijective means a one-to-one mapping between input and output points). For more thorough explanations the reader is referred to the previous references.

There are four properties of a normalizing low that may have to be calculated in any given implementation. I will first atempt to describe these in intuitive, high level terms:

* The forward calculation, $\bm{T}$. Used to transform a simple random variable to a more complex one.
* The backward calculation (inverse of forward calculation), $\bm{T^{-1}}$. Used to transform the complex random variable to its simpler form.
* The determinant of the forward's Jacobian, $\bm{|det J_{T}| = |det \frac{\delta T}{\delta u}|}$. Used to calculate the probability distribution of the complex random variable by taking the probability distribution of the simpler random variable.
* The determinant of the backward's Jacobian, $\bm{|det J_{T^{-1}}| = |det \frac{\delta T^{-1}}{\delta x}|}$. Used to calculate the probability distribution of the simple random variable by taking the probability distribution of the more complex random variable.

_Formal definitions and relations between above mentioned functions_:

Let $\textbf{u} \in \mathbb{R}$ be a random variable and $\textbf{T}: \mathbb{R^{d}} \mapsto \mathbb{R^{d}}$ a bijective function a smooth bijective function. This $\textbf{T}$ can be used to map $\textbf{u} \sim \textbf{p(u)}$ to $\textbf{x = T(u)}$. Since the function is bijective it can also be inverted, we call this inverse $\mathbf{T^{-1}}$.

## Model usage

Flow based model support two types of operations, sampling from the model, and evaluating the model's density.

* _Sampling from the model_ - $\bm{x=T(u)}$: Sample from simple distribution $\bm{u}$ and evaluating the forward transform $\bm{T}$ to generate a sample $\bm{x}$. Relies on samples from $\bm{U}$.

* _Evaluating model density_ - $\bm{p_{x}(x) = p_{u}(T^{-1}(x))|det J_{T^{-1}}(x)|}$: Take a sample $\bm{x}$ and evaluate its density (probability) under the model. It requires computing the inverse transformation $\bm{T^{-1}}$ and its Jacobian determinent, and evaluating the density $\bm{p_{u}(u)}$. Relies on samples from $\bm{U}$.

Main thing to remember: forward transformation $\bm{T}$ is used for sampling from the model, inverse transformation $\bm{T^{-1}}$ is used for evaluating densities.

## Common loss functions

Depending on the available data, different approaches to learning are more suitable. Fitting a flow-based model is done by fitting the model distribution $\bm{p_{x}(x; \theta)}$ to target distribution $\bm{p^{*}_{x}(x)}$ using a divergence between them. Only the Kullback-Leibler divergence will be treated next since this is the most commonly used form. Naming is adapted from _"Normalizing Flows for Probabilistic Modeling and Inference"_.

* _Forward KL Divergence and Maximum Likelihood Estimation_

The following loss function is mostly suited for situations in which we have samples from the target distribution, but cannot evaluate its density. Because of this the loss function is expressed such that it contains only terms of $\bm{x}$.

$L(\theta) = D_{KL}[p_{x}^{*}(x) || P_{x}(x;\theta)] = -\mathbb{E}_{p^{*}_{x}(x)} [log p_{u}(T^{-1}(x;\phi); \psi) + log |det J_{T^{-1}} (x; \phi)|] + const$

During training only evaluation of the model density is necessary as described in the previous section. Thus training requires computing the inverse transformation $\bm{T^{-1}}$ and its Jacobian determinent, and evaluating the density $\bm{p_{u}(u)}$. The operations of evaluationg the forward transform $\bm{T}$ and sampling from $\bm{x}$ are not needed during trainig, but are necessary if the model will be used for sampling after training.

* _Backward KL Divergence_

This loss function is mostly suited for situations where we can evaluate the target density and do not have access to samples from the distribution, but we can sample from $\bm{u}$. So the loss function is expressed such that it contains only terms of $\bm{u}$.

$L(\theta) = \mathbb{E}_{p_{u}(u;\psi)} [ log p_{u}(u; \psi) - log | det J_{T}(u; \phi)| - log \widetilde{p}_{x}(T(u; \phi))] + const$

The requirements for using this loss function are opposite for the use of the Forward KL Divergence.

## Useful mathematical theories and concepts to look at

* _Measure Theory_.
* _Group Theory_.
* _Topology_.
* _Information Theory_.
