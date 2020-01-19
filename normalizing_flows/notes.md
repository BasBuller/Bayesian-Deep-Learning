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

* The forward calculation, $\bm{f}$. Used to transform a simple random variable to a more complex one.
* The backward calculation (inverse of forward calculation), $\bm{g}$. Used to transform the complex random variable to its simpler form.
* The determinant of the forward's Jacobian, $\bm{|det \frac{\delta f}{\delta z}|}$. Used to calculate the probability distribution of the complex random variable by taking the probability distribution of the simpler random variable.
* The determinant of the backward's Jacobian, $\bm{|det \frac{\delta g}{\delta y}|}$. Used to calculate the probability distribution of the simple random variable by taking the probability distribution of the more complex random variable.

_Formal definitions and relations between above mentioned functions_:

Let $\textbf{z} \in \mathbb{R}$ be a random variable and $\textbf{f}: \mathbb{R^{d}} \mapsto \mathbb{R^{d}}$ a bijective function a smooth bijective function. This $\textbf{f}$ can be used to map $\textbf{z} \sim \textbf{q(z)}$ to $\textbf{y = f(z)}$. Since the function is bijective it can also be inverted, we call this inverse $\mathbf{g (= f^{-1})}$.

## Useful mathematical theories and concepts to look at

* _Measure Theory_.
* _Group Theory_.
* _Topology_.
* _Information Theory_.
