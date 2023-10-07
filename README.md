# VMD-MFRFNN

## Introduction

**Multi-step-ahead stock price prediction using recurrent fuzzy neural network and variational mode decomposition**

*Authors:* [Hamid Nasiri](https://www.linkedin.com/in/hamid-nasiri-b5555487/), [Mohammad Mehdi Ebadzadeh](https://www.linkedin.com/in/mehdi-ebadzadeh-28bb3b35/)

*Abstract:* Financial time series prediction has attracted considerable interest from scholars, and several approaches have been developed. Among them, decomposition-based methods have achieved promising results. Most decomposition-based methods approximate a single function, which is insufficient for obtaining accurate results. Moreover, most existing research has concentrated on one-step-ahead forecasting that prevents market investors from making the best decisions for the future. This study proposes two novel multi-step-ahead stock price prediction methods based on different decomposition techniques, including discrete cosine transform (DCT), i.e., a linear transform, and variational mode decomposition (VMD), i.e., a non-linear transform. DCT-MFRFNN, a method based on DCT and multi-functional recurrent fuzzy neural network (MFRFNN), uses DCT to reduce fluctuations in the time series and simplify its structure and MFRFNN to predict the stock price. VMD-MFRFNN, an approach based on VMD and MFRFNN, brings together their advantages. VMD-MFRFNN consists of two phases. The input signal is decomposed to several intrinsic mode functions (IMFs) using VMD in the decomposition phase. In the prediction phase, each IMF is given to a separate MFRFNN for prediction, and predicted signals are summed to reconstruct the output. DCT-MFRFNN and VMD-MFRFNN use the particle swarm optimization (PSO) algorithm to train MFRFNN. In this research, for the first time, the gradient descent method is used to train MFRFNN. Three financial time series are used to evaluate the proposed methods. Experimental results indicate that VMD-MFRFNN surpasses other state-of-the-art methods. VMD-MFRFNN, on average, shows a decrease of 31.8% in RMSE compared to the MEMD-LSTM method. Also, DCT-MFRFNN outperforms MFRFNN and DCT-LSTM in all experiments, which reveals the favorable effect of DCT on MFRFNN’s performance. To assess the effectiveness of PSO in training VMD-MFRFNN, we compared its performance with twelve different metaheuristic approaches. PSO, on average, shows a decrease of 9.4% in MAPE compared to other metaheuristic methods.

This repository contains MATLAB source code of the following paper:

[Multi-step-ahead stock price prediction using recurrent fuzzy neural network and variational mode decomposition](https://www.sciencedirect.com/science/article/abs/pii/S1568494623008852)

## Source Code and Dataset

To run the code simply execute `main.m`

**Datasets:** 
The [`Benchmarks`](Benchmarks/) folder contains three financial time series datasets used for testing and evaluating the code.

+ `HSI_Index.mat` corresponds to HSI time series.
+ `SSE_Index.mat` corresponds to the SSE time series.
+ `SandP_Index.mat` corresponds to the SPX time series.

## Citation

This repository accompanies the paper ["Multi-step-ahead stock price prediction using recurrent fuzzy neural network and variationa"](https://www.sciencedirect.com/science/article/abs/pii/S1568494623008852) by [Hamid Nasiri](https://www.linkedin.com/in/hamid-nasiri-b5555487/) and [Mohammad Mehdi Ebadzadeh](https://www.linkedin.com/in/mehdi-ebadzadeh-28bb3b35/), published in Applied Soft Computing journal.

If you use either the code, datasets or paper, please consider citing the paper.

```
@article{nasiri2023multi,
  title={Multi-step-ahead stock price prediction using recurrent fuzzy neural network and variational mode decomposition},
  author={Nasiri, Hamid and Ebadzadeh, Mohammad Mehdi},
  journal={Applied Soft Computing},
  pages={110867},
  year={2023},
  publisher={Elsevier}
}
```

## Contact Me

If you have any questions, do not hesitate to reach me via [Linkedin](https://www.linkedin.com/in/hamid-nasiri-b5555487/) or email: h.nasiri@aut.ac.ir

Thank you so much for your attention.
