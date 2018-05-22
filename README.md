# RevenueOptimization-WINE2017

The project contains the code I used for experiments in the paper: "Dynamic Pricing in Competitive Market" I presented at WINE 2017 conference held at Bangalore.

This code compares the performance of various price update rules in a CES gross subsititutes market. Specifically, the following methods are tested empirically:
1. Online gradient descent with decreasing step size
2. Optimistic mirror descent
3. Tatonnement update step proposed by Cole et al with fixed step size
4. Tatonnement update step proposed by Cole et al with decreasing step size

The relative performance of these approaches is measured in terms of the excess demand (demand - supply) and revenue. With the obtained plots, we can also empirically compare how fast the dynamic converges to equilibrium.

Additional details can be found here:
http://lcm.csa.iisc.ernet.in/wine2017/
https://link.springer.com/book/10.1007%2F978-3-319-71924-5
https://www.youtube.com/watch?v=dlRf4Q-205M&t=0s&list=PLoPqhV1N_AmrE19QiS8OB0IHHTJyqRLKZ&index=2