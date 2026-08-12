# Training Timestep Sampling

$$
\operatorname{shift}(x)=\frac{3x}{1+2x}
$$

## 1. DiffusionNFT origin

$$
\begin{aligned}
x_0,x_1,\ldots,x_9
&=\operatorname{LinearSpace}(0.001,1,10),\\
T_j&=\operatorname{shift}(x_j),\qquad j=0,1,\ldots,9,\\
t_0,t_1,\ldots,t_8
&=\text{从 }\{T_0,T_1,\ldots,T_9\}\text{ 中无放回随机取 9 个。}
\end{aligned}
$$

## 2. Logit-normal

$$
\begin{aligned}
z_i&\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),\\
x_i&=\operatorname{sigmoid}(z_i),\\
t_i&=\operatorname{shift}(x_i),\qquad i=0,1,\ldots,8.
\end{aligned}
$$

## 3. Uniform over $N$ candidates

$$
\begin{aligned}
x_0,x_1,\ldots,x_{N-1}
&=\operatorname{LinearSpace}(0.001,1,N),\\
T_j&=\operatorname{shift}(x_j),\qquad j=0,1,\ldots,N-1,\\
t_0,t_1,\ldots,t_8
&=\text{从 }\{T_0,T_1,\ldots,T_{N-1}\}\text{ 中均匀、有放回地随机取 9 个。}
\end{aligned}
$$
