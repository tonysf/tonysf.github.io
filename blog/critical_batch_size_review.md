# Intro

Some recent blog posts [1](https://leloykun.github.io/ponder/steepest-descent-crit-bz/#22-convergence-bound-without-weight-decay)[2](https://leloykun.github.io/ponder/steepest-descent-convergence/) by Franz Louis Cesista[^1] attempt to analyze the convergence of what they call "steepest descent" with Nesterov momentum and weight decay. The analysis has several serious errors that invalidate the conclusions drawn. First, the "convergence bounds" do not actually show convergence. The "rates" contain terms that do not go to zero and do not depend on noise, making the bounds completely meaningless. Second, when weight decay is added, the algorithm becomes a conditional gradient method solving a *constrained* problem, for which the correct stationarity measure is the Frank-Wolfe gap, not the gradient norm. The analysis is aimed at the incorrect quantity and produces meaningless residual terms that are then misidentified as a "noise floor." Third, the critical batch size theorem is defined in terms of this non-vanishing "floor" rendering it useless.

I will use all the notation that Cesista uses to make it easy to relate the posts. I will always assume $\mathcal{W} = \mathbb{R}^p$ for some $p>0$ since all the results only hold in that case anyways.

# Nonsensical "convergence" bounds

There are several errors in these blog posts but the most egregious is that the "convergence" bounds are meaningless - they don't go to $0$. This is the first "convergence" bound given.

!(Screenshot 2025-12-14 at 03.32.04.png)

What are these terms, $Y, \widetilde{Y},$ and $Z$?

!(Screenshot 2025-12-14 at 03.32.46.png)

(A reminder: $\beta$ is a momentum parameter and $\eta$ is the step size or learning rate.)

Disregarding the general looseness of the analysis, this does not go to $0$, so there is no convergence! $Y$ and $\widetilde{Y}$ are not being divided by $T$ but are at least proportional to $\sigma^2$ and $\sigma$ respectively, so can reasonably be considered some noise floor. $Z$ is not proportional to $\sigma$ though. In $Z$, the terms with $\eta$ can be controlled but $\frac{D}{2}$, which is not multiplied by $\eta$, is not going to $0$ and also does not depend on the noise at all ($D$ is a fixed constant that depends on the norm we've chosen). This makes this bound completely meaningless from an optimization pov; it is not a "noise floor".  This is also not a fundamental property of the algorithm - this could easily be fixed by doing a better analysis, in which case this term won't appear.

## Weight decay

The next results of their blog post try to analyze what happens with weight decay.

### A detour about normal cones, gaps, and constrained problems

Before we go on to the "convergence" theorems, I want to point out that when you add weight decay to an lmo update you get the conditional gradient algorithm - this is a mathematical fact. More concretely, if we follow the notation of Cesista, adding weight decay with parameter $\lambda$ to the unconstrained update we get

$$
\begin{equation*}
W_{t+1} = (1-\lambda)W_t + \eta \mathrm{lmo}(C_t) = (1-\lambda) W_t + \lambda \mathrm{lmo}_{\frac{\eta}{\lambda}}(C_t)
\end{equation*}
$$

which is the conditional gradient of Frank-Wolfe algorithm with stochastic estimator $C_t$ for the gradient. The step size in the eyes of the Frank-Wolfe algorithm is $\lambda$, and the radius of the constraint is $\frac{\eta}{\lambda}$. Using pytorch scaling where $\eta = \eta_t$ and $\lambda = \lambda_t = \rho \eta_t$ for some $\rho$ that we specify, we get that $\frac{\eta}{\lambda} = \frac{1}{\rho}$ and so the radius of the constraint set is implicitly selected by the chosen weight decay. What Cesista does is not exactly this, he instead makes an assumption that $\lambda \eta \leq 1$ and this reparameterization still works for the ball of radius $\frac{1}{\lambda}$, then.

What's the significance of this to the blog post?

We know that adding weight decay like this gives an algorithm that solves a *constrained problem*. But, the "convergence" theorem given is in terms of the gradient norm, which might not go to $0$ even if the algorithm finds a stationary point. There is a difference between optimality for the two problems

$$
\begin{equation}
\begin{aligned}
\min\limits_{W \in \mathcal{W}} f(W)\quad\quad&\text{vs}\quad\quad\min\limits_{W\colon \|W\|\leq \frac{1}{\rho}}f(W)\\
\underbrace{0 = \nabla f(W^\star)}_{\text{\#1}}\quad\quad&\quad\quad\quad \underbrace{0 \in \nabla f(W^\star) + N_{\frac{1}{\rho}\mathcal{B}}(W^\star)}_{\text{\#2}}
\end{aligned}
\end{equation}
$$

The first optimality condition #1 can be measured by $\|\nabla f(W)\|^\dagger$ but the second one #2 could be solved by some $W^\star$ even when $\|\nabla f(W^\star)\|^\dagger=0$. If we had access to the gradient of $f$ and $\mathrm{lmo}_{\mathcal{B}}$ at a given point $W$, we could check how far $W$ is from solving #2 by computing the so-called Frank-Wolfe gap,

$$
\begin{equation}
\langle \nabla f(W), W-\frac{1}{\rho}\mathrm{lmo}_{\mathcal{B}}(W)\rangle.
\end{equation}
$$

This quantity is the analog of $\|\nabla f(W)\|^\dagger$ for the constrained case. For any $W \in \frac{1}{\rho}\mathcal{B}$, we know it must be nonnegative since

$$
\begin{equation}
\langle \nabla f(W),W\rangle \geq \langle \nabla f(W), \mathrm{lmo}_{\frac{1}{\rho}\mathcal{B}}(W)\rangle
\end{equation}
$$

by definition of $\mathrm{lmo}_{\frac{1}{\rho}\mathcal{B}}$. 
The optimality condition for the constrained case involves something called the Normal cone $N_{\frac{1}{\rho}\mathcal{B}}$ which is a *set-valued* map with the following definition

$$
\begin{equation}
N_{\frac{1}{\rho}\mathcal{B}}(W) = \left\{Z\in \mathcal{W}\colon \langle Z, Y-W\rangle \leq 0, \ \forall Y\in\frac{1}{\rho}\mathcal{B}\right\}
\end{equation}
$$

If we want $0 \in \nabla f(W^\star) + N_{\frac{1}{\rho}\mathcal{B}}(W^\star)$, then it's equivalent to have $-\nabla f(W^\star) \in N_{\frac{1}{\rho}\mathcal{B}}(W^\star)$. If we plug $-\nabla f(W^\star)$ into the above definition with $W=W^\star$, then the optimality condition amounts to finding $W^\star$ that satisfies

$$
\begin{equation}
\begin{aligned}
\forall Y \in \frac{1}{\rho}\mathcal{B}\colon\quad & \langle -\nabla f(W^\star), Y-W^\star \rangle\leq 0\\
\iff&\langle \nabla f(W^\star), W^\star - Y\rangle \leq 0
\end{aligned}
\end{equation}
$$

Since this is required to hold for all $Y\in\frac{1}{\rho}\mathcal{B}$, we can just check if it holds for the max. If we take the max in the above with respect to $Y\in\frac{1}{\rho}\mathcal{B}$, we would be computing none other than $\mathrm{lmo}_{\frac{1}{\rho}\mathcal{B}}(\nabla f(W^\star))$. Computing $\mathrm{lmo}_{\frac{1}{\rho}\mathcal{B}}(\nabla f(W_t))$ and substituting that into $\langle \nabla f(W_t), W_t - \mathrm{lmo}_{\frac{1}{\rho}\mathcal{B}}(\nabla f(W_t))\rangle$ thus gives us a check on how close the normal cone inequality is to being satisfied for $-\nabla f(W_k)$ (so, a quantification of the optimality condition #2), which is exactly the analog of looking at the value of $\|\nabla f(W_t)\|^\dagger$ in the unconstrained setting (how far are we from satisfying #1).

### Back on track

The Frank-Wolfe gap *should* appear naturally in the proofs from the descent lemma as well (it doesn't in Cesista's blog, though). This is because at some point in the descent lemma you have $W_{t+1}-W_t$ and this becomes either $\eta \mathrm{lmo}(C_t)$ in the unconstrained case or $\eta(\mathrm{lmo}(C_t)-\lambda W_t)$ in the constrained case. In both cases, you get the certificate of optimality (either the gradient norm, or the Frank-Wolfe gap). Because Cesista has tried to brute force the gradient norm directly instead of using the gap, they have quantities that do not go to $0$ in their "convergence bound".

!(Screenshot 2025-12-14 at 04.26.14.png)

Let's see what $X, Y, \widetilde{Y}$, and $Z$ are.

!(Screenshot 2025-12-14 at 04.26.43.png)

In $Z$ we see the terms $\frac{2L^2}{\lambda D}$, $\frac{D}{\lambda}$, and $2D$. These are not noise terms, these are artifacts from not properly analyzing the algorithm with decay using the gap. Again, these could be removed if the analysis was done correctly.

### Corrected analysis

I am posting here the correct analysis; it's a very minor change from the analysis we already did when writing the Scion paper[^2] a year ago. The Nesterov momentum is the only difference (and it is minor) besides the change in notation, which was done to make it more accessible.

$$
\begin{equation}
\begin{aligned}
f(W_{t+1}) &\leq f(W_t) + \langle \nabla f(W_t), W_{t+1}-W_t\rangle + \frac{L}{2} \|W_{t+1}-W_t||^2\\
&\leq f(W_t) + \langle \nabla f(W_t), \eta(A_t^* - \lambda W_t)\rangle + \frac{L}{2} \|\eta(A_t^*-\lambda W_t)\|^2\\
&\leq f(W_t) + \eta\langle \nabla f(W_t) - C_t + C_t, A_t^* - \lambda W_t\rangle + \frac{L\eta^2}{2}\\
&\leq f(W_t) + \eta\langle C_t, A_t^*-\lambda W_t\rangle + \eta\langle\nabla f(W_t)-C_t,A_t^*-\lambda W_t\rangle + \frac{L\eta^2}{2}\\
&\leq f(W_t) + \eta\langle C_t, \mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-\lambda W_t\rangle + \eta\langle \nabla f(W_t)-C_t,A_t^*-\lambda W_t\rangle + \frac{L\eta^2}{2}\\
&\leq f(W_t) + \eta\langle \nabla f(W_t),\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-\lambda W_t\rangle + \eta\langle C_t - \nabla f(W_t),\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-\lambda W_t\rangle + \eta\langle \nabla f(W_t)-C_t,A_t^*-\lambda W_t\rangle + \frac{L\eta^2}{2}\\
&\leq f(W_t) + \eta\langle \nabla f(W_t),\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-\lambda W_t\rangle + \eta\langle C_t-\nabla f(W_t),\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-A_t^*\rangle + \frac{L\eta^2}{2}\\
\end{aligned}
\end{equation}
$$

Now we can apply Cauchy Schwarz to the last inner product. This gives

$$
\begin{equation}
\begin{aligned}
f(W_{t+1}) & \leq f(W_t) + \eta \langle \nabla f(W_t), \mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-\lambda W_t\rangle + \eta \|C_t-\nabla f(W_t)\|^\dagger\|\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-A_t^*\| + \frac{L\eta^2}{2}.
\end{aligned}
\end{equation}
$$

We can go one step further and bound the norm $\|\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))-A_t^*\|\leq 2$ using the triangle inequality since each of $\mathrm{lmo}_{\mathcal{B}}(\nabla f(W_t))$ and $A_t^*$ have norm equal to $1$. We can then rearrange and divide both sides by $\eta\lambda$ to get

$$
\begin{equation}
\langle \nabla f(W_t), \mathrm{lmo}_{\frac{1}{\lambda}\mathcal{B}}(\nabla f(W_t))-W_t\rangle \leq \frac{f(W_t)-f(W_{t+1})}{\eta\lambda} + \frac{2}{\lambda}\|C_t-\nabla f(W_t)\|^\dagger + \frac{L\eta}{2\lambda}
\end{equation}
$$

The left hand side is *exactly* the Frank-Wolfe gap for the problem

$$
\min\limits_{W\colon \|W\|\leq \frac{1}{\lambda}}f(W)
$$

Now if we just find a bound on the noise of our gradient estimator (that is, a bound on $\|C_t-\nabla f(W_t)\|^\dagger$) and set our parameters $\eta$ and $\beta$ correctly, then we will have a bona fide convergence rate. This is exactly what was already done in the Scion paper[^2]; the twist of putting the Nesterov momentum does not qualitatively change the analysis at all. Compare that with what you get after (36) in the blog post:

!(Screenshot 2025-12-14 at 05.07.42.png)

Most of these terms are just artifacts of the analysis Cesista has done, and not actually needed to bound the norm of the gradient.

# Critical Batch Size

No surprise, the previous bounds being incorrect led to incorrect conclusions about the critical batch size.

!(Screenshot 2025-12-14 at 05.09.39.png)

Look at how $\epsilon'$ is defined. You cannot ask for an $\epsilon$-critical point if $\epsilon$ is smaller than $Z'$ (which is even bigger than $Z$). Remember the constant $Z$ in the previous convergence rate, which was not going to $0$ and likely huge for problems of interest? Defining $\epsilon'$ this way makes this theorem completely meaningless. This is a cargo cult of optimization theory. They've seen optimization theorists have bounds, and they know what bounds ought to look like or some words around them (e.g., noise floor), but they clearly do not understand the meaning behind the bounds or why people care about them.

Eventually, the blog turns to learning rate scheduling based on batch size. An assumption is made:

!(Screenshot 2025-12-14 at 05.13.06.png)

This assumption is NOT satisfied by the lmo that we are usually interested in (sign, matrix sign). Any time the unit ball is not a smooth set, if it has any "corners", then the lmo will not be continuous, let alone Lipschitz-continuous.

# The followup blog post

The followup blog post claiming a convergence rate of $O(1/\epsilon^4)$ is completely wrong. There are terms in the "noise floor" that again don't depend at all on the noise. The claim that the rate $O(1/\epsilon^4)$ cannot be improved upon is misunderstood - this rate cannot be improved upon for *convergence to a critical point*. What's in the blog post is not even convergence to a noise-dominated region (which would be something like $O(\frac{1}{T^\alpha} + \sigma)$ for some $\alpha>0$); there is no bound on the rate then. For instance, in the Scion paper[^2] we showed convergence to a noise-dominated region with an exponent in the rate that was $1/2$, much better than $1/4$, and there is no contradiction because it was convergence to a noise-dominated region and not a critical point.

The statement that we find in the discussion (shown below) is nonsense, and misunderstands the claim in the cited paper Arjevani et al., 2022; probably because it's just repeating what was claimed in Kovalev, 2025 without understanding what they were referring to. This is more cargo cult behavior.

!(Screenshot 2025-12-14 at 05.17.56.png)

# A final note

The algorithm given in the blog posts of Cesista is *not* steepest descent.

Steepest descent specifically corresponds to the following update

$$
\begin{equation}
W_{t+1} = \mathrm{argmin}\{W\colon \langle C_t, W\rangle + \frac{1}{2\gamma}\|W_t-W\|^2\} = W_t + \gamma \|C_t\|^\dagger\mathrm{lmo}(C_t)
\end{equation}
$$

which has scaling by the dual norm $\|C_t\|^\dagger$. What is used in practice, and what we give in the Scion paper[^2], is rather a conditional gradient or normalized steepest descent update

$$
\begin{equation}
W_{t+1} = \mathrm{argmin}\{W\colon \langle C_t,W\rangle + \iota_{\gamma \mathcal{B}}(W-W_t)\} = W_t + \gamma \mathrm{lmo}(C_t)
\end{equation}
$$

where

$$
\begin{equation}
\iota_{\gamma\mathcal{B}}(W-W_t) = \begin{cases} 0 & W-W_t\in \gamma\mathcal{B}\\ +\infty & W-W_t\not\in\gamma\mathcal{B}\end{cases}
\end{equation}
$$

is the so-called indicator function that gives $+\infty$ if the argument is not in the set (this gives a way to encode constraints of optimization problems). The set $\mathcal{B}$ is the unit-ball for the norm $\|\cdot\|$, and we interpret scaling the ball to mean $\gamma\mathcal{B} = \{W\colon \|W\|\leq \gamma\}$.

!(Screenshot 2025-12-14 at 03.10.08.png)

What's written in (5) is actually the stochastic conditional gradient algorithm with "Nesterov momentum" used as a stochastic estimator of the gradient.

[^1]: Apparently Kaiyue Wen is also an author now, but was not when I was sent these blog posts.
[^2] [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529)