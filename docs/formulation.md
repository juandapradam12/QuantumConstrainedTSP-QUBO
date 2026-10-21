# QUBO for the constrained tour

This note expands the energy used in the computation. The tour it selects is described in the main text.

## Variables

Cities are $\{0,1,2,3,4\}$ and steps are $\{0,1,2,3,4\}$. The binary $x_{i,t}$ is 1 when city $i$ stands at step $t$. The tour returns from step 4 to step 0; that hop is part of the objective, not an extra step.

Two occupations are fixed by the problem rather than by a penalty:

$$
x_{0,0} = 1, \qquad x_{2,2} = 1.
$$

City 0 and city 2 are each visited once, so

$$
x_{0,t} = 0 \text{ for } t \neq 0, \qquad x_{2,t} = 0 \text{ for } t \neq 2,
$$

and steps 0 and 2 hold no other city,

$$
x_{i,0} = 0 \text{ for } i \neq 0, \qquad x_{i,2} = 0 \text{ for } i \neq 2.
$$

The free variables are $x_{i,t}$ for $i \in \{1,3,4\}$ and $t \in \{1,3,4\}$, nine bits. In the matrix they are ordered by step and, within a step, by city: $x_{1,1}, x_{3,1}, x_{4,1}, x_{1,3}, x_{3,3}, x_{4,3}, x_{1,4}, x_{3,4}, x_{4,4}$.

## Energy

Let $E$ be the set of undirected roads and $d_{ij} = d_{ji}$ their lengths. With $t+1$ read modulo 5,

$$
H_{\mathrm{length}} = \sum_{t=0}^{4} \sum_{\{i,j\} \in E} d_{ij}\, x_{i,t} x_{j,t+1},
$$

$$
H_{\mathrm{missing}} = \sum_{t=0}^{4} \sum_{i,j} \bigl(1 - A_{ij}\bigr)\, x_{i,t} x_{j,t+1},
$$

where $A_{ij} = 1$ if $\{i,j\} \in E$ and $A_{ij} = 0$ otherwise, including $i = j$. The permutation defects on the free variables are

$$
H_{\mathrm{step}} = \sum_{t \in \{1,3,4\}} \left( \sum_{i \in \{1,3,4\}} x_{i,t} - 1 \right)^2,
\qquad
H_{\mathrm{city}} = \sum_{i \in \{1,3,4\}} \left( \sum_{t \in \{1,3,4\}} x_{i,t} - 1 \right)^2.
$$

The steps and cities that were fixed contribute $(1-1)^2 = 0$ and are omitted. One penalty multiplies all of the soft terms:

$$
H = H_{\mathrm{length}} + \lambda\, H_{\mathrm{missing}} + \lambda \left(H_{\mathrm{step}} + H_{\mathrm{city}}\right).
$$

## One-hot expansion

For a binary triple $y_1,y_2,y_3$,

$$
(y_1 + y_2 + y_3 - 1)^2 = 1 - (y_1 + y_2 + y_3) + 2(y_1 y_2 + y_1 y_3 + y_2 y_3),
$$

using $y_k^2 = y_k$. Each free variable belongs to one step triple and one city triple, so its linear coefficient in $H_{\mathrm{step}} + H_{\mathrm{city}}$ is $-2$. A pair shares at most one triple, and then the product coefficient is $2$. Six triples give the constant $6$. Scaled by $\lambda$,

$$
\lambda(H_{\mathrm{step}} + H_{\mathrm{city}}) = 6\lambda - 2\lambda \sum_a x_a + 2\lambda \sum_{\{a,b\} \in P} x_a x_b,
$$

where $P$ is the set of pairs that share a step or a city. When a triple has exactly one 1, its contribution is zero and the constant is cancelled by the linear term.

## Length after the substitution

Four of the five hops touch a fixed city, so they are linear. The hop from step 3 to step 4 is quadratic. Writing $\ell(i,t)$ for the coefficient of $x_{i,t}$:

| Variable | Roads it fixes | Coefficient |
| --- | --- | --- |
| $x_{1,1}$ | $0 \rightarrow 1$, $1 \rightarrow 2$ | $3 + 3 = 6$ |
| $x_{3,1}$ | $0 \rightarrow 3$, $3 \rightarrow 2$ | $4 + 1 = 5$ |
| $x_{4,1}$ | $0 \rightarrow 4$, $4 \rightarrow 2$ | $2 + 6 = 8$ |
| $x_{1,3}$ | $2 \rightarrow 1$ | $3$ |
| $x_{3,3}$ | $2 \rightarrow 3$ | $1$ |
| $x_{4,3}$ | $2 \rightarrow 4$ | $6$ |
| $x_{1,4}$ | $1 \rightarrow 0$ | $3$ |
| $x_{3,4}$ | $3 \rightarrow 0$ | $4$ |
| $x_{4,4}$ | $4 \rightarrow 0$ | $2$ |

The only road between a city at step 3 and a city at step 4 is $1{-}3$, of length 4. It appears as two monomials, one for each direction:

$$
4\, x_{1,3} x_{3,4} + 4\, x_{3,3} x_{1,4}.
$$

Every other product $x_{i,3} x_{j,4}$ is a missing road, or a repeated city, and carries coefficient $\lambda$ in $\lambda H_{\mathrm{missing}}$.

These are the entries of the left matrix in the main text. Diagonal entries are the linear coefficients above. Off-diagonal entries are the coefficient of the product, not half of it.

## Quadratic form

The code stores a symmetric matrix $Q$ such that

$$
H(x) = x^{\mathsf T} Q x + C, \qquad C = 6\lambda.
$$

For that identity, the diagonal of $Q$ equals the linear coefficient, and an off-diagonal entry is half the product coefficient, because $x^{\mathsf T} Q x$ counts each pair twice. The matrices drawn in the main text undo that halving: entry $(a,b)$ there is the coefficient of $x_a x_b$.

## Why $\lambda > 7$ is enough, and why the gap locks at 3

The order $0 \rightarrow 1 \rightarrow 2 \rightarrow 3 \rightarrow 4$ is a permutation, so its one-hot terms vanish. Four hops are roads and sum to 9; the hop $3 \rightarrow 4$ is missing. Its energy is $9 + \lambda$.

The feasible tour $0 \rightarrow 4 \rightarrow 2 \rightarrow 3 \rightarrow 1$ has energy 16 for every $\lambda$. These two energies meet at $\lambda = 7$. For every $\lambda > 7$ the feasible tour lies strictly below this missing-road order. Exhaustive search of the 512 states shows that no other configuration undercuts it either once $\lambda \ge 8$: from that value the ground state is unique and equals the tour.

The other feasible tour has energy 19. The missing-road energy passes 19 at $\lambda = 10$. From $\lambda = 10$ the gap from the ground state to the next level is $19 - 16 = 3$, and it stays there, because every state that is not one of the two feasible tours has an energy that keeps rising with $\lambda$.

## Ising form

The same polynomial is an Ising energy. Substitute $x_a = (1 + s_a)/2$ with $s_a \in \{-1,+1\}$. Each linear term becomes a field, each product becomes a coupling, and the constant shifts. The ground state is unchanged. A quantum annealer minimises that Ising energy; the computation here minimises $H$ directly and checks it against every one of the 512 configurations.
