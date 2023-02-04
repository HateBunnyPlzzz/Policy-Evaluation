Today we tackle  Policy Evaluation and we are going to do this in the GridWorld that we created in the previous post about creating your own OpenAI Gym Compliant environment for RL.

Since we're not going to be doing a game type of solution remember that we already know the state-transition probabilities so we're really just solving the system of equations.

We do need to know it's a terminal state, we don't need **magicSquares** to be added to the grid, we do not need to know agent **row and column**, no need to **setSate**, We do need to know if we're attempting to **move off** the grid. No need for step function and no no need for **reset** and **render** functions. We will use functions to print out the agents estimate for the value function as well it's policy but those will be separate utility functions that live outside the class. 

To initialise the state-transition probabilities, there are two ways to doing it:

1. You can take into account all of the possible combinations between states, so the probability of transitioning from state 1 to state 62 is 0 which is obvious.
2. Or you can simply say we're only going to take into account the combinations that can actually happen withing the environment. *(Easier to do)*

- Iterate over states and our possibles actions

As we need some Convergence Criteria, so the agent's estimate of the value function approaches the true value function asymptotically that means you take their difference, you're gonna get smaller and smaller number over time. We introduced a parameter ***THETA***.

Initial estimate of the value function, this is arbitrary as we gonna use optimistic initial values as the value function for every state is large negative number at any rate and so if we bias it as Zero, via agent we would get exploration in any case even with a purely greedy strategy. (Epsilon-approaches)

Policy can be anything you want, we're gonna improve upon it anyway.

One reasonable place to start is the equiprobable random strategy.

that means for each state in the environment, the agent can move in any of the possible directions with equal probability.

Policy evaluation
We want to check for convergence as we iterate through the loop and check for convergence we need both the old value and the new value of the estimate of the value function so that new value is given by the policy.

Value iteration has better convergence than the Policy iteration 

Policy iteration is composed of policy evaluation and policy improvement. What we want to do is initialise some stability flag set it to True, we want to iterate over each state in other words conduct a sweep of the state space, save our old action, calculate the maximum action for out current state given the state-transition probabilities and the rewards and the current estimate of the value function for the new resulting states. If the old action is different from the new action then the policy is not stable then keep going, if the policy is stable then stop, return the estimate of the value function and the policy.

After running the code with value iteration approach, it only took 2 iterations so it takes the same number of iterations of evaluation and improvement but it requires 2 orders of magnitude less sweeps of state space hence it is significantly faster.

*In my other posts,I will be sharing light over the Monte-Carlo methods which are a Model-Free non-bootstrapped version of Reinforcement Learning.* 

  

