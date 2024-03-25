## Description

As part of a ML course, we were asked to work on an algo that meets the academic requirements for our RL exam. 
Thus, in a team we focused on a variant of the famous game Snake & Ladder. This variant of the game allows you to play with 3 different dice.
each with its own properties for advancing. The board has shortcuts and traps of different types that allow you to gain or lose time. 
The game also has two modes, ”circle” which turns the course into a continuous circle requiring the player to land precisely on the last one to finish the game; 
or ”no circle” which offers a classic course.


##Objective 

The goal is to converge to an optimal strategy (using Markov Decision Process). So leading to maximisation of the total reward r of successive steps and in our case of the Snake and Ladder to minimise the costs. 

Q(s, a) ← Q(s, a) + α [r + γ min Q(s′, a′) − Q(s, a)]

## Implementation Q-learning 

Three initial functions Safe Matrix(nbr steps), Normal Matrix(layout, nbr steps, circle) and Risky Matrix(layout, nbr steps, circle) were constructed in order to build the transition probability matrices that the value and policy iteration needed to work. These matrices allow the transition probabilities between states to be adapted according to the location of the traps, the dice used, the structure and the game mode.

markovDecision(layout,circle) applies the value and policy iteration method to the letter. It calculates in P1, P2 and P3, the transition probability matrices of the three dice thanks to the functions mentioned above. It then constructs the rewards vectors which depend on the dice used and the location of the type 3 trap (prison). Thus, going to jail implies additional rewards. Once the necessary inputs have been provided, the function applies the value and policy iteration defined on the previous page. 

It calculates the values of each state for each of the 3 dice and retains the minimum as the optimal value. In a second step, it retains the index of this minimum value to indicate which die is used to obtain this optimal value. All this is done in an iterative way until the changes between 2 iterations are too small. 

Several sub-optimal policy functions have been constructed to develop dice policies that differ from the optimal policy. These policies, which will later be used to prove that these strategies are less efficient than the optimal strategy, return a vector of dice to be used in each state. For example, the function Dice1 strategy(layout, cir- cle) constructs a policy that uses only dice 1. The function Purely Random Strategy(layout, circle) constructs a purely random policy.

Simulator(layout, circle, Dice, nbr sim, start) = the algo that plays the game conditionnaly to Dice strategy imposed on it, the location of the traps and the game mode. After playing thousands of games (nbr sim), simunlator() returns the average number of turns used to finish the game for the start box. 


