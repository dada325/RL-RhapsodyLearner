# RL-RhapsodyLearner
Rhapsody Learner is the framework of Deep Reinforcement learning 

This is for my research on the Algorithmic Aspects of Learning in Games, more specifically it is a learner which learn optimal strategies in complex, multi-agent environments where the other agents are also learning and adapting their strategies. 

But it is also an implementation of some algorithms in deep RL that is very classic, in JAX (WIP) and pytorch. 


# Multiagent

## Experience Sharing: 
One way to implement collaborative learning is to have each agent learn not just from its own experiences but also from the experiences of other agents. This can be achieved by sampling experiences from the memory buffers of all agents during the learning phase.

## Action Consideration: 
Another common technique in collaborative multi-agent learning is to have each agent consider the actions of the other agents when deciding on its own action. This can be done during the action selection phase.
