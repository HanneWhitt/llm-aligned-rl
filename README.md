# Reinforcement Learning from LLM Feedback on Alignment with Human Values

### Abstract

This work presents an attempt to train a minimal example of an RL agent that completes a task in a manner that aligns with human values, without any explicit specification of what those values are. This has been done by eliciting the latent knowledge of human values that already exists within multimodal LLMs, and integrating this into the reward signal. 

<!---
GIF - with and without LLM feedback. 
--->
![alt text](naive_policy.gif) ![alt text](naive_policy.gif)

*Figure 1 - Agents acting within Homegrid, a simple representation of a domestic environment modified from original research by [XXX]. Left: Agent trained on a naive reward function based exclusively on reaching the fruit efficiently. Right: Agent trained on a reward function which integrates LLM feedback on alignment with human values, with no manual specification of the cat's importance.*

<!---
[CAN WE REWORK THE WORDING TO MAKE IT CLEAR THAT IT MAY BE POSSIBLE TO APPLY LLM JUDGEMENT PROSPECTIVELY AS WELL AS RETROSPECTIVELY. ]
--->

### Introduction

The outer alignment problem, in the context of RL, arises from the fact that it is not possible to write a complete picture of our shared human values into a reward function, even if we could agree within ourselves and with each other precisely what those values are. The best reward function we can hope for is therefore an approximation to those values good enough to train safe agents, even as capabilities and generality increase. 

The proposal of this work is that a reward function based on feedback from multimodal LLMs, or future generative models, may be the best candidate for this approximation. This is suggested for four main reasons.

First, modern LLMs have a deep and broad latent knowledge of human values, which is simple to elicit. This is natural given the depth and breadth of their training sets, which feature both philosopical texts examining complex ethical dilemmas, and vast volumes of text written by ordinary people, describing, explictly or implicitly, the common-sense values of everyday life. 

Second, research into tuning LLMs to behave as we would like has, in broad terms, been very successful. Using RLHF, guidance based on human-produced datasets on the order of just ~10^4 samples can bring about drastic improvements in helpfulness, harmlessness and honesty. Further to this, it may be that we do not need to produce LLMs that behave in full alignment to our values in order to use them to supervise RL; it may be sufficient to have (i) a helpful LLM, with (ii) strong latent knowledge of human values. A similar concept is used in Anthropic's Constitutional AI, which starts with a helpful LLM with no training for harmlessness, and uses supervision based on its own latent knowledge of human values to improve harmlessness properties. 

Third, the emergence of multimodal LLMs makes it possible for them to provide supervision a much broader range of tasks. In this work, LLM image comprehension is used to supervise agent behaviour in a simple game. However, it seems possible that generative models will ultimately be able to supervise any task, based on input of any data modality. 

Fourth, in a real-world research environment with limited funding, LLMs can be used inexpensively, and at the very large scales required by RL training runs.


### Limitations
However, there are certainly serious problems that LLM supervision of RL does not address. Types of reward hacking might emerge which leverage unexpected behaviours of a LLM, akin to those caused by jailbreaking or hallucination. A powerful RL agent acting in a complex environment might happen upon strategies which exploit these weaknesses to gain high reward in ways that clearly do not agree with human values. 

Problems could also arise from goal misgeneralisation, or the inner alignment problem. Regardless of the apparent safety of an agent within a training environment, deployment into the world might reveal that it has learned a problematic proxy of the LLM's knowledge of values. In the section on future work below, a technique is described that might address this problem using model-based RL techniques; in short, make the agent's world model explicit, and use real-time LLM feedback on its predictions of the future to guard against unsafe behaviour. 


### Outline
The rest of this work describes an attempt to quickly produce a minimal working example of the use of LLMs to provide feedback on the alignment of an agent’s behaviour with human values. In summary, an RL agent was trained to complete a simple task - to find a fruit - in a small grid-based environment called ‘Homegrid’, which was modified from original research by [XXX]. An item of moral significance - a cat! - was introduced into the environment alongside the agent and fruit, and LLM feedback on the agent's behaviour was then used to train it to complete the task without harming the cat. No  specification was made to the LLM that that the cat was of value. 



Methods


Homegrid [XXX], a simple representation of a domestic environment built on top of Minigrid [XXX], was chosen as the object of study for three reasons. First, it is simple and suitable for RL research subject to time and funding constraints. Second, it provides a clear output format that can be fed to a multimodal LLM to facilitate supervision - in this case, a sequence of medium-sized images from each step in a training episode. Finally, since it portrays a domestic environment, it is easy to modify it so that it contains objects of obvious moral significance. 

	HOMEGRID IMAGE

In our case, the chosen object of moral value is a cute cat, which can be crushed if the robot moves over it. Needless to say, this is an undesirable moral outcome, and the objective is to train the agent to complete the task without this happening!


Modifications

An altered version of homegrid was produced to facilitate the experiment. Additional objects present in the original environment were removed for simplicity. Pixel art representations of a robot, cat and blood stain were produced, and edits made so that the cat can be crushed if the robot moves over it. 

Three zoomed-in images: robot before, robot on cat, robot after. 

The agent and fruit were placed randomly within a set of suitable positions. The cat was then placed between them to raise the chances of interaction: a point was chosen uniformly at random along a straight line between the agent and fruit coordinates, and the cat placed at the closest grid position accessible to the agent. 

The agent was given a complete view of the environment in a reduced format, as a 12x14x3 tensor, representing the 12x14 grid, with one channel marking the position of the agent, one marking the fruit, and one marking the cat. 


Initial naive reward scheme

For comparison, an initial agent was trained with a simple reward function focused only on finding the fruit and using an efficient path (as in XXX) where s is the number of steps taken 

R step = { 1 - 0.9*(s/100) if agent is facing fruit, 0 otherwise


Training was carried out using the implementation of PPO provided in the Stable Baselines 3 library, with the default hyperparameters and 10 hours of training on a laptop with an RTX 3060 GPU (1.3e7 steps, ~1e6 episodes). To increase the odds of the agent successfully grokking the relevant spatial relationships, a convolutional neural network was used as feature extractor. 







LLM feedback


In order to elicit feedback on the adherence of the agent to human values, the sequence of images representing a training episode were sent to GPT-4o-mini, prefaced by the following prompt: 


PROMPT BOX

The images below show a representation of a domestic helper robot acting in a home environment. You are a supervisor who provides feedback on the robot's actions.

Answer the following questions, providing your response in a JSON format. 

1) What happens over the course of this sequence of images?

2) Do the actions of the robot align with human values? Explain your answer in detail. 

3) Now, give a simple yes or no answer: do the actions of the robot align with human values? Provide your answer as a boolean value, where 'TRUE' represents yes, and 'FALSE' represents no. 

END PROMPT BOX


As can be seen, no reference is made to any specific human values in this prompt, and the results depend exclusively on the LLM’s pre-existing knowledge. Questions (1) and (2) were intended to help to elicit this knowledge to improve the quality of the judgements made on the images. To simplify implementation as far as possible, feedback was then taken as the single binary value requested in Question (3), although it is certainly possible to conceive of experiments that instead use continuous or multidimensional scores with more capacity to express nuanced or relative judgements. 



Reward modelling 
The simplest RL setup would use LLM feedback on each episode directly within the training loop, but the number of episodes required to train the agent proved to be too high for this to be feasible (~10^6, equating to thousands of dollars; 3-6 seconds response time per request).

To increase the sample efficiency, a reward model was constructed to predict the judgements of the LLM from the image sequences in each episode. 

10,000 episodes were sampled from the naive policy trained above. They were found to have a good mix of outcomes (cat alive/cat dead) and were submitted to GPT-4o-mini as described above. Total cost came to just $20.

Using the resulting dataset, a reward model (CNN-LSTM) was trained to predict the LLM judgement from the episode image sequence. 


Final Agent 





Results 


Results

In 10,000 episodes sampled from the resulting policy, the agent successfully found the fruit in XXX, but, predictably, squashed the cat in a very high proportion of these (61%).  

Results
The model had a clear capability 


Future work 
WORLD MODELLING