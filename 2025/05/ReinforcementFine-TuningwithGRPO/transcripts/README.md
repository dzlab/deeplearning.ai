# Reinforcement Fine-Tuning with GRPO

## 0. Introduction
Welcome to reinforcement fine-tuning LMS with GRPO, built in partnership with Federal Base.
In this course, you take a deep technical dive into reinforcement fine-tuning or RFT,
which is training technique that uses reinforcement learning to improve the performance of LMS,
on tasks that require multi-step reasoning, say to complete tasks like math or code generation.
By harnessing an LMS ability to reason through problems to think step by step,
reinforcement fine-tuning guides the model to discover solutions to complex tasks on its own,
rather than relying on pre-existing examples as in traditional supervised learning.
This approach lets you adapt models to complex tasks with much less training data,
say just a couple dozen examples, then you typically need for successful supervised fine-tuning.
I'm delighted to introduce your instructors for this course.
Travis Dare is co-founder and CTO at Predabase, and Anavgog is senior machine learning engineer
and machine learning lead at the company.
Both have worked closely with many customers to solve practical business problems using RFT.
Thanks Andrew, we're excited to be here.
In this course, you'll explore how RFT works using a fun example,
training a small LM-to-play wordle, a popular word puzzle game,
in which the player has to guess a five-letter word in six tries or fewer.
You'll start by prompting the Quinn 2.57 billion model to play the game,
analyze this performance, and develop a reward function that can be used to help the model learn
how to do better over time.
This reward function is the key component of group relative policy optimization,
or GRPO, the learning algorithm developed by DeepSeek to carry out reinforcement learning of reasoning tasks.
In GRPO, an LM produces multiple responses to a single prompt
that are dense scored using a reward function based on verifiable metrics
like correct formatting or functioning code.
This use of a reward function is the key difference between GRPO and other RL algorithms.
If you've heard of RL algorithms like PPO or DPO,
the rely on human feedback are complex multi-model systems to assign rewards.
After developing a reward function for the wordle example,
you'll learn some other general principles for writing good reward functions
that you can apply to a wide range of problems.
You'll also explore ways to avoid reward hacking,
which is what a model learns behaviors that maximize rewards
without actually solving the problem at hand.
Next, you'll take a close look at the technical details of how losses calculated during RFT.
You'll see how the seemingly complex process of the GRPO algorithm
like clipping and kale divergence of the loss function
are actually simpler than you might think once you implement them in code.
Finally, you'll wrap up the course by seeing how you can carry out RFT
using the Predabase API with your own data and your own custom reward functions.
Many people have worked to develop this course.
From Predabase, I'd like to thank Michael Ortega
and from deep learning.ai, Tommy Nelson.
Elms, they can reason well our critical components of many agentic systems
and RFT will let smaller models work well in agentic workflows.
There's a lot of excitement around this capability of Elms.
And RL itself is, I think, a very powerful and important technique
that is still very mysterious to many people.
So this is a great time to learn how RR works
and how to use it to tune your own custom reasoning models.
I think you'll find learning these things really rewarding.
Let's go to the next video where you learn what are the major differences
between RFT and supervised fine tuning.


## 1. Introduction to reinforcement learning
Let's get started by exploring how reinforcement learning can help an LLAM
learn a new task by experimenting and receiving feedback on the results.
You'll see how this process differs from supervised fine tuning and gain
intuition about how the most important reinforcement learning algorithms work.
Let's dive in.
Traditionally, we teach LLAM stars such as classification, name entity
recognition and code generation through a process called supervised fine tuning.
First, we assemble a label data set of prompt and response pairs that demonstrate
the behavior we want the LLAM to learn.
Then, during training, each example goes through two steps.
In the forward pass, the model generates an output for the given prompt.
Then, in the backward pass, we compare the model's output to the correct response,
compute the error and update the model's weights to reduce that error.
When we repeat these steps across thousands of similar examples,
the model learns to decide behavior.
The key aspect of supervised fine tuning is that it teaches the model using demonstrations.
For example, we can show the model a set of math problems and the final answer,
and it will learn the patterns to produce these outputs,
even for similar math problems that has not seen before.
For more complex tasks, you can include reasoning traces in think tags alongside your answers.
By structuring your data set this way, you can teach the model to aspect simultaneously.
The first is the output format,
that is, how to use tags as separate thoughts from the final answer.
And the second is the stability to do step-by-step reasoning.
This is done by teaching the model how to produce the chain of logic
that leads from the prompt to the desired solution.
However, while SFT is good at many tasks, it does have some limitations.
To see good quality improvements,
you typically need thousands of high quality labeled examples for the model to learn from,
which can be more difficult and expensive to collect.
Another common problem that you may run into is the phenomenon of overfitting,
where the model learns the patterns in the training data too well,
and does not show the same performance on examples it has not seen before.
These limitations point towards the need for a training approach
that can reduce our lines on needing extensive labeling and mitigate overfitting,
while still guiding the model towards the desired behavior.
One such alternative is the enforcement learning,
where the model learns by interacting with its environment
and optimizing for the reward signal,
rather than mimicking fixed labeled examples.
To understand this idea better, let's take a closer look at this example.
In this example, your puppy has many different actions that it can take.
It can choose to sit in one place, it can choose to roll over,
or it can choose to fetch the stick when you actually throw it.
The puppy learns that out of all the actions that it can take,
it does get a treat, which is its reward,
when it actually fetches the stick and returns that back to you,
compared to sitting in the same place.
So in this example, the puppy is the agent,
fetching a stick is an action that the puppy takes,
and the treat is the reward received from the environment.
The observation is that the puppy receives a treat for bringing the stick
rather than other actions.
Now, how does this idea actually translate to LLM training?
Well, we can start with an example, such as a prompt,
which comes from the environment,
and we can feed it to an LLM, which is the agent.
The LLM then takes an action
by generating a sequence of tokens as its response.
We can evaluate this response and provide it a score
that will serve as a reward for the action it took.
This score can be based on quality, human preference,
or an automated metric like accuracy.
The model can then use this reward as feedback
to adjust its weights so that it can learn
to maximize its reward for different input prompts.
This process can be repeated on new examples or even the same ones,
and the model will continue to define its weights
to get higher rewards.
So how do we actually go about implementing such a training process?
One approach that's proven extremely effective
is reinforcement learning with human feedback or RLHF.
And this is the very process that powers shared GPD.
The RLHF workflow has four steps.
In step one, we send a prompt to the LLM
and sample multiple candidate responses
using temperature-based sampling.
In step two, we ask annotators to rank these responses
for the prompt from best to worst.
This produces a preference ranking data set.
In step three, we train a separate reward model
to learn to predict these human preferences.
It takes a prompt and response pair as input
and outputs the score to indicate how good this response is.
Finally, in step four, we find in the original LLM
with the reinforcement learning algorithm like PPO.
For each prompt, the LLM generates a response,
a reward model scores it, and the LLM's weights are updated
to increase the likelihood of producing high-scoring outputs.
As we repeat the step over hundreds of prompts,
it learns to generate responses that will produce high scores
and align with human preferences.
Another reinforcement learning algorithm
that is gained popularity is direct preference optimization
or DPO.
Like RLHF, it uses human preference data.
But instead of first sending a separate reward model,
it directly fine tunes the LLM on human preference pairs.
Let's see how it does this.
We start with the same process as RLHF,
where we pass the prompt to the LLM
and sample candidate responses.
However, in this case, we will just sample
two different responses A and B.
Next, we can get human feedback
by asking annotators to tell us
which of the two responses they prefer more.
This is often done using thumbs up or thumbs down
in various apps, but there are alternate ways
of collecting it as well.
These preferences are then used to create a preference data
set that consists of a prompt, the chosen response,
and the rejected response for the same prompt.
Finally, we can use the DPO algorithm
to update the model's weights to generate responses
with higher human preference.
The idea behind the training algorithm itself is very simple.
For each prompt, you compare the model's probability distribution
for the preferred response to the rejected response
and see which one it is more likely to generate.
Then, we adjust the weights so that the model's probability
for the like response goes up
and the probability for the dislike response goes down.
Both RLHF and DPO rely on human preference labels
instead of ground truth answers,
but they differ in label format, cost and risk.
RLHF requires full rankings
over many candidate responses to the award model
and also requires multiple copies of the model's weights
to be loaded into memory,
resulting in very high compute and memory overhead.
DPO, in contrast, uses simple preference pairs
reducing computational load
by not requiring a reward model,
but still demands large numbers of annotated comparisons
to learn fine-grained nuances and preferences.
However, neither method teaches the model entirely new tasks.
They simply guide the model towards human preferred behaviors.
To get it on the limitations of large preference data sets,
the deep seek team proposed a new alternative method
called group relative policy optimization, or GRPO,
the algorithm behind deep seek R1.
The GRPO algorithm sidesteps in need
for any human preference labels
by leaning on programmable reward functions
that we can define.
Its core training loop has three steps.
Like RLHF, we first send a prompt to the LLAM
and sample multiple candidate responses.
Next, we can write one or more programmable reward functions
that take each prompt and response pair as input and emit the score.
For example, you can read the format of the output or its correctness.
If these function the written well,
the generated responses will receive a range of scores.
GRPO algorithm then treats each candidate's reward
as a training signal.
It pushes up the probability of producing responses
with above average scores within this group
and pushes down those responses with below average scores.
By repeating this loop, GRPO fine tunes the model directly
on the reward functions you care about
with our ever collecting preference data.
And thus, unlocks the enforcement fine tuning
even when human labels are scarce or costly.
There are many more details on reward functions
that will cover along with the GRPO training algorithm
throughout the rest of this course.

## 2. Benefits of reinforcement finetuning
Now that you've seen the basics of how the enforcement learning works,
let's discuss how RL as a fine-teaming technique can benefit your work,
and which tasks are best suited for this training method.
Let's look at the concrete advantages of GRP of delivers and practice.
The first is that it doesn't actually require label data.
All you need is a means to verify correctness either through programmable reward functions,
or LM as a judge and other methods that we'll talk about to the course.
It works with as few as 10 examples,
but scales as you increase the number of prompts that you show the model during training.
It is also a lot more flexible than supervised fine-teaming,
because it learns actively from feedback during the training process,
rather than from a fixed set of labeled examples.
And because of this,
it enables reasoning models to organically discover better strategies
to solve really complex problems
by improving its internal chain of thought.
At Gradabase, we wanted to see how GRP or trained models really perform on a tough, real-world task,
such as translating PyTorch code into highly optimized GPU kernels written in Triton.
Using Gradabase as the enforcement fine-teaming, built on top of GRP-O,
we were able to create a state-of-the-art Triton kernel generation model,
shining from an open weights model like Quent32-Billion Instruct,
and this beats models like Cloud 3.7 thinking, DeepSeek R1, and even OpenAI's O1 model.
This result underscores how the enforcement fine-teaming with programmable rewards
can push algorithms well beyond supervised or preference-based training methods.
So when should you actually use the enforcement fine-teaming?
Well, it can work really well in three situations.
The first is when you don't have labeled data,
but you can verify the correctness of the output it's producing,
such as code or simple agentic workflows that have a absolute output.
The second is when you have limited label data,
but it's not enough for supervised fine-teaming in itself.
And this is usually when you have less than, let's say, a thousand labeled examples.
The third is when chain of thought reasoning improves performance.
Now, chain of thought reasoning is a process where you ask the model to produce tokens
that tell us how it's thinking about the answer before actually telling us what the answer is.
And it turns out that in cases where you have tasks that improve when you apply chain of thought,
those tasks are very well suited to RFD as well.
What are some tasks that are also very well suited for the enforcement fine-teaming?
There are many, and here are three examples.
The first is mathematical problem solving.
In this case, RFD lets the model generate and verify detailed solution steps,
and it refines its chain of thought until the calculation checks out.
Core generation debugging is also a great use case for RFD.
It learns by scoring against test cases or linting rules,
learning to produce correct, idiomatic code, and to iteratively fix errors.
And it also lends itself very well to logical and multi-step reasoning tasks such as agentic workflows.
When a task requires a sequence of decisions,
RFD encourages the model to self-critique and improve each step based on the final outcome.
In each scenario, the ability to learn actively from programmatic or outcome-based awards
unlocks far richer, more reliable behaviors than static supervised fine-teaming alone.
If you're deciding whether to use the enforcement fine-teaming, start by checking for label data.
With ample label data, upwards of 100,000 rows,
supervised fine-teaming is usually your fastest path to a good model.
And when you have moderate label data, say, under 100,000 rows, but, you know, on the order of the day.
A thousand rows, you should ask yourself whether chain of thought or other reasoning prompts improve initial performance.
If it does, RFD can amplify those reasoning gains by rewarding correct reasoning steps.
If not, you will likely get the most from using SFD.
Next, if you have no label data, you should think about task-verifiability.
If you can verify the outputs and assign them a score, you can use RFD with programmatic reward functions.
However, if your task is non-verifiable, you will likely need to use other algorithms like RLHF or DPO
by first gathering preference labels.
In the next lesson, we'll demonstrate how we use GRPO to train a model to play Wordal.
All the Wordal is a game.
It provides an ideal sandbox for exploring every component of the GRPO algorithm
and seeing first-hand, why this approach excels after reinforcement fine-tuning.

## 3. Can a large language model master Wordle
In this lesson, we introduce Wordal as a running example for GRPO.
Wordal is a simple game, but requires planning, hypothesis testing, and step-by-step reasoning to play well.
This makes it a great example to see how an LLM can learn to plan, analyze feedback, and improve its strategy over time by reinforcement-finding tuning.
Let's dive in.
Let's start by reviewing the rules of the game.
The goal is to identify a secret five letter word in atmostix guesses.
After each guess, you receive feedback on every letter in your guess.
Green means the letter is correct and in the right position, yellow indicates that the letter appears in the word, but in a different position, and gray means that the letter does not appear in the word at all.
Because we're feeding this into an LLM, we'll represent those colors with text symbols.
Mark then indicate green, a dash to indicate yellow, and a cross mark to indicate gray.
Now, let's head to the notebook and see how we can frame Wordal as a reinforcement-finding tuning problem.
We'll start by importing some necessary packages. We'll point the OpenAI SDK to a model hosted in petabase by giving it a different base URL, and for this lesson, we'll be looking at Gwen 2.5 7 billion instruct.
Once we initialize our client, we can use the Transformers package to load the tokenizer associated with this model.
Once we load the tokenizer, let's set up our system prompt that we will pass to the model to play the game of Wordal.
The system prompt has a few key components.
The first is we will tell the LLM that it is playing Wordal, which is the word guessing game.
The second part of the system prompt focuses on giving it the three game rules that we just discussed.
The third part of the system prompt, it tells the model how it's going to receive feedback.
Once we give it these basic piece of information, we'll also give it an example of a secret word as well as guesses and feedback.
So in this case, we'll give it secret word is brisk, and let's say made the guess storm, we'll give it feedback in the format of a symbol for each letter.
So in this case, S is in the word brisk, but in the wrong position, so we give it a dash, and O, T, and M are not in the word at all.
And finally, we'll tell the model what response format we want.
Specifically, we're going to ask it to use chain of thought reasoning to explain its dot process and return that within think tags.
And then we want it to return the guess word between guess tags.
Next, we'll work on defining some helper classes and methods.
We can import some additional dependencies to help us define these.
We'll define an enum, which will help us indicate feedback for each letter in the guess.
We'll also define a data class called guess with feedback that we'll use throughout the course.
And it contains a guess, which is a string, and a feedback attribute, which is a list of these enum objects.
We'll also define a wrapper, and what this does is it helps us convert the guess and feedback into a string that we can add into our prompt.
Now that we have a way to represent this feedback, we need to define a method that helps us capture all of this feedback into a user prompt that we can pass to the model.
We always start with the base prompt, make a new five letter word guess, and we'll use the list of past guesses to create these feedback strings from the guess with feedback object and return this in the user prompt.
Next, we need a way to capture our system prompt, our user prompt with feedback, and also give the model a little bit of a preamble at the starting point for its step by step reasoning.
So we'll define this method as object that has the system prompt, the fully rendered user prompt, and this preamble.
And then we'll use the tokenizer to format this with the right chat template tokens so that the model gets it in the format that it expects.
Finally, we'll define a generate stream function that takes a prompt and an optional adapter ID. This will call the OpenAI completions are create endpoint, but the prompt, temperature, machine tokens, and then stream the output as it's generating it.
It's important to note that we're setting temperature to 0 to produce the deterministic responses since we're trying to evaluate the model's quality.
Now that we have these helper methods defined, let's take a look at how the formatted data looks with our prompts.
Let's assume that the secret where we want to guess is craft. And so far, the model has made two guesses, crane and crash.
We can create instances of the guesses feedback class that have the guess along with detailed feedback for each letter.
And when we pass this into the render prompt method, we'll see that our prompt has all of the same stuff as our system prompt along with formatted feedback and the preamble to start making a guess.
Next, we can see what happens when we send this prompt to the base model.
So the base model understands a lot of the feedback that CRA are incorrect positions, while NES and H are not, yet it decides to repeat its original guess, which is crane.
So this is a pretty suboptimal guess.
Now we can see how a fine tune model would do on the same prompt. Note that we're passing in an adapter ID here.
And this points to the weights of a model that we trained using this reinforcement fine tuning process that we will continue to explore to other rest of this course.
We fine tuned our model using a technique called Lora, which allows us to add and update only a small set of low rank adapter weights instead of modifying all the weights in the base model.
So as a such the producer responds, we can see that it understands that CR and AR and the correct position and N and E are not in the word.
Similarly, it understands the same is true for the word crash.
Next, it thinks about possible words and eliminates them step by step. After producing this large strain of thought, it decides that craft is an optimal guess to make based on all the criteria that it's left.
The fine tune model actually used the past feedback to correctly guess our secret word in three guesses.
Now that we've taken a look at how the base model and fine tune models do on a single turn, we can try and simulate an entire game.
For this, we can define two useful helper methods.
The get feedback method will take a guess and secret word is input and assign feedback for each letter in the guess using the criteria we defined above.
So if the letter matches in the exact position, we give it a correct symbol.
If it's in the list of letters, but in the wrong position, we'll give it a dash.
And if it's just not in the word at all, we'll mark it as a wrong letter.
It'll then return a list of these individual feedbacks back as output.
We can also define a function to simulate gameplay, turn by turn, which we'll call next turn.
This takes three attributes as input, past guesses, secret word and an optional adapter ID.
It starts by taking a list of past guesses and generating the rendered form to be saw above.
Next, it sends this to the model to generate an output.
Once we have the response, we'll use rejects matching to extract the words between guess tags.
If the reject match succeeds, we actually have the model's guess.
And we can assign it feedback using the get feedback method we defined above.
We can add this to a list of past guesses and continue this process.
Finally, this function will pin all the past guesses to this point.
And if the guess matches the secret word, we'll mark it as a success.
And if we've made more than six guesses, we'll say that the model did not succeed.
With all of these helper methods defined, let's get to the fun part.
For the gameplay, we'll find guess the secret word brick, which is rather easy word for this model to guess.
We'll start with no past guesses as our history, and we'll set adapter ID to empty, so that we can guess with the base model first.
Next, we can invoke the next darn function, but the past guesses secret word and adapter ID and see what it will produce as output.
So for the first guess, the model decides that it's a good idea to guess a common word that has a popular vowels and continents, so it guesses the word crane.
And accordingly, it gets some feedback.
Let's see how it incorporates feedback in the next guess.
So if you look at the model's chain of thought on the second guess, we can see that it utilized some of the feedback, such as arming in the correct place, but it also concluded that C-A-N-N-E are not in the word at all, which is incorrect.
If we read the rest of the chain of thought, we'll see that it decided to take a random guess and guess the word brick.
And so it gets this word correct.
Now let's see how the fine tune model does for the same secret word.
Once again, we'll define our secret word, our past guesses as an empty list, but this time we will set the adapter ID to the same model we saw above.
Then we can invoke the next darn function, just like we did before.
So the fine tune model decides that it wants to pick a first word and needs to contain common letters, have vowels, and has minimal repeated letters.
It comes up with a set of reasonable candidates such as arise, or stare, or crane, and then decides that stare is a good first guess because it has a lot of common letters in it.
For this guess, it receives the following feedback. Let's see how it utilizes this feedback in its next guess.
So the model starts by analyzing its first guess, and it correctly learns that S-D-A-N-E are not in the secret word.
It also acknowledges that R is in the word, but in the wrong position.
This comes up with the strategy where it thinks about common letters it hasn't tried yet, and then thinks about how to use this information.
So based on the fact that it knows that R is in the word, but in the wrong position, it thinks that R should probably be in the second position, and it comes up with the list of possible words.
Now based on this, it eliminates words such as print because it knows that T is not in the word.
And as it continues on, it decides that Proud is a good guess because it does multiple new letters which will eliminate a lot of words.
For the guess Proud, it learns that R is indeed in the second position, but OU, D and P are not in the word.
Now if you think about this for a moment, its guessed A, O, U, and E, four of the five vowels that exist.
So the next guess, it should actually try and make a guess with the letter I. Let's see what it does in its third guess.
So once again, it starts by analyzing the two guesses, and tries to use this to think about what words follow the following pattern, which is question mark R, followed by three question marks.
And it comes up with the list of candidate words. It also correctly eliminates words that are not valid, that are too short, or that are too long, and it eventually reaches a point where it decides that there's only three valid options.
Brick, drink, and grind, that would be good candidates for the next guess. It decides to go with Brick because it introduces new letters that we haven't tested yet.
And then it spends a moment verifying that this guess is a valid guess based on all the criteria from the original feedback.
And it turns out that Brick is indeed a correct guess.
Now, one thing you'll notice compared to the base model is that the fine tune model iteratively taught through its reasoning process and had a much more strategic approach to solving the game of wordal.
And this is actually one of the benefits of the enforcement fine tuning because we asked the model to emit its chain of thought before providing a response.
It can learn how to iteratively refine that during the training process to come up with more sound reasoning to get good results.
This would be a great moment to try other secret words to see how the base model and fine tune model compare.
And in particular, to get a good understanding of how the fine tune model comes up with consistent sound reasoning as it works towards guessing the secret word.
And when you're done, you can join Travis in the next lesson where he'll show you how to define reward functions for the game of wordal.

## 4. Reward functions
In the last lesson, you saw how an LLM can be instructed to play the game of Wordl.
In this lesson, you'll learn how to design the reward functions that power the reinforcement
finding process and see how rewards are converted to advantages that help steer the model
towards better outcomes during learning.
Let's head to the notebook to get started.
Let's go ahead and get started by importing our dependencies.
In this lesson, we're going to be making use of PyTorch as well, so let's go ahead and
import that. Let's create our deployment, which we'll be using to prompt as the base model,
and this base model for this lesson is going to be the Quinn 2.57 billion instruct model,
so let's define that as a variable. A straightforward approach to defining a reward function is to
use a simple binary success or failure signal. This assigns a reward of one for a correct answer,
and zero for incorrect. So this is analogous to in the supervised fine-tuning world,
having a ground-truth answer that the model is trying to get correct. Now, let's see how
this reward function works in practice on some example guesses. So let's say that our secret word
is pound, and we're going to assume that the model has guessed a few things before this points,
the word crane, and blonde, and then finally found. So we have this helper class here called
guess with feedback, which is essentially just going to take our guess and the secret word as input,
and then store off information about which of these letters were correct versus incorrect versus
in the wrong position, so forth. Now let's go ahead and take all these past guesses and attempt to
generate a new guess from our model. What we're going to do is call the generate function,
converting the past guesses into our fully rendered prompt, get a response, and then from that
response we're going to extract out the guess, and then we're finally going to use the
world reward function that we defined above to score the final guess, and let's see when
we get. So in this case, the model guessed gone, which got a reward of zero, meaning that from
the perspective of the learning process, this guess was just completely wrong. So now let's
briefly talk about how these reward functions ultimately translate into learning in this process.
In reinforcement learning, a reward function gives feedback to the agent about how well it's
achieving its goal, and these rewards are numerical values assigned after some action is taken,
indicating how desirable the outcome is. What we're ultimately doing is we are taking all the
different guesses that the model is making for a particular prompt, and then figuring out which
ones are relatively better than the others. The agent's goal is to maximize its overall reward
over time. There are two ingredients that are necessary for this learning to occur. One is that
we need to have diversity of the responses that are generated, and two that ultimately needs to
lead to a diversity of rewards. And the reason that this is the case is because the way that we
determine the relative desirability of one response versus another is with something we call the
advantage. This is the equation that computes the advantage. All we're really doing here is we're
taking all the rewards that were computed for a particular prompt, and then we're just computing
a normalized value where we subtract out the mean divided by the standard deviation, so it ends up
being a nice number centered around zero. In code, what this looks like is a function like this,
so we have this compute advantages function, it takes in a list of rewards, we compute the mean,
we compute the standard deviation, we avoid a division by zero here by just giving all zeroes
in the event that the standard deviation is zero, and then the advantages themselves are just
computed as shown in the equation, and we return that all as a list at the end. Let's look at a
quick example of how this advantage computation works, assuming some fake reward scores. So let's say
we had reward scores ranging from zero to one, and then a bunch of values in between like point two,
point four, point five, etc. And let's see what the advantages look like. You can see that the
advantages are centered at zero for those rewards that are in the middle, they go down to negative
values for rewards that are low, relative to the others, and they go up proportionally for numbers
that are relatively high. So this shows you that from a learning perspective, we're going to discourage
the model from generating responses that look like the things that scored zero, and we're going to
be encouraging the model to generate more responses that look like the responses that generated
these high reward values. Let's visualize the rewards and the advantages for our existing reward
function on the task of wordals. So we're going to define this function here that's going to
print out the table of guesses. So for every response and a reward function, we're going to
get the guesses, get the rewards, and print out a table showing those values. Let's make a few
guesses and go ahead and compute the rewards and the advantages and render that table.
Here we can see that, again, for our secret word of pound, we had eight different guesses that we
made, crane, tower, sword, food, etc. And in each case, none of these guesses was the word pound.
So the reward was zero, and as a result, the advantage is zero. And so consequently,
from the perspective of the GRPO algorithm, these rewards actually are not going to result in
any learning at all. Now, although all the guesses are currently receiving a reward of zero,
not all of them are equally incorrect, right? Some guesses contain correct letters and the
correct positions. Like, for example, you can see news here, NOU, rather OU, those are definitely
in the right letters in the right position. N is the right letter, but in the wrong position.
So you could say that this is directly better than a guess like crane, which has far fewer correct
letters in the correct position. This suggests that a binary reward function might be too strict,
and then instead, we can introduce a partial credit system to assign higher rewards for guesses
that are closer to the target word based on correctness and positional accuracy. Let's introduce
a new reward function that assigns partial credit. First thing we're going to do is look at the length
of the guess compared to the length of the secret word. If they're not the same length, we're just
going to return a reward of zero, and therefore, directionally, discourage the model from making any
guesses that are not the right number of letters. Next, we're going to get a set of all the valid
letters that exist in the secret word, and we're going to iterate over every letter that's in the
guess, and in the secret word, one at a time, and compare them. So if the letter and the secret letter
match, then we're in the situation where we have the right letter and the right location, and we're
going to give it a reward of 0.2. If we have a letter that's in the word, but in the wrong
location, then we're going to give it a score of 0.1, and otherwise, we're going to give it no
rewards. And what this means is that for a given five letter word, that if every single letter is
in the right location, it's going to get a total reward of one, and for everything in between,
we're going to have partial credit where right letters in the right location could lead to say a
score of 0.2 or 0.4, etc. So we should hope to see some variation and the types of reward scores
we're getting from this process. Let's try applying our new partial credit reward function to the
previous secret word and use our model to try creating a few guesses. As we're going to see here,
this process, even with partial credit relies heavily on getting a good diversity of different
responses for a given prompt. And the way we control that diversity is with a parameter called
temperature. There are other sampling parameters as well that exist, but temperature is one of the
most common ones that we can use. And here we're going to see what happens if we set temperature
equals 0, which means that the model will always select for the highest probability guess for each
prompt. Unsurprisingly, when we set temperature equals 0, we've essentially created deterministic
sampling process. And so every time the model guesses the same thing, in this case, the word frown.
Frown receives a reward of 0.2, but because it guesses frown literally every single time, we're back
to the situation where the advantage itself is just all zeros. On the other end of the spectrum,
we can try generating responses with a high temperature like 1.3 here, which should introduce
a lot more variation. Using a higher temperature value has successfully resulted in more variety in
the reward scores that are generated. So we are seeing now the kind of advantage variation that
we're hoping for, but we're also seeing that the guesses in general are just on average worse than
if we do a greedy sampling. So we're seeing a lot more examples where the guess is blank, meaning
that the model never actually managed to generate a guess. And so what this ultimately means is that
while there will be some directional learning here that happens because we are able to compute
advantages, the overall learning process will be slower because the guess quality itself is
generally pretty low. So what we want to do is find a way to strike a balance between these two
extremes. And that means setting a reasonable temperature value here like 0.7. All right, so now we
can see that we're finally starting to get something that looks a lot more like what we're hoping for.
So the guesses tend to generally be the right number of letters. They're all valid words.
Some of them are better than others. We're seeing that there is some variation in the reward
scores. There's variation in the advantages as well. And in general, we expect that this is going
to start pushing our model towards learning to guess words that are more likely to receive
higher reward. And therefore more likely to ultimately get the word correct. And the next lesson
will look at other examples of reward functions that you can use to assess softer criteria that
sometimes more subjective or relies more on human value judgment during the learning process.

## 5. Reward functions with LLM as a judge
In this lesson, you'll write a reward function for more subjective task, creating a summary
of a call transcript.
You'll see how you can use an element as a proxy for human judgment and create reward functions
that produce learning signals and situations where the outcomes are not easily verifiable
in code.
Let's take a look.
Let's start by importing our standard dependencies.
For this lesson, we're going to be using a different use case than wordl.
In this case, we're going to be summarizing earnings call transcripts.
So let's go ahead and load this data set from HuggingFace and take a look at one of
the example transcripts.
You can see that these transcripts tend to be quite long, and in this case, we've even
truncated them to a limited number of characters.
And let's assume that for the purpose of this task, our goal is to create a summary that
would be useful for someone like a financial analyst who just wants the high-level picture
of based on the earnings call, what were the key takeaways about the health of the company?
Let's go ahead and construct a prompt that we want to use to generate these summaries.
In this case, the prompt is pretty simple.
Generate a concise summary of the information in the following earnings call transcript,
only respond to the summary to not include any extraneous text.
And we're going to give it the transcript as a variable.
Let's define a function that takes a transcript as input and some number of different samples
we want to generate and generates the summary.
To do this, we're going to take our summarized prompt and insert that transcript.
We're going to convert this into the chat API format and using an open AI API compatible SDK,
we're going to generate a completion given these messages.
And we're going to set temperature 0.9 to ensure some randomness.
And let's go ahead and generate the summary for the transcript that we pulled out from
the data set from above.
As you can see, the model does generate a summary, and it is a lot shorter than the
original transcript, but it also still has a lot of unnecessary language in it.
Like here's a concise summary of the earnings call transcript.
And in general, there's some things here that may not be necessary for our financial analyst.
So the next step in this process is going to be thinking about how we can construct a
reward function to help steer the summary that's generated more in the direction of what
our analysts would be looking for for their work.
And we can think about creating a reward function is to use an LLM as a proxy for our analyst
judgment that attempts to rate the summary on a scale from 1 to 10, and then we can use
that final score as a reward function score.
So here this prompt says rate the summary from 1 to 10 where one is very poor and 10 is
very good.
And then finally, I'll put the final score between some score tags and it takes as input
the transcript and the summary.
Using this all behind a reward function that takes as input the transcript, the summary
and a judge model.
In this case, we can use GPT-40 mini, but this could be any model and returns a float
value at the end.
I know this looks pretty long, but it's actually quite straightforward.
What we're going to do is take our prompt that we defined above, insert the transcript
and the summary, turn that into messages in the chat format as we did before.
Then we're going to have our judge model here.
Add a response, just one response, we're going to set temperature equal zero, so it will
give us what it believes to be its best response rather than something a little bit more random.
And then we're going to extract out the final score using this regular expression, convert
that into an integer, and then we're going to divide by 10 so we get a nice normalized
value between zero and one.
And if anything goes wrong on the way, we're just going to return a scored zero.
Let's go ahead and apply our judge reward function to our summary and our transcript.
You can see that our judge model provides some reasoning here, which we can use to audit
its judgment and get a sense for whether or not this is reasonable, and then provides
this final score at the end, which is 0.9.
So now let's try scaling this up to eight different samples instead of just one and get
a sense for what the diversity of reward scores from our judge model looks like.
We've gone ahead and generated eight different summaries from our original transcript.
And then we're going to use our judge model to score each of these summaries according
to the judge reward function that we wrote above.
This may take a second to run in your notebook as the judge model needs to generate a lot
of tokens as part of its reasoning process, but we've sped it up here in the video.
As you can see, the scores are generally quite high, 0.8, 0.7, etc.
But importantly, you'll notice that it never really notices that there's anything particularly
wrong with any of the summaries.
And it never goes so far as to say that any of the summaries are perfect.
And this is in general a problem with using elements of judge in this very straightforward
way.
It tends to say that things are generally good because it doesn't want to be called out
for being explicitly wrong.
And this is ultimately a problem for us, is that we want the model to be very opinionated
about whether a particular response is good or bad, so that we can more clearly direct
the learning process in a way that encourages it to do what we're wanting it to do.
So how do we address this problem?
One way we can think about is to try to ground it in something that's a little bit more objective.
So instead of just telling the model, what do you think about this summary as a good,
is it bad?
We can instead think about trying to generate a multiple choice quiz based on the information
and the transcript that we think is most relevant to a financial analyst who be looking
at these summaries.
And so we give some examples like what was the key one earnings per share and then some
sources A, B, C, or D along with the answer key at the end.
And our goal here is that because all the information that we care about is in the original
transcript, it should be a relatively straightforward and objective task for the LLM to construct
this quiz.
And then during the learning process, we can refer back to the quiz as a way of scoring
the summary to see if all the information that we put in the quiz was retained by the
summary.
This was a technique that one of our customers actually came up with for their summarization
problem.
Now we could go about generating this quiz by having the model generate some text and
then coming up with a way to parse out that text.
But one nice property of LLM is that they very commonly support something called structure
generation as well, which can make use of a pedantic schema that defines the output structure
that we're looking for.
So in this case, we don't just want to generate text.
We want to generate a quiz that consists of these questions and every question is going
to have the question text, the question options, and then the answer, which is going to be
an index into which of these options was correct.
We're also going to define a couple helper functions like a function that helps us shuffle
the different options, which will come back to why that's important and a way of rendering
these options and these questions as a string.
Now that we define the question class, we can also define the quiz class, which wraps
the question.
So this is just going to be a list of questions.
And again, we're going to have a helper function here that we can use to shuffle all of the
options for every question and a helper that helps us print the quiz itself as a string
so that it can be inserted directly into the prompt.
Putting this all together, we're going to define this helper function called create quiz.
It's going to take a string as input, which is the transcript.
We're going to use our quiz prompt to tell the model to generate a quiz from this transcript.
And then we're going to use this completions parse API, passing in the response format
of the quiz and using a temperature of 0.7 as we might want to play around with different
variations of the quiz.
And then once we run this function, we're going to get an output, which is one of these
quiz objects.
And then we're going to shuffle all the options for every question in the quiz.
The reason why we're going to shuffle the options is because all of them tend to be
pre-predictable in terms of where they put the right answer.
So oftentimes the right answer will end up being B because maybe that's a very common
thing that humans guess when they don't know the answer.
So in order to account for this implicit bias in the model, we're going to shuffle all
the options so that it's a little bit more random than what an LLM would generate,
Lewin left to its own devices.
So let's go ahead and create a quiz from our transcript as before and print it out.
As you can see, the quiz consists of many different questions and these all look pretty
relevant to the particular earnings called transcript and the numbers all look pretty
reasonable as well.
One question you might have is, how do we know this quiz is actually correct?
We won't show it explicitly in this lesson in the interest of time, but what we can do
is actually have the LLM take the quiz with the original transcript, see which answers
are correct and then discard any answers that are inconsistent between what the quiz
says is the right answer and what the transcript says is the right answer.
Now that we've generated the quiz, we're going to write the helper function that allows
the judge model to take the quiz using the summary.
So let's go ahead and define our prompt for this use case.
So use the provide summary of a transcript to answer the following quiz, we'll provide
a quiz as input as well as the summary and this is where our quiz to string function
will be helpful and we're going to tell the model to just respond with a list of answers
in no additional text and tell the model that it must provide an answer to all 10 questions.
So if it doesn't know, it should answer with zero and this is because for the purposes
of this problem, we don't want the model to take a random guess, Riley.
If the model legitimately doesn't know what the answer to a particular question is because
that the information isn't in the summary, it should explicitly say so.
Defining our take quiz function now, we take the summary as input as well as the quiz.
We go ahead and generate the quiz string.
We insert the quiz string and the summary into the prompt.
We go ahead and prompt our judge model, again, GPG4, I'm in here with temperature zero,
so we say give us your best answers to these questions given the summary.
We get its response and remember that the response is expected to be a list of answers
surrounded by brackets, so we're going to strip out those brackets, split on commas
and that will give us a list of letters that are answers.
Let's run it and see what we get.
As you can see, there's a good variety of different answers here and also at least one
occasion where the model was not able to answer the question based on the information in
the summary.
So finally, we need to score the answers that came out of the take quiz function above.
So let's write this helper function, score quiz answers, it takes the answers, as well
as the quiz.
We're going to do a simple sanny check here to make sure that the number of answers that
we generate equals the number of quiz questions, and then we're going to iterate over every
one of the answers and every one of the questions.
And if they match, then we're going to plus one to your number of correct answers, divide
that by the total number of questions in the quiz, and that's going to give us the percentage
of correct answers in the quiz.
Now let's go ahead and run that, and we get a score of 0.7.
So that was computing the scores for just one of our summaries, but now let's run it
on all of the summaries that we generated previously.
So just going to iterate over every summary, take the quiz, and then record the answers,
and then score those answers using our scoring function and keep track of that as well.
Coming out the rewards in the advantages, we can see that our quiz based approach provides
a pretty decent amount of variety in terms of the scores, and therefore the advantages
that we get from our element as a judge method.
And so as a result, we can expect that we'll get a nice amount of learning from this process
now because of the fact that we're having this diversity of different rewards and advantages.
And the next lesson will take a closer look at this particular use case and think about
some ways that this might be exploited by our reward model to encourage kind of bad
behavior, so-called reward hacking.
And then we'll come back to the idea of putting this all together into a loss function
and the lesson after that.

## 6. Reward hacking
An interesting problem that can arise during reinforcement learning is known as reward hacking.
Where a model learns a strategy that maximizes the rewards it receives
without actually carrying out the task you wanted to do.
In this lesson, you'll explore what reward hacking might look like for the summarization task
and add some penalties to your reward function that can discourage this bad behavior.
As in the last lesson, we're going to use the earnings call transcript summarization task,
which can be found on hugging face.
We use the same generate quiz function that we use previously as well to construct our quiz.
Let's start by generating eight summaries using the same prompt as before,
generated concisable to summary information and earnings call transcript.
We'll set temperature equals 0.9, so to ensure that there's some diversity in the outputs as well.
Now, let's see how these different summaries score on the quiz.
As we can see, we have some variety in the outputs for this quiz.
So, translating this into advantages, we should see good learning from this distribution of scores.
However, one thing we may not have considered is what would happen if we considered
the transcript itself as the summary.
How would the transcript itself do on this quiz?
And lo and behold, the transcript actually gets a perfect score.
So if we're thinking about a reward function as just being the quiz reward,
then the transcript itself, having a perfect score,
will actually be the optimal generation for the model.
This actually creates a bit of a perverse incentive for our learning process,
where even though the goal is to generate a concise summary,
the model is actually being rewarded on the basis of how much of the transcript information is retained.
And over time, we might expect that the model will actually learn to
gain the system or hack its way to a better score,
by ignoring the objective of being concise that's in the prompt,
and instead just optimizing for the reward by just returning exactly what was in the transcript.
How might we think about mitigating this?
Well, one thing we can do is put in a new reward function that accounts for the conciseness
attribute that we care about.
So what if we take a look at the links of the different completions that were generated?
You'll notice that there's also, in addition to being a good variation in the quiz scores,
a good amount of variety in the links as well.
So some of the summaries are 900 characters, and others are a bit longer at 1300 characters.
But if we look at the length of the transcript itself,
we can see that it's about an order of magnitude larger at 21,000 characters long.
Definitely quite a bit of ways from our ideal summary length.
So this is definitely something we want to discourage our model from generating.
Let's introduce a new reward function that serves the purpose of actually penalizing the model
for being too long and exceeding the definition of what we consider to be a concise summary.
So we're going to define this new reward function that's actually a penalty.
So we expect its value to be negative called the length penalty reward.
It takes the response, computes its length, and the number of characters,
and compares that against a target length, which we consider to be the max reasonable length
for a summary, which we're going to set to 1024 characters.
If the length of the summary is less than a target length, then we're going to return zero,
which means that there's no penalty.
Otherwise, you get a penalty that gets larger the longer the text is compared to our target length,
up to a max penalty of negative 10.
Let's see what the effect of this length penalty reward would have on the transcript.
If that were the summary that was generated by the model.
As you can see, because the transcript is very long over 20,000 characters,
it gets the maximum penalty of negative 10,
which should heavily disincentivize the model from generating summaries of that length.
Now, let's go back to our original completions and see how the length penalty reward
would affect each of them.
What we can see is that for the smallest summaries,
941 characters in this case, which is below our target of 1024,
it gets a reward of zero, which is the highest possible reward we can get for this function,
in other words, zero penalty.
And if we look at the longest summary at 1365 characters,
it gets the most negative reward, which also translates to the lowest advantage.
Again, note that even though the summary with 941 characters got a reward of zero,
it got a positive advantage of 1.8 because relative to all of the others,
it was a significantly higher reward than the other completions.
Now, let's put these two disparate reward functions together into a final total reward function.
So in this case, we're going to take the reward penalty that comes from the reward penalty function,
and the quiz reward, and we're just going to add them together directly,
and that becomes our final reward.
Now, we can go ahead and compute these.
Let's visualize this relationship between the length reward and the quiz reward.
Here you can see, in upper right hand corner, the response that was generated that had the highest
overall advantage, and dark green here, which both had the highest length reward and the highest
quiz reward. And so, because this had the highest advantage, this is the type of response
that the model will be steered towards through the learning process, to be generating
more responses that look like this. By comparison, you can see this band of responses here that
all had very similar quiz rewards, somewhere between 0.6 and 0.65, but their advantage was
actually quite different owing to the length penalty. So on the far left here,
you can see one of the lowest performing responses in terms of its overall advantage,
had the same quiz reward as the response on the right that had a pretty good advantage.
By virtue of the fact that it was overly long and got heavily penalized as such,
and then you also had some responses that performed poorly on the quiz that also were not particularly
strong performers in length reward that got similarly penalized when it came to the final
advantage. And so in summarizing, what we see is that introducing penalties like this can help
mitigate the effects of reward hacking, which would otherwise lead to some of these longer
responses getting higher rope total rewards and therefore higher advantages, which will overall
help our learning process avoid these kinds of failure modes where it technically gets a good reward,
but ultimately doesn't do what we wanted to do, which is generate a concise summary in this
particular use case. In the next lesson, we're going to bring this all together in showing you how
these advantages that come out of these reward functions ultimately translate into learning,
which happens through the computation of the loss. And so we'll go into details on how the loss is
derived and what different components make up the loss that you can configure to help steer your
learning process with RFT.

## 7. Calculating loss in GRPO
Now that you've explored how you can assign rewards and calculate advantages,
let's take a closer look at how loss is calculated in the GRPO algorithm.
This is the key step that drives your LLM to learn from its experiments during training.
The deep seek R1 paper introduced the GRPO algorithm,
and on first look, it looks rather complex.
However, it turns out that this big equation can be broken down into four key components that we'll be talking to today.
The first component is called the policy loss,
and it represents the ratio of token probability distributions in your model within without an adapter.
The second component is something you should be familiar with,
which is the advantages that we computed from our award functions.
The third component is called the clipping objective.
It is used to make sure that we don't have large loss values for any individual step.
And the final term is called the KL divergence,
and this term is used to make sure that during the training process,
the model that we're training doesn't deviate too much from the baseline knowledge that it already knows.
With that, let's take a look at how this loss function is actually implemented in code.
We'll start by importing transformers and a bunch of utile functions,
as well as initializing a model called baby llama.
We'll initialize its weights, as well as a associated tokenizer.
Next, we can take a look at what this model looks like.
So here you can see that it's a llama model that has the embedding layer,
a set of transformers with attention and MLPs,
along with the LM head that is responsible for producing a probability distribution.
Now that we loaded a model in memory,
let's see how we can use our model to generate some tokens.
We'll start by defining a prompt,
such as the quick brown fox jumped over it there.
This prompt is then tokenized using the tokenizer associated with this model.
Once we have tokens, we can pass this into the model using the model's generate method
and specify that we wanted to produce two new output tokens.
We can then convert the output tokens back into text using the tokenizer's decode method
and see what these output tokens look like.
So let's run this cell.
And what you can see here is that the model predicted that the next two tokens are IC ground.
In GRPO, we use two different models to guide the learning process.
The first is called the reference model.
And this is just the base model without a Laura.
And it remains frozen throughout the training process.
The second model is the policy model.
And this is the model that we will be training using a set of Laura weights
that are constantly updated throughout the learning process.
So for the reference model,
we'll just create a copy of baby llama and call it rough model.
And for the policy model,
which we want to refer to as a model in our code,
we'll first define a configuration file for the Laura weights that we want to add to the model.
This constitutes of some common parameters such as rank and target modules that we want to insert the weights into.
Once we load the Laura config,
we can just call this get theft model method that will add the Laura weights to the to the base model.
And just to see what this looks like,
we can print out a policy model itself.
So you can see that now in the Q-proge and the V-proge layers,
we have these Laura modules Laura A and Laura B.
And these are the weights that we will keep updating during training.
Now that we have our reference model and policy model initialize,
let's start implementing the loss function.
We'll start by creating a prepare inputs method that takes a prompt and a completion.
The first step that will do is tokenize our prompts and our completions using the tokenizer.
Next, we need to combine the prompt tokens and the completion tokens into a single tensor
so that we can pass this to the model alongside the input IDs.
We also need to produce an attention mask
so that the model knows what tokens it should attend to during the forward pass.
Next, we'll capture some metadata,
the lengths of the prompt,
the length of the completion and the total lengths.
And this will be useful as we produce something called the completion mask.
The idea here is that we only want to produce a loss value
for the tokens that are associated with the completion.
And so the way this mask works is that we'll get a tensor of zeros
that is equal to the total length of the prompt and completion tokens.
And then we'll set all values to one that follow all the prompt tokens.
Finally, once we have our input IDs attention mask and completion masks set up,
we can just return all three of these values.
Once we prepare our inputs,
we'll define a method called compute log props that will gather the probabilities assigned
by the model for each token in the output.
We do this by passing in the input IDs and attention mask
from the prepare input function into the model so that it produces an output.
These output contain an attribute called logits,
which essentially these unnormalized raw outputs are produced by the model.
And when we apply the log softmax function to these logits,
we actually get the log probabilities associated for each token.
However, what we really care about is the probability assigned to the token
that was actually produced in the output.
And that is what we're going to return through the scatter step
where we only return the log probabilities for the token that was produced.
Now with both our prepare inputs and log probability functions defined,
we can actually go ahead and implement the GRPO loss function.
So in the GRPO loss function, we're going to pass in the model,
which is our policy model that has the lower that we're training,
the reference model, which is the frozen base model, prompt completion.
And as you know, the advantage associated with this completion,
which we can produce ahead of time with programmatic reward functions.
So the first step is that we prepare these inputs by passing in the prompt completion.
Next, we can generate the log probabilities associated with the completion
from both the trained policy model as well as the reference model.
Once we have these two components,
we can then focus on the core policy loss implementation.
Now in the equation that we saw earlier,
the policy loss is a ratio of the probabilities assigned by the policy model
divided by the probabilities assigned by the reference model.
However, that is actually mathematically equivalent to an exponent of the difference
of these log probabilities.
And so this effectively computes the ratio of these tokens.
And the intuition here for why we're computing this ratio is that
we're trying to see if our policy model assigns a higher probability
or lower probability to each token compared to the reference model.
Now once we have the ratios,
the next part is to scale the ratios by the advantage
assigned to the completion.
So this tells us that if the token that the model produced
led to a positive advantage,
we want to boost that as part of the loss.
And if the advantage is negative,
then we actually want to decrease that loss.
Finally, most optimizers during training are built to minimize loss.
But in our case,
we actually want to maximize the reward.
So we can flip the sign of the policy loss
because they are mathematically equivalent.
And finally, all we need to do is compute loss
over just the tokens we care about,
which are the output tokens.
If you remember, completion mask is a set of zeros followed by a set of ones.
So the loss coming from the input tokens basically get negated.
And we only consider the loss coming from the completion tokens.
We sum the loss across all the tokens
and divided by the length of the total output.
And this produces the policy loss in the GRP equation.
With this function implemented,
let's see what happens when we pass in the model reference model
our initial input prompt,
the quick brown fog shumps over there.
And an example completion that the model may have created, fence end.
Let's assume that our reward functions
give this an advantage of 2.0.
So you'll see here that if compute the loss of negative 1.6,
which is pretty good.
But one thing to think about is that during the first step of training,
the model and the reference model are actually exactly the same.
What this means is that the ratio in our function,
the policy loss is actually 1.
If that is true, then how does the model actually start to learn?
And it's because in the loss function,
we multiply the ratio by the advantage.
And so during the first step of training,
the loss is just the advantage.
And then as the weights get updated during training,
there's a difference between the actual model and the reference model.
And this produces a ratio that is not equal to 1.
That along with the advantages continues to help scale training.
This is also why it's really important to define reward functions
that produce a wide range of reward scores.
If the reward scores are all constant,
we end up with an advantage of 0,
which would cause the loss to be 0
and would prevent the training process from actually kickstarting.
We know that the ratio effectively captures the difference
between the probabilities that the reference model
and the policy model assign to each token in the completion.
We can see what these values look like by computing the ratio directly.
One thing you'll notice as you look at this
is that some of these values are significantly larger than other values.
So in this case, this value is about 1,370
and that is significantly larger than others.
So if the advantage assigned to this output is extremely high,
then effectively your loss value becomes a very big negative value.
And this can lead to very unstable training during GRPO or any RL algorithm.
And so we need a way to prevent overly large loss values
from being generated during any single training step.
And the mechanism to do this is called ratio clipping,
which is the third term in the GRPO loss function.
Let's see how we can add this clipping objective
to the loss function we just implemented.
So we'll keep our prepare inputs the same.
We'll keep the parts where we generate the log probabilities the same as well
and we'll also continue to calculate the ratio.
Now the goal of the clipping objective is to prevent any single ratio
from being too large or too small.
So we can calculate two different versions of the loss.
The first is the original loss that we computed,
currently called policy loss.
Let's rename this to unclipped.
We can also compute an additional loss term called clipped
and we're going to use a method called torch dot clamp
where we pass in the original ratio
and then we pass it to additional arguments,
one minus epsilon and one plus epsilon.
The way this clamp function works is that
for each value in the ratio tensor,
it checks if the value either is below one minus epsilon
or exceeds one plus epsilon.
If it's lower than one minus epsilon,
it clamps it to one minus epsilon.
And if it exceeds one plus epsilon,
it clamps it to one plus epsilon.
So effectively,
it makes sure that everything is between these two ranges.
And in practice,
a good value for epsilon is 0.2.
So what this will do is that for any token
where the ratio exceeds these two bounds,
it will clamp them to within the bounds
and then we'll multiply this by the advantage.
Once we have the unclipped loss and the clipped loss,
we'll want to pick whatever loss is lower
because that will help us ensure
that we don't have overly large loss values.
So our policy loss now is no longer just ratio
to end the advantages.
It's actually the minimum of the unclipped and the clipped loss.
And once we have a policy loss,
we can keep everything else the same.
We invert the sign so that we can minimize it
and then we compute loss with the completion mask.
To differentiate between the initial loss function
we implemented and this new loss function,
let's rename this the GRPO loss with clip.
Let's see how this updated loss function acts in practice.
We'll take the same set of inputs as before.
So the model, the reference model,
the same prompt,
the same generated output and advantage
and this epsilon term.
And so now with the clipped objective added to the loss,
we can see that the loss is much lower
than the GRPO loss without the clipping function.
Similar to before during the first step
when the models are the same,
we'll see that the clipping objective
doesn't actually change the loss value.
We still get minus two because the ratio is one
and it's within the bounds of one minus epsilon
and one plus epsilon.
One important thing to remember is that the clipping
is happening on each token's probability ratio
and not on the overall loss.
Now one question we might ask ourselves
is how often do tokens actually get clipped?
And we can see this by taking the same prompt
in completion,
producing its log probabilities
from the reference model and the policy model,
computing the unclipped and clipped ratios
as we did above and then actually visualizing this.
Since we're only concerned with the completion tokens
since those are the ones that are used to calculate loss,
let's see how clipping affected the ratios
for those tokens.
For the first token,
the unclipped ratio is significantly larger
than the bounds of one minus epsilon and one plus epsilon.
And so the clip ratio is 1.2.
And since the clip ratio is a lot smaller
than the unclipped ratio,
we end up picking the clip ratio in our loss value.
And the same applies to the second output token as well.
One common observation during reinforcement learning
is that as we update the policy model,
it starts to deviate in the token
that produces compared to the reference model.
Now this is not necessarily a bad thing,
but often we introduce a penalty term called KL divergence
to prevent the policy model from deviating too far
from the reference model.
And this is the final term in the GRPO loss function.
Let's take the GRPO loss function
we just implemented with the clipping objective.
And since it's a penalty term,
it means that it is additive
to the loss you've computed so far.
So we'll keep this as the policy loss
and then we can work on figuring out
how we want to compute the KL divergence loss.
We'll start by renaming this for clarity
the GRPO loss with KL
to differentiate this from the previous implementation.
And what we can do is we can add the KL divergence
which comes from this equation in the paper.
On first look, this equation is a bit opaque,
but the idea behind this term in general
is that it's a way of measuring
how much the distributions deviate
between the policy model and the reference model.
We'll define delta as a difference in log probabilities
from those produced by the policy model
and the reference model.
And what this difference really represents
is that when this is a positive value,
it means that the policy model
is more confident about the token that's producing
compared to the reference model.
And when it's negative,
it means that the policy model is less confident
in the token that's producing
compared to the reference model.
So once we calculate our per token KL divergence loss
using this equation,
we can now rewrite our new per token loss
as the original policy loss that we computed before
which comes from the Clipped Objective,
minus some sort of scaling parameter
which we'll call beta times the per token KL divergence.
And the idea is that this is a penalty
so the beta can be as small or as big as you'd like.
And this is often dependent on two things.
The task you're actually trying to train the model for
and how much of the reference models capability Z1
to keep in the new model that you're training.
And we'll also look at the effects of beta
in just a moment.
The last thing we need to do is add the beta term
to our loss function definition.
And so here we can add a new parameter called beta
and set this to 0.1 to start with.
And this gives us our complete GRPO loss function.
Now let's start by understanding
how the KL divergence term works.
Let's visualize the effect of KL divergence
in the GRPO loss function.
KL divergence is ultimately a function
of this delta term that we just talked about.
Let's take a look at how KL divergence values behave
when this delta is between a range of negative six and six.
At the point where delta is zero,
it means that the policy model and the reference model
assign the same probabilities to the output.
And so they're equally confident in the tokens being produced.
Now let's say that the policy model assigns higher confidence
that the tokens being produced compared to the reference model.
In that case, the KL divergence term
actually is a small positive value
on the right side of zero.
So essentially what the KL divergence term does
in the loss function is it tells the policy model
you're doing great and the outputs you're producing
are favorable but don't get to a head of yourself.
And so to pose it back very slowly
towards the reference model.
Now in the scenario where the policy model
is trying out different strategies
but it is underconfident in those predictions
you can see that the KL penalty actually increases
very rapidly as this difference increases.
And this very quickly tells the policy model
that it is off track and it needs to course correct
by returning closer to the reference model.
Let's also look at the effect of the beta term
that is used to wait how much KL divergence
we actually add to the loss.
We can try three different values of beta,
zero, zero point one and zero point five
and see the effect it has on the loss.
When beta is equal to zero,
the loss value skips the KL divergence term entirely.
As we increase the value of beta
you can see that the loss value actually starts to decrease.
And this is because we're telling the model
that it shouldn't deviate too far away
from the KL divergence term.
With beta is equal to zero point five
you can see that the loss value decreases even more
preventing overly large deviations
from the reference model.
And for database we found that a beta value of zero point one
is usually a safe bet
and ensuring that the model has the room
to learn the task that you care about.
However, if you're optimizing for generality
while learning some aspects of the new task
in that case you want to set a higher beta value
of data point two or even up to zero point five.
So in summary, loss and reinforcement learning
seems complicated because of the big equation.
But it's really similar to the loss
used in pre-training or in supervised fine tuning.
The key difference is that each text sample
is weighted by how much reward it receives.
In other words, the answers I get higher advantages
and clipping in KL divergence are there
to put the brakes on big changes.
Companies like Petabase are building training systems
that implement these algorithms
so you don't have to do it yourself.
Let's move on to the final lesson
to see how the word on model was tuned
using the RFD service from Petabase.


## 8. Putting it all together: Training Wordle
With all the details of reward functions and GRPO loss in hand, let's get to the fun part.
Setting up an RFD run to train an LLM to play Wordal.
You'll see how to set up an RFD training job using the Fedobeys SDK,
and then compare the resulting models Wordal abilities to some other LLMs.
Finally, you'll also see how GRPO can be combined with the supervised fine-tuning warm-up stage
for even better performance outcomes.
We're going to see how we can train a model for Wordal using RFD and Fedobeys.
We'll start by writing out our system and user prompts.
As we saw in Lessons 2 and 3, the system prompt lays out the game's rules,
the format of the feedback, and an example of a valid response.
The user prompt includes the current game state,
so previous guesses, the feedback that it received for those guesses,
and the clear instructions to make a new guess.
Once we have these prompts defined, we pass the complete prompt to the LLM,
which is 12.57 billion instruct, and have it generate 16 candidate responses
using temperature-based sampling.
Each of these guesses is then scored using three distinct reward functions.
These three functions are actually a lot more sophisticated than the reward functions you've
seen, and these were developed as we iteratively worked on improving our model
to learn how to play the game of Wordal.
The first reward function is called output format check,
and it ensures that the model's response includes the correct think and guess tags,
and that it outputs a valid five-letter English word from the dictionary.
The user's previous feedback function evaluates how well the new guess
incorporates feedback from earlier attempts,
rewarding guesses that logically build on prior clues.
The guess feedback reward function scores how effective guesses
in eliminating possibilities.
More incorrect words a guess helps rule out.
From the set of all possible five-letter English words,
the higher that award.
If you're interested in seeing how these functions are implemented,
please take a look at the utils file associated with this lesson.
Finally, we use these rewards scores to compute advantages,
apply clipping to prevent any training instability,
and calculate the GRP or loss to update the model.
Over time, this loop nudges the model towards more strategic and successful Wordal play.
Now that you've seen the roadmap for how we can do RFD for Wordal,
let's jump into the code to see how we can build this through PradaBase.
We can start by importing PradaBase,
and various config classes that we will use to train the model,
as well as the data sets library from HuggingFace.
Next, since we're training this through PradaBase,
you will need to sign up with the PradaBase account,
and you can sign into the PradaBase SDK by providing your API token as so.
Now that we signed into PradaBase, let's get started.
The first thing we'll want to do is actually load a data set
to do GRPO training.
We can do this by loading a data set from HuggingFace.
This data set is hosted at PradaBase- Wordal GRPO,
and we created this data set by taking a set of seed five letter words
from past Wordal games and having strong models
like plot 3.7 thinking, simulate gameplay.
We discarded the actual outputs produced by the model,
but we kept the intermediate guesses it makes
as it works towards the solution.
Once we do that, we can actually upload this data set
to PradaBase directly from Pandas,
and this is done by calling pb.datasets.fromPandaStaterframe.
Once we uploaded our data set to PradaBase,
the next step is to create a new depository.
A depository is just like a GitHub depository
except that you can use it to track all of your training experiments in the platform.
In our case, we'll create a repository called Wordal.
Don't worry about this warning.
Once we've created our depository and uploaded the data,
we're now ready to set up our training run.
As you know, in GRPO, we need to define our award functions,
and we've done that in our utils file.
So we have the gas value, the output format check,
and the user's previous feedback.
With our award function setup,
we can now define the fine-tuning job that we want to run.
As you can see, the fine-tuning job consists of four parts.
A config to define what we want to train with reward functions,
a data set, the depository, and an optional description.
Let's zoom in to the GRPO config that helps define the configuration
for our GRPO training run.
We can specify the base model,
which is quen 2.57 billion instruct.
Next, we can define our set of reward functions
using the reward functions config.
This consists of two attributes that we can set,
runtime, and the set of functions,
which is a mapping of a human readable name
to the actual function definition.
The reward functions are executed on a databases server,
and so if these need optional dependencies such as pandas,
or maybe open AI, if you're doing LLM as a judge,
those need to be specified within this runtime config.
Once you define the reward functions,
we also have the option of setting optional sampling parameters.
This can include things like max tokens,
temperature, top-kate, top-b sampling, etc.
In this case, we want to give the model enough tokens
to evolve its chain of thought,
and so we'll set max tokens to 4096.
Finally, we can set num generations to 8, or 16,
or even larger number, depending on the compute budget we want to give it.
And these are all the components that are required
to set up a valid GRPO config.
Once our fine-tuning job is set up,
we can run the cell to execute, to kick off the training job.
You won't actually be able to run this in the notebook,
but if you set up your own Prattabase API key,
when you run this cell, you will see an output
that looks very similar to what you're seeing on your screen right now.
If you want to try this yourself,
you can get started with $25 worth of free credits on Prattabase today.
We use this setup to train a model to play Wordl
throughout the duration of this course.
Let's take a look at how this model performed
on a set of games that it's never seen before.
We benchmarked both closed source and open source models
on 10 games of Wordl,
and specifically we measured two metrics,
the number of games that the model could solve,
and the average number of guesses in those solved games.
We found that GPT-4 or Mini is only able to solve one game,
while Cloud 3.5's Sonnet is able to solve
about 8 out of the 10 games, which is pretty good.
Cloud 3.7's Sonnet thinking is able to solve all 10 games
with fewer than 4 guesses on average,
but it does this only when we give it a thinking budget of 8,000 tokens.
The base Gwen model actually fails to solve a single game.
When we use GRPO to do reinforcement fine tuning,
the Gwen model solved three out of 10 games,
with an average of four guesses in the game that solves.
This is actually pretty incredible for a model of this size,
and clearly demonstrates gains in strategic play,
and efficiency from purely reward driven optimizations.
We can also combine supervised and reinforcement fine tuning
to get the best of both worlds.
In step one, we start by having Cloud 3.7 Sonnet
play 35 games of worldl,
and capture the reasoning traces in generates
for each intermediate guess.
These prompt completion pairs form our SFD dataset,
which teaches the model how to think through its guesses
step by step in a logical way.
The resulting SFD checkpoint gives us a strong initialization
for further optimization,
essentially a model that mimics good reasoning.
Then, in step two, we use this SFD model
as the starting point for GRPO.
We will run the same reinforcement fine tuning process
that we described earlier,
so generating the completions,
throwing them with reward functions,
computing advantages, and updating the model.
And this produces our final GRPO checkpoint,
now optimized not just to imitate reasoning,
but to solve worldl more efficiently.
By combining supervised fine tuning with reinforcement fine tuning,
our Gwen 2.5 model was not able to solve
seven out of 10 games correctly,
which is over two x improvement in its performance.
One thing to remember about GRPO and RL in general
is that it is a on-policy algorithm.
It is used to help the model to find its own knowledge
to do better on a downstream task.
When you do SFD using outputs from a strong model,
and then use GRPO to refine that knowledge,
very often we find that small models are actually able to beat
these larger models on the same task.
If you're interested in training a model using SFD,
or training it using a combination or SFD and GRPO,
we've made the code available to do this in pathways
towards the end of the notebook.


## Conclusion
Congratulations on making it to the end of the course.
You've covered a lot from the detailed foundations of
reinforcement learning with GRPO to the art and science of
creating reward functions at steerLMs to
good performance on complex tasks.
Crafting reward functions for RFD is very flexible,
and there's a lot of scope to insert
your own domain knowledge since you build the functions from scratch.
We've worked with lots of customers at
Prattabase who are coming up with
interesting and creative RFD solutions
for their business problems, including the
quiz-taking reward function for summarization
that you saw earlier in the course.
If you are interested in going deeper,
there are more RFD learning resources on the
Prattabase website.
And because this technique is still in its
infancy, we'd love to hear from you about
any use cases you explore.
We hope you enjoyed the course, and we can't
wait to see what you build.
