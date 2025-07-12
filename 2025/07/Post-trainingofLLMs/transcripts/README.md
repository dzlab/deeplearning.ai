# Post-training of LLMs

## 0. Introduction
Welcome to post-training of LMS, taught by An Ho Du, whose assistant professor at the University of Washington as well as co-founder of Nexus Low.
An Ho Du has trained and post-trained many models and I'm delighted that he is a instructor for this class.
Sex Andrew, I'm excited to be here.
Training a large language model has two phases.
Free training where a model learns to predict the next word or token.
From a compute and cost point of view, this is the bulk of training and may require training on trillions or tens of trillions of tokens of text.
For very large models, this could take months.
Then post-training.
This is where the model is further trained to perform more specific tasks such as answering questions.
This phase usually uses much smaller datasets and is also much faster and cheaper.
In this course, you learn about three common ways to post-training and customize OMs.
In fact, you will download the pre-trained model and post-training yourself in a relatively computationally affordable way.
You learn about the techniques supervised fine tuning or SFT and direct preference optimization, also called DPO, and online reinforcement learning.
Supervised fine tuning trades a model on label prompt response pairs and hopes to learn to follow instructions or use tools by replicating that in pre-pronged and desired response relationship.
Supervised fine tuning is especially effective for introducing new behaviors or making major changes to the model.
In one of the lessons, you'll fine tune a small quen model to follow instructions.
Direct-pression optimization or DPO teaches a model by showing it all's good and bad answers.
DPO gives a model two options for the same prompt, one preferred over the other.
DPO, through a contracting loss, pushes a model closer to boot and away from bad responses.
For example, if the model says, I'm your assistant, but you wanted to say I'm your AI assistant.
You label I'm your assistant as back and I'm your AI assistant as a good response.
You will use DPO on a small quen a struct model to change its identity.
With online reinforcement learning, the third or the three techniques, you give the prompts, it then generates responses, and then a reward function scores the quality of the answers.
The model then gets updated based on these reward scores.
One way to get a reward model to get reward scores is to start with human judgments on the quality of responses.
Then you can train a function to assign scores to the responses in a way that's consistent with the human judgments.
The most common algorithm for this is probably proximal policy optimization or PPO.
Another way to come up with rewards is via verifiable rewards, which applies to tasks with objective correctness measures like math or coding.
You can use math checkers or for coding, use unit tests to measure in an objective way if generated math solutions or code is actually correct.
This measure of correctness then gives you the reward function.
A powerful algorithm for using these reward functions is GRPO or group relative policy optimization, which is introduced by deep seek.
In this course, you use GRPO to train a small quen model to solve math problems.
Many people have helped in creating this course.
I'd like to thank Olexi Krishalev from in the year and Tiantao Zhao from UC Berkeley.
From Deeplander AI, it's now Gagari also contributed to this course.
The first lesson will be an overview of post-training methods.
In this lesson, you learn when you should do post-training as well as what is the menu of post-training options you can choose from.
Let's go on to the next video to get started.

## 1. Introduction to Post-training
In this lesson, we will learn basic concepts of post-training methods.
Let's start with, let's first see what is post-training.
Usually, when people train language model, we start from a rendering initialized model
and do pre-training first.
So here, we try to learn knowledge from everywhere, including from Wikipedia or CommonCraw,
which is crawling from all the internet data or GitHub for coding data.
After pre-training, we'll get a base model that is able to predict next word or token,
where each token is a subword highlighted in the figure here.
So starting from this base model, we will do post-training as the next step, which
is trying to learn responses from curated data.
This includes chat data or tool using or agent data.
So after this procedure, when you get an instruct model or chat model, which is able
to respond to instructions or talk to the user, when there's a question what is capital
or press, the model will be able to answer the question, saying the capital or press
is powers.
After this step, we can even do further or continue post-training, which tries to change the model
behavior or enhance certain capabilities of the model.
And after this, we arrive at the customized model, which is specialized in certain domains
or have specific behaviors.
So in this example, it might be able to write a better SQL query for any instructions
here.
Let's take a bit of master's use during ARM training.
To better understand post-training method, let's actually first start from the pre-training
method, which is usually considered as on supervised learning.
So usually, one can start from a very large scale on labeled tech corpus, which includes
Wikipedia, Common Core, or GitHub, etc.
So one can usually extract more than 2 trillion number of tokens from this corpus and train
on all of them.
So usually, we train on a few paragraphs or sentences.
And as a minimal example, one might see sentence like I like cats.
And in this case, we're trying to minimize the negative log probability for each token,
condition of all the previous tokens.
So it would be first minimize negative log probability for i, and then the negative log
likelihood for like given i, and then for cats given i like.
So in this way, we're training the model to predict next token given all the prior
token scene.
After pre-training, it will be followed by different post-training methods.
So one simplest and most popular post-training method is supervised fine tuning or SFT.
It's considered as a supervised learning or imitation learning, where when you create
a data set, that's a labeled prompt response pairs, where the prompt is usually the instructions
to the model and the response is the ideal response the model should respond with.
In this case, when you really only need from 1,000 to 1 billion tokens, which is much
less than the scale of pre-training, and the biggest difference in the training loss
is that we only train on the tokens of responses, but not the tokens of the prompt.
So besides supervised fine tuning, we also have other more advanced post-training methods.
The second one is direct preference optimization or DPO.
In DPO, where you really create a data set is a format of a prompt and a good and bad
responses.
So for any given prompt, one can generate multiple responses and select one that is considered
good and select the other that's considered bad.
And we try to train the model so that it pushes away from the bad responses and learns
from good responses.
So in this case, you really only also need from 1,000 to 1 billion number tokens, and
one has a more sophisticated loss function for this direct preference optimization, which
will go over in the specific lesson later.
The third method in post-training is online grandfather learning.
So for online grandfather learning, when you really only need to prepare the prompt
and a reward function.
So whenever we start from a prompt, we usually ask the language model itself to generate
a response, and we generate a reward for that response using a reward function.
And we use that signal to replace the model.
So in this case, when you have like 1,000 to maybe 10 million or more number of prompts,
and the target here is to maximize reward for the prompt and response, where the response
is actually generated by the language model itself.
Usually post-training requires getting three elements crack.
The first one is a good code design of data and algorithm.
As we discuss, there are different choices of post-training algorithms, including SFT,
DPO, or different online grandfather learning algorithm like Rainforest, RRO, GRPL, or PPL.
The nature of the algorithm will require a slight different data structure to prepare.
A good code design of data and algorithm will be really important for your success of post-training.
The second element is a reliable and efficient library that implements most of the algorithms
correctly.
This includes Hagenfist TRL, which is one of the first library that's simple to use and
implements most of the algorithm mentioned here.
Throughout this course, we will be using this TRL for most of the coding practices.
Besides Hagenfist TRL, I would also recommend you to try out more sophisticated and memory
efficient libraries, including OpenRTRAF, VRL, and NemoRL.
So the third element here would be an appropriate evaluation suit, when it needs to understand
after and before post-training what is needed as an evaluation suit that we need to track
the model performance and ensure that the model is always performing well.
Here we have an incomplete list of popular language model evaluations that when you
track in these days.
So the first one, Chatbot Arena, is a human preference for chat, where people can vote
for which model is better in their own taste.
And as a surrogate to human preferences, there are also different LMS as a charge for
chat models.
This includes ApakaEvo, MTBench, or Arena Heart.
There are also different static benchmarks for those in track LMS, where a live code bench
is one of the popular coding benchmarks.
And ARME, 2024, 2025, can be a recent popular mass evaluation data set for hardcore mass
questions.
There are also knowledge and reasoning related data set, like GPQA or MMMA Pro.
There are also instruction following evaluation data set, like IFEvo.
For function calling and agent, there are also different data sets for evaluation, which
includes BFCL, NexusBench, TauBench, or 2centbox, where those TauBench and 2centbox focus more
on multi-ton to using situation.
By listing all the evaluations here, I'd like to mention here that it's easy to improve
any of the benchmarks, but it can be much harder to improve some benchmark or change certain
model behavior without degrading other domains.
Throughout this course, we will be exploring which method gives the best improvement without
degrading other domains.
Lastly, I want to mention that it's not necessarily in every use cases you have to do post-training
of your model.
So there are different scenarios where there might be different methods that are more
appropriate for your use case.
For example, if you just want the model to follow a few instructions, like do not discuss
something sensitive or do not compare your company with some other company, one can easily
do prompting to make this happen.
So usually such prompting method can be simple yet brittle.
In external cases, the models may not always follow all the instructions you provide here.
A second use case might be about query some real-time database or knowledge base, in
which case, a retrieval augmented generation or search-based master could work better
since it can adapt to rapidly changing knowledge base here.
There are also scenarios where you'd like to create a domain-specific model, like
medical language model or separate security language model.
So in those cases, usually what really matters is a continual pre-training followed by
a more standard post-training to make the model first learn the knowledge, then learn how
to talk to the user.
So in this case, for continual pre-training, what you'll need to inject a very large-scale
domain knowledge that's not seen during the pre-training dataset.
And ideally, those domain knowledge should be at least more than 1 billion number of tokens.
And lastly, if your use case is about following 20 or more instructions tightly, or you really
want to improve some target capabilities, like create a strong SQL model, a function
call-in model, or a reasoning model, this is where post-training
can be most helpful.
It can help to reliably change the model behavior and improve target capabilities.
So if post-training is not done correctly, it might degrade other capabilities that you
didn't train on.
So in this lesson, you have learned about what is post-training, how to do post-training,
and when to do post-training.
In the next lesson, we will have deep dive into the first method of post-training, which
is supervised functioning.
All right, see you there.

## 2. Basics of SFT
In this lesson, you will learn basic concepts about supervised fine tuning, including the
method, common use cases, and principles for high quality data curation in SFT.
Let's dive in.
So unit SFT can be considered as imitating example responses.
You can start from any language model you want, which can predict a response given the
prompt.
It can be a base model, where when the user asks the question, the base model might
just predict the most likely token in the next word.
So it might just follow and predict very similar question instead of answering the question.
In order to perform SFT on those space model, we usually need to create some labeled
data in the format of user questions and ideal assistant responses.
The data might be in the format of telling me about your identity, and the assistant
will respond saying, a llama or any model that you want it identity to be.
The user might also ask, how are you, and the assistant can say, I'm doing great.
By preparing a large data set of such labeled data, we're ready for doing SFT and imitating
those example responses provided in the labeled data.
The way SFT works is by minimizing the negative log likelihood for the response given
the prompt.
And when you really take the sum over all the labeled data in this case, we'll go deeper
into this loss function is the next slide.
After performing SFT on base model, when you get a fine tune model or an instruct model,
which is able to respond to any user query properly if done correctly.
So let's take a closer look at the formula here.
Actually, SFT, as I'm minimizing the negative log likelihood for the responses, where minimizing
the negative log likelihood is equivalent to maximum likelihood and use a cross-natchable
loss here.
So for any data of index i, where the i-stata is just a specific prompt response pairs,
the loss for SFT will be the negative of that log probability of the response given
the prompt, where it can be further written as the negative log likelihood where the
likelihood is a product of the probability for the tokens in the responses, given all
the prior tokens, including the prompt tokens.
So in this way, we trend the model to maximize the possibility of outputting your provided
response given the prompt.
That's why SFT is trying to imitate those example responses here.
So there are a few best use cases or most appropriate use cases for supervised fine
tuning.
The first one is when once you jump start a new model behavior.
So it might be the case where you want to turn a pre-trained language model to an instruct
model or the case where you want to turn a non-reasoning model into a reasonably model.
Or there might be specific scenario where you want the model to use certain tools without
providing the tool descriptions in the prompt, and the model would just assume that it already
has access to the tools and call the tools in responses.
In those cases, SFT will be very ideal for jump-studying such model behaviors.
And the second is to improve certain model capabilities.
And one scenario I'd like to highlight here is to distill capabilities for a smaller model
by training on a high quality synthetic data generated by a larger model.
So in this case, you are essentially distilling a larger model capability into a small model
using supervised fine tuning.
So there are some principles of recommended ways to do supervised fine tuning data curation.
So the column methods for high quality SFT data curation include following few examples.
The first one is distillation.
As we discussed before, one can generate those responses from a stronger and larger instruct
model and let a smaller model to imitate those generated responses.
The second one can be a best of K or rejection sampling, where one can generate multiple responses
from the same original model that you want to train on.
And you can select the best among them using either a reward function or some other automatic
method.
One can get the best response and try imitating those best responses generated by the model
itself.
And the third case would be a filtering idea, where you can stuff from a very large scale SFT
data set collected from hugging phase or from your internal database.
Then you filter them according to both the quality of the responses and the diversity of the props
to get a smaller scale SFT data set that's of a higher quality and diverse enough.
Besides the column methods mentioned here, I'd also like to highlight that,
usually in SFT data curation, the quality is much more important than quantity for improving
capabilities.
If you have 1,000 really high quality and diverse data, that can usually outperform the
SFT results of 1 million mixed quality data.
The rationale behind this is that SFT usually requires imitating all the data provided by
you.
If there are some really bad responses in the mixed quality data, the model will be forced
to imitate such response and thus equating the performance.
So data quality here can be really important for the success of SFT.
Lastly, I'd like to highlight one orthogonal direction in model tuning that's completely
parallel and orthogonal to any post tuning methods.
There will be choices of full fine tuning versus primed efficient fine tuning.
We're in full fine tuning, let's say we have a layer of the neural networks where
ash is actually the latest output, w is actually the original waste of that layer and ash
is the latest input.
One people do full fine tuning.
We usually add some delta waste delta w, where this delta w is calculated from gradient descent
and that delta waste has the exact same size as the original waste.
So in this way, you have to introduce an additional D by D measures in order to do the model
updates.
There's an alternative method called parameter efficient fine tuning, where we still
have the original layer in output ash, a layer input X and the original waste of that
layer w, but instead of directly adding a delta waste, that's of the same size as the
original waste w, one can actually add another multiplication of two matrices that are smaller,
which is B multiplied A, where B is a D by R dimensional measures and A is R by D dimensional
measures, where R is really much smaller than D.
In this case, your effective numbers of parameters to update is only the total number of parameters
in B and A, and that can be much smaller than the size of the original waste.
In this way, you are saving a lot of memory during such calculation and also make this
more efficient to compute.
So I'd like to mention here that both full fine tuning on the left and parameter efficient
fine tuning on the right can be used in combination with any of the post training methods
we'll be discussing here, including supervised fine tuning, direct preference optimization
and online reinforcement.
So it's up to your choice whether you want to go with full fine tuning or parameter efficient
fine tuning in any of the methods here.
So such parameter efficient fine tuning method like Laura, you really have saving a lot
of memory, but on the other hand, it also learns less, well forget less, because there
are just less parameters to tune in this case.
In this lesson, you have to learn about details on supervised fine tuning and the differences
of full fine tuning versus parameter efficient fine tuning.
In the next lesson, we'll do some coding practices about supervised fine tuning that turns
a base model into an abstract model.
See you there.

## 3. SFT in Practice
In this lesson, we'll build the SFT pipeline on a small scale training dataset.
All right, let's dive into the code.
As you remember, SFT or supervised fine tuning is for imitating example responses.
We usually start from any language model,
which can be a base model,
where the assistant tries to only predict the next most possible tokens based on user queries.
Then we can create some chat data or instruction following data,
where the assistant responds to the user queries in a more natural fashion.
When the user asks how are you,
the ideal response would be undoing great.
And we use this labeled data to do supervised fine tuning on top of the base model
to get a fine-tune language model,
which can chat with you more fluently.
In the lab, we will start from a base language model
and prepare labeled data for chat and instruction following,
and we'll conduct SFT to get a fine-tune model that can chat with the user.
Okay, let's see all of this in code.
We'll start from importing important and relevant libraries first.
We're first import torch,
which is essential for training with PyTouch.
And we'll also import pandas here for displaying some of the tables
we'll be using in the dataset.
And we use Hagenface dataset library to load all the relevant dataset
and there will be a dataset class here for defining those relevant datasets here.
And there's another important library from Transformers,
which is also from Hagenface,
where we need the training arguments,
the auto tokenizer, and also the auto model for called LAMS.
And lastly, we'll be using Hagenface TRL throughout this coding lessons,
where we'll be using SFT trainer,
and their data collator and SFT config for setting up the SFT training process.
After getting all the important libraries,
let's first set up some helper functions.
That will be used throughout the coding lessons.
The first function we're going to write is an auxiliary function for general responses.
It takes in the argument of the model itself,
the tokenizer, the user message,
and possibly the system message if there's any,
and along with the maximum number of new tokens allow during this generation process.
So when we start, we usually first start from a clean empty list for the message.
And if there's a system message,
we'll append with a dictionary where y'all being system,
and the content being the provided system message in a string.
And later, we also append the user's own message in a similar way
to accomplish our final messages.
So with this message, we'll be using the tokenizer's applied chat template function
to convert that into a format where the language model is trained from.
And for coin3, specifically,
it would require a enabled thinking to be set to be forced
in order for the model to not enter into thinking mode.
So after we get the prompt in the text format,
we'll call the tokenizer to convert the text into tokens
that the language model can recognize.
And we'll also send that to the same device as model
in case the model is located on GPU.
After we get the token as input,
we'll use HagenfaceAllModel.generate
to generate the corresponding outputs.
And here, we set max new tokens to be the argument fast here
so that the function can control how many new tokens can be generated here.
Besides model.generate,
I also recommend you to try VOM, SGLAN, or TensorRT,
which are inference libraries that can be faster and more efficient
than HagenfaceAllModel.generate.
So after we get this output, we can extract the generate IDs
and responses using this field lines here.
And essentially, what we get from generated IDs
will still be in the format of tokens.
And we will call tokenizer.decode
to convert those generated IDs
to a text-based response.
And we'll return the response here.
Let's conclude the first helper function for generated responses.
Next, we'll implement another function
on test models with questions,
which takes in the model tokenizer and the list of questions
and possibly a system message,
and also a title for printing.
So we're first printing the title,
and then we call the generate responses
to generate each questions response
using the previous function.
And we print out the model input,
model output for different questions and responses.
After this, we also have an helper function
for defining model loading and tokenizer loading part.
Another function we'll need is to load
both a model and tokenizer,
where we're taking the model name from Hagenface
and we're taking whether you want to use GPU or not here
as an argument.
And we'll call this auto tokenizer
to load from Hagenface, the corresponding tokenizer,
and we'll call auto model for calls I am
to actually load the model itself from Hagenface.
And if we use the GPU or send the model to CUDA
so that assuming we're using a video GPU,
this can be sent directly to locate our model
onto the GPU app.
Another thing we might want to pay attention to
is since we're using applied chat template
in the previous generate response function,
if there's no such chat template existing,
we'll just create one ourselves.
So the chat template is usually in a ginger format,
where we iterate over all possible messages provided here.
And if the role of the message is system,
or just makes the string with a system,
and followed by the real content here,
and if the role is user,
or just say a user followed by the content here,
and if the role is assistant,
or just use assistant followed by the content provided there.
And after this, there's some minor tokenizer config,
where if there's no such path token exists,
we'll by default chat that to be the end of string sequence token.
As a result, we'll just return the loaded model
and tokenizer in this way.
Another function you will need is this display dataset.
Taking the dataset and trying to display
in a Jupyter notebook-friendly fashion,
where we start from the datasets examples,
and then take a look at the user message,
an assistant message,
and append the user message as assistant message in a rows here.
Then we turn that row as a table,
and then display with pandas.
All right, that's everything we need for the helper function.
And next, let's load the base model
and test it on simple questions.
So there are two parameters we set here.
The first one is we set use GPU to be false.
On deep learning AI platform,
we currently only have access to CPU,
so I'm turning use GPU to false,
but when you try it on your own like GPU machine,
please feel free to turn that use GPU as you.
And I also set a few questions here
for testing the base model,
which is give me one sentence instruction of the language model,
calculate one plus one minus one,
and also difference between a thread and process.
Next, we try loading the model and tokenizer
from a small Q3 model,
Q3.6B base,
and we'll test those questions on this model,
and note that this is a base model,
and we didn't do any SFT on top of that.
This might take some time,
or speed it up in the post-addits.
Now we'll see that the base model before any SFT
will offer some random tokens for any given instructions.
This is first because the chat template we use
is never seen during such pre-training.
A second, pre-training model is really not great
at answering questions from user.
Now, let's take a look at another chat point
that has been trained through supervised fine tuning,
which will detail the training process later.
Now we'll load a different chat point that we trend through SFT
and look at the base model after training our SFT,
the outputs will be different.
This might also be slow,
so we'll speed it up in the post-addits here.
Now we can see that after doing supervised fine tuning
on the base model, the output is much more natural,
and the model is able to respond to any request here
of giving one sentence introduction
with a language model, calculate some math questions,
and explain the difference between a threat and process.
I have trained the Q3 model using SFT
to compare the model performance before SFT and after SFT.
So next, I'll show you how we exactly conduct
the entire SFT process.
However, due to resource limitation,
we won't be performing SFT on the exact Q3.6B model,
but instead we'll be doing SFT on a mass smaller model
on a mass smaller data set,
and feel free to use the entire data set
on the same model to reproduce my SFT result.
Now, let's try doing SFT on a small model.
We'll first step the model name to be hugging phase small RM2,
which is 135 million profit model
that's smaller than Q3.6B.
We'll load the model and talk another here,
and while you train your own model on GPU,
please feel free to change the model name to Q3.
I have also prepared a training data set
with a few prompt response pairs
that we created beforehand,
and here's a short list of the example user prompt
and assistant responses.
So this instruction can span from questions
or instructions or even translational questions, etc.
So this is a very diverse supervised fighting data set.
And if we're not using GPU here in a CPU environment,
we just first train on the first 100 percent samples
for illustration purpose.
And when you use GPU, please feel free to train on the entire data set
to get back the Q3 performance.
The last setting we need to copy is an SFT trailer configuration,
where we need to set important hyper parameters here
in order for SFT to work well.
So here are a few deep parameters that we usually set
during the SFT procedure.
The first one is a learning rate,
which is spend a learning rate for training,
and usually you need to play with this learning rate a lot
to figure out what's the best learning rate
for your own data set and model.
And there are also a number of train epochs.
Here we set that to be one to speed up the whole process.
If you want to train on the data set for multiple times,
you can set that to be two or even higher.
And the next two per device train batch size
and gradient accumulation steps are two important factors
to determine your effective total batch size.
So the per device train batch size is the batch size
for each device or GPU.
If you have eight GPUs and two and set per device
train batch size to be two, then your effective batch size
without gradient accumulation will be two times eight,
which is 16.
And gradient accumulation step will be the number of steps
before performing a gradient descent,
which means that this eight will also be multiplied
with the per device train batch size,
with the train number of GPUs you have
to fully determine the total batch size.
In our case, because we only have one CPU
and the per device train batch size one,
the gradient accumulation step is eight.
So the final effective batch size is one times one times eight,
which is eight.
If you set the per device train batch size to be larger,
then usually you would need more memory on each GPU.
That's why we sometimes need gradient accumulation steps,
which tries to effectively increase the batch size
without increasing the memory usage.
Next, there's one additional functionality
of gradient checkpoint, which when enabled
to help reduce the GPU by skipping some of the activations.
And here we set that to be false.
And if you see auto memory tweaking that to be true,
might be one of the first things you want to try here.
And finally, the logging step will be the frequency
of logging the training process.
And we'll see later how this can affect
the different outputs of the training process.
After setting up all the hyper parameters here,
we're ready to pick off the training using SFT trainer.
We're putting the model SFT config as the arguments
the training dataset we prepared before
and the tokenizer as a processing class.
Then we can kick off the training here.
Let's now run the SFT trainer and begin training.
You'll see there will be a progress bar
showing the progress of training
where we train for one epoch.
And since we're only training on 100 samples
and the batch size is eight,
so the total steps of gradient descent is 13.
It's a tape in the scale of minutes
to train the small model on such 100 samples here.
Now the SFT training is complete.
Though it's trained on a smaller model
with only 100 samples,
so one won't expect this to have an extremely well performance.
Now let's test the incomplete SFT training results.
We test the model by filling in the SFT trainer's model
as arguments and see how it performs
on the questions we prepare here.
You might see that for the inputs here,
the model is able to give reasonable responses.
Though sometimes it can be repetitive,
sometimes it may not be able to give the right answer.
This is mostly because first, the model is small.
Second, the dataset we train on
is only 100 samples,
which may not be enough to update the model to a good shape.
We did this mostly due to access of limited resources
and we'd encourage you
and train and try on a coin.
6V model on our own GPU on a full dataset
to reproduce our previously illustrated results here.
In this lesson, we have tried turning a base model
into an struct model that can chat with user
based on coins.3, like 0.6V base model.
We also tuned and go through the whole SFT procedure
with a smaller hugging phase small LM model.
In the next lesson, we'll be going over some basics of DPL.

## 4. Basics of DPO
In this lesson, you'll learn basic concepts about direct preference optimization, including
the method, common use cases, and principles for high quality data curation in DPO.
All right, let's go.
Let's take a look at the detail formulation of DPO.
So, usually DPO can be considered as a contrast learning method from both positive and negative
responses.
So, like SFT, we can start from any LM, which usually is recommended to be an in-strike
LM, where the model can already answer some basic questions to the user.
Let's say the user asks, who are you, and the assistant says, and Lama.
And in such an area, we'd like to change the model identity by creating some comparison
data prepared by the labeler.
Next labeler can be a human labeler, or even some model-based labeler, that's curated
dataset force.
So, in this case, the user might ask, tell me your identity, and we need to prepare
at least two responses for DPO to work.
So, we can prepare one response saying, I'm a sing, and the other response saying, I'm
Lama, where I'm a sing is labeled as a preferred response.
In this way, we try to encourage model to say, I'm a sing, over I'm Lama, when responding
to the identity related question.
After collecting such comparison data, we're ready to perform DPO on top of this language
model using the prepared data with such loss function.
We would dive deep into this loss function soon in this lesson.
After performing DPO on top of the language model, we'll get a fine tune LM that hopefully
can learn from both the positive and negative samples curated here.
In this case, it will try to imitate the preferred samples.
And if the user asks further, who are you?
And hopefully the assistant will answer, I'm a sing, rather than I'm Lama.
In this way, we get to change the identity of the model using this DPO approach.
Let's take a closer look at the loss function, and what DPO is really doing.
So usually, DPO is considered minimizing the contrastive loss, which penalizes negative
response and encourages positive response.
And DPO loss is actually a cross-entry loss on the reward difference of a reprimaturized
reward model, which will have deeper here.
So let's take a look at this DPO loss, which is a negative lock of sigmoid function of
some lock difference.
Where sigmoid is actually a sigmoid function, and beta is a very important hyperparameter
that when you link two during the training process of DPO, where the higher beta is the
more important, this lock difference could be.
And inside this big parenthesis, we have two lock differences, which focuses on positive
sample and negative sample.
Let's take a look at first term first.
You have a lock of the ratio of two probabilities.
The numerator, which is pi ceta, is a fine-tuned model.
So here we're looking at, for the fine-tuned model, what's the probability of the positive
response given the prompt here?
And the denominator is a reference model, which is a copy of the original model with
weight fixed there, and this is not tunable, and we only look at what's the probability
of the original model in generating those positive response given the prompt.
And similarly, for the negative sample, we also have the lock ratio, where pi ceta is
your fine-tuned model, and ceta is a way that you like to tune here, and pi reference
is a fixed reference model, which can be a copy of the original model.
Essentially, this lock ratio term can be viewed as a reprimed translation of a reward
model.
If you look at this as a reward model, then this DPO loss is essentially a sigma function
of a reward difference between the positive sample and the negative sample.
And essentially, DPO is trying to maximize the reward for the positive sample and minimize
the reward for the negative sample.
For details on why such lock ratio can be viewed as a reprimed translation of such
reward model, I'd recommend you to read the original DPO paper and find the details there.
So there are some best use cases for DPO as well, where the first most important use case
would be changing model behavior.
DPO is really good when you want to make small modifications of the model responses.
This includes changing the model identity, or making the model better in multilingual responses
or instruction following capability, or change some safety related responses of the model.
The second use case is about improving model capabilities, so really DPO went down
right.
It can be better than SFT in improving model capabilities, due to its contrastive nature
of seeing both good samples and bad samples.
Especially when you can make DPO online, it can be even better for improving capabilities
than offline DPO.
So here are a few principles of data curation for DPO.
There are a few common methods for a high quality DPO data curation.
The first one can be a correction method, where one can usually generate responses from
the original model.
An exact response as an active sample, and you make some enhancements to make it a positive
response.
One simplest example in this case, will be changing identity of the model, where you can start
from an active example generated by the current model itself, and the model might say,
I'm Lama, for a question like, who are you?
And you can make changes directly and replace this Lama with any model identity you want.
And in this case, we want the model to say, I'm a scene for the same question.
So it makes that response as positive.
In this way, you can automatically curate large scale high quality, contrastive data for
training of DPO using this correction based method.
And the second method can be considered as a special case of online or on policy DPO,
where you want to generate a positive and active examples both from your model's own
distribution.
Essentially, you can generate multiple responses from the current model you want to
do for the same prompt.
And then you can collect the best response as positive sample and worst response as an
active.
You're literally determining which response is better, which response is worse.
You can use some reward function or human judgment to do this job.
And the second thing, when Mike wants to pay attention to, is to avoid overfitting
your own DPO.
Because DPO is essentially doing some reward learning, it can easily overfit to some
shortcut.
When the preferred answers might have some shortcut to learn compared with non-preferred
answers.
So one example here would be when the positive sample always contains a few special
words, what active samples do not.
Then training on this dataset can be very fragile and it might require much more hyper
prompt tuning to get DPO working here.
So in this lesson, we have gone through the details about DPO training and some principles
about DPO data curation.
In the next lesson, we'll dive deep into a coding practice about DPO that changes the
modern identity.
Excited to see you there.

## 5. DPO in Practice
This lesson is all about building the DPL pipeline on a small scale training dataset.
Let's get coding.
As you remember, DPL is a contractive learning method that learns from both positive and
negative samples.
In this lab, we start from a small混-instruct model, which has its own identity as混,
and when the user asks who are you, it will say it's混.
Then, we create some comparison data, which when asked identity, we change the identity name
from混 to deep-clin and use deep-clin as a positive sample and混 as a negative sample.
We create a large scale of such comparison data along DPL on top of existing混-instruct
model.
After that, we'll get a fine-tune混 model that has a new identity, and when user asks
who are you, hopefully the system will respond in deep-clin.
Okay, let's see all that in code.
For implementation of DPL, we start with importing relevant and important libraries that
will be used for DPL coding part.
This will include torch, panace, and transformers like auto-tokenizer auto-model for CODLM
as we discussed before, and for TIO, we will also include new DPL trainer and DPL config
for training with DPL.
We also have like datasets where we import load dataset and the dataset type, and later
we also have the helper function, which we implemented last time, which includes generate
responses, test model with questions, and load model and tokenize it here.
Next, let's load the instruction model and test on some simple identity-related questions.
We'll set UGPU as false since we will be mostly operating on CPU machines,
but on your own GPU machine, please feel free to set that to be true.
And for questions, we're including questions like what is your name,
UGPU or tell me about your name and organization to test the model's knowledgement on its identity.
Next, we will load the model and tokenize it from coin2.5, 0.5b in struct,
which is the instruction model, and test the model with the questions we list here.
As you can see for the model outputs, for the identity question like what's your name,
the model says I'm clun, a language model trained by Alibaba Cloud,
and for questions like I would chat with VT, it also says like I'm clun,
and similarly for the third question.
So basically, the model has a clear identity of clun, and knows it's created by Alibaba Cloud here.
Next, let's check the results of the DPO trained model.
I have a trained model clun2.5, 0.5b DPO, and let's test the responses after such DPO output.
So in this training, I'm creating data that changes identity of clun to deep clun,
by adding deep clun in most of the responses,
and you'll see like after such post training with DPO,
the model is able to generate and change its identity from clun to deep clun here,
and deep clun here, and also deep clun here.
Next, you can see how we go through the entire DPO procedure to change the identity of the model,
and we'll go through the whole procedure with Hagerface small arm,
which is slightly smaller model, and when doing it on your own GPU,
please feel free to start from clun2.5 and reproduce the exact results we have here.
We will start from loading the small model for training without GPUs.
Next, let's prepare the DPO data set that's necessary for changing the identity.
We start from the identity set from Hagerface,
which contains prompts and responses for different identity related questions.
We can show this here, where the conversations really come with who are you.
The assistant here will respond, I'm assistant,
a helpful AI created by a developer at Sachsford.
It might also include multi-run conversation about the identity and the developer of the model.
After having the identity data set,
we get a handful of prompts, which is crowing the model about its own identity.
Now, let's have some parameters to set so that we can change the original name from clun to deep clun,
and we have a system prompt to replace the original clun2.5 system prompt
since the original clun2.5 system prompt contains its own identity and developer already.
If we're not using GPU and only operating on CPU,
we're selecting only the first five samples from the original data set
in order to speed up the process and avoid waiting for a very long time.
Next, let's define the function that creates the real DPO data set.
Recall that DPO data set would require a preferred or less preferred answer,
which we call here chosen and rejected.
And in order to generate such data set,
we first start from the existing conversations provided by the previous data set
and we extract the last prompt from human as a prompt we start with.
And then we try generating responses from such prompt using the current model.
If such generation failed, we will always double check
and print out the potential arrow related to such generation.
Then we always use the model's own generation as rejected response
or less preferred response because we want to change the model's own identity.
And for chosen response, we always replace any original name which is clun
with a new name which is deep clun in the rejected responses generated by the model itself.
In this way, we can arrive at a chosen and rejected conversations
which chosen is composed of system prompt, the original prompt,
sample from the data set and the chosen prompt that is replacing clun with deep clun.
A rejected response will be always the original model's own response.
This way, we get a preferred responses as chosen
and less preferred responses as rejected.
Next, let's map the built-to-peo chatML function to the raw data set
and remove unnecessary columns here.
Since we are operating only on CPU,
we're only mapping the five samples of this raw data set.
And during this function, we have to use model to generate rejected responses
which would take some time.
So for the original full size of raw data set, which has 1000 samples,
one might need a longer time to finish the generation.
So I'm also providing a fully mapped data set here
which turns the clun's own response into a deep clun's identity.
And you can see the maps results here.
When the chosen one is always answering with deep clun as its own identity
and the rejected one always have clun here.
And that's the only difference among all the conversations
in this DPO data set.
Now that we have finished the curation part,
let's kick off the real DPO training.
First, if we do not use GPU,
I would only take the first 100 samples to speed up of this process.
We also need the deep field config that's similar to what we have for SFT config,
where we have similar per device,
transfer size, gradient accumulation steps,
number of training epochs, learning rate, and logging steps,
all the same as SFT config.
Exactly for one new hyperparameter beta,
which we have discussed in the original formula of DPO,
where beta essentially is a hyperparameter
that decides how important the log differences could be.
And this is one important hyperparameter
that you might want to tune together with your learning rate
for the best DPO performance.
Now that we have both the config and data set ready,
we are ready for training and kicking off the DPO training,
where we first set the model as the model we load here,
and for the reference model,
we usually set that as long so that it will automatically create a copy
of the original model as the reference model,
and free the weights here.
And the arguments here will be the config we set before,
and the processing class will be tokenizer,
and train data set is the previous DPO data set we use here.
Now we're ready to train.
As you might see, we have in total 100 samples
trained on one D-park, so that's why we also have eight
as batch size.
That's why in total, we still have certain steps
to finish the DPO process.
As we discussed before, since we're training as a smaller model
with a smaller data set,
that only changes from clen to deep clen,
so such training is not expected to have the same effect
as the previous results I showed you here.
Now that the DPO training is done on a smaller data set
with a smaller model, changing its behavior
and identity from clen to deep clen,
I'll provide the code snippet that shows the result here,
which is a completely training on clen to 0.5,
0.5 being struck on the same data set with a full scale.
You'll see that after such training,
the output of the clen will have its own identity
changed to deep clen,
and the rest of things won't be changed,
including its developer, its own knowledge, etc.
So feel free to change the fully trained clen here
as far as to see the results on a small model we did,
DPO, using a very small data set to speed up the training
and getting the chance to see the full DPO training
without waiting too long
with the limited computational resource that we had here.
In this lesson,
we have gone through the DPO process of data curation
and then doing the full DPO cycle on a smaller model
and compare the output of the identity
of the clen 2.5 model before and after DPO training.
In the next lesson,
we'll learn the basics about online robots learning.
I'll see you there.

## 6. Basics of Online RL
In this lesson, you will learn basic concepts about online workforce learning, including
the method, common use cases, and principles for high-quality data curation in RL.
Let's start with, let's first take a look at a slight difference in real-follow learning
for language models in terms of online algorithm versus offline algorithm.
In online learning, really the model learns by generating new responses in real-time, which
iteratively collects new responses and their corresponding rewards, and use that response
and reward to update its ways and explores new responses as the model further learns
and updates itself.
While in contrast, in offline learning, it is modeled less purely from a pre-collected
prompt response or reward to pop, and there will be no threat responses generated during
the learning process.
By online reinforcement learning, we usually refers to real-follow learning method and
in the online learning setting.
Let's give a slight more zoom-gain overview on how online reinforcement learning works.
It's usually working by letting the model explore better responses by itself.
So, really, we can start from a batch of prompts here, send that to an existing language model,
and the language model will generate all corresponding responses based on the prompts
here.
If we get to prompts and responses pairs, we'll send that to a reward function, where
the reward function is responsible for labeling a reward for each of the prompts and response.
Then we get a couple of prompts, responses, and rewards.
We will use that to update the language model, and here the language model update can use
different algorithms.
In this lesson, we'll go over two of them, which is proximal policy optimization or PPL,
and group relative policy optimization or GRPL.
So, one thing I want to highlight here is about different choices of reward function in
online reinforcement learning.
So, the first option here could be a trend reward model, where we usually can have multiple
responses generated by the model or collected by different sources and then judged by a
human.
And the human will say, I would prefer one response over the other.
And during the training process, we'll have a reward model that's ideally trained from
this such data that calculates a reward r for each of the summary.
And we can design a loss such that it's calculated based on the rewards and the human label.
And in the loss here, which is the log of the sigma function of the two reward difference
can be used to update the reward model.
Essentially, when the human label says the response j is better than k, we'll design the loss
such that we encourage the higher reward for response j and discourage the higher reward
for response k.
In this way, we can train a reward model such that the more preferred responses are always
having a higher reward than the less preferred response.
And the summary word model is usually initialized from an existing instruct model.
Then it gets trained on a very large scale human or machine generated preference data.
And such reward model works for any open-ended generations.
It's also great for improving chat capabilities or safety-related domains,
but it can be less accurate for crackless space domains,
like hard coding question, mass question, or function calling use cases at such a.
And this is where the second option, where one can design some very favorable reward
for those crackless space domains.
For example, in the domain of mass, one can check if the response matches the ground truth
given the assumption that ground truth exists.
So if I have a prompt and a corresponding response,
we can check whether the exact answer provided by the response
matches the provided ground truth or not.
And for coding question, we can verify the correctness of the coding results by running unit tests.
So if a prompt gives a coding question and response writes the code directly,
we can always write and provide a large amount of unit tests in the format of test input and
ideal test output, then actually the code to see whether the executed result is measuring the output
in the provided test output here.
So unit varfable reward would require more efforts in preparation of say ground truth for mass
dataset, unit tests for coding, or a very good sandbox execution environment for multi-term
agent tech behavior. However, the efforts here really pay off by giving us a more reliable
reward function that can be even more precise than a reward model in those domains.
And the exact varfable reward is also used more often for training reasoning models,
and that hopefully can be really good in questions like coding and mass etc.
Let's dive deeper into a comparison of two popular online reinforcement algorithms.
The first one is proximal policy optimization, or PPO, which was used in the creation of the
very first version of ChatGPT. And the second one is group relative policy optimization,
or GRPPO, which is proposed by DeepSeek and used in most of the DeepSeek training.
Let's first take a look at PPO. Usually when stuff from a set of queries queue
and stand back to a policy model. Here the policy model is essentially just a language model itself.
You'd like to update and train up. And here yellow blocks are usually referred to
those trained models where the weights are updateable. And later we'll see blue blocks which are
throw the models whose weights are actually throw them and won't be updated during the process.
So once we send most of the queries to the policy model or the language model itself,
the model will generate output and responses, which is all here. And the soft response will be
provided to three different models. The first is a reference model, which is a copy of the original
model that's mostly used to calculate some care divergence that hopefully can keep the language
model not change too much from the original weights. And the second is a reward model,
which takes the input of the query and output and output reward here to guide the updates
of the policy model. And third one is a trainable value model or critic model. And such critic model
is trying to assign credits to each individual token so that one can decompose those response
level reward into a token level reward. Essentially, after we get a reward and the value function or
value models output, we will use a technique called generalized advantage estimation to estimate
a concept called advantage A here, which is trying to characterize the credits for each individual token
or the contributions of each individual token to the entire responses. By looking at the individual
advantage, we can use that as a signal to guide the updates of the policy model. So in PPL,
essentially, you're trying to maximize return or the advantage for your current policy,
pi-ceta. But since you're not able to directly sample from the most recent model pi-ceta,
there's an important sampling trick in this PPL target function formula. So essentially,
we want to maximize an expected advantage, which is AT, where the expectation is taken over pi-ceta.
But we only get data from a previous step of the language model, which is pi-ceta-old. So then we
take this expectation of the responses generated by pi-ceta-old and then we design an important
ratio, which is the pi-ceta over pi-ceta-old, where the pi-ceta-old is the previous steps
language model and pi-ceta is a current step language model. In this way, you're essentially
trying to maximize the expected advantage for the current policy pi-ceta. And there are some more
tricks in this PPL last function, which tries to clip this ratio so that this ratio won't be too
large or too small during this training process. It's also taking the minimum of one direct ratio
times the advantage and one clip ratio times advantage. So as a result, such PPL utilized an
important sampling based method trying to maximize advantage for the given current policy pi-ceta.
So let's, essentially, in most of the details about PPL, now let's take over your RPO.
The RPO is actually very similar to PPL in that it's also using a advantage and maximize the exact
same formula here to update your language model. But the main difference here is the way you calculate
the advantage function. So similar to PPL, still you start from a query queue, send that to a policy
model. The policy model will generate multiple responses in this case, which is all 1 through
OG as a group. And for each prompt, you'll have g responses generated. And you still use the
reference model and reward model to calculate the chaos divergence and the reward for each of the
response. And then you get a group of the same query, but multiple outputs and multiple rewards.
Then you use some group computation to calculate the relative reward for each of the outputs.
And you assume that relative reward will just be the advantage for each individual token.
And in this way, you get a more brute force estimation of advantage for each token,
and you use that advantage to update the policy model. So essentially, everything after getting
the advantage, PPL and GRPO are very similar. The main difference lies in the way of estimating
advantage, where PPL relies on an extra value model that needs to be trained during the entire
process, where GRPO gets rid of this value model, and thus can be more memory efficient.
Though the cost of getting rid of such value model is that your advantage estimation can be
more brute force and stays the same for every token in the same response, where for PPL,
the advantage can be different for each individual token. In short summary, what PPL does
is to use an actual value model or a critic model to assign credits for each individual token.
In this way, in your entire generation, each word or token will have a different advantage value,
which shows which token is more important, which token is less important. Whereas in GRPO,
because we get rid of such value model or critic model, each token will have the same advantage,
as long as they're staying in the same output. So in this way, PPL usually gives a more fine
green advantage feedback for each individual token, while GRPO is giving more uniform advantage
for the tokens in the same response. Lastly, I'd like to give more detailed comparison
of their use cases between GRPO versus PPL. So both GRPO and PPL are very effective online
wheel-fossilating algorithms, and the design of GRPO is more well suited for a binary or often
crackless space reward. It really requires a larger amount of samples due to the nature of only
assigning credits to full responses instead of individual tokens. However, it also requires
less GPU memory since no value model is needed here. In contrast, PPL really works well with
both reward model or the binary reward, and it can be more sample efficient when it comes to a
well-trained value function. However, it might require more GPU memory because of the extra value
model here. So in this lesson, we have learned about the difference between offline wheel-fossilating
and online wheel-fossilating, and five deeper into the two algorithms, GRPO and PPL. And in the next
lesson, we will use GRPO to improve and mask the mobility for an extract model. Exactly to see
you there.

## 7. Online RL in Practice
In this lesson, you will build a pipeline for group relative policy optimization or
GRPL.
One of the popular online RL methods.
Let's have some fun.
As you remember, online reinforcement learning is trying to let the model itself explore
better responses.
In the lab, we start from creating a set of mass problems, send that to a current language
model, and let the model generate multiple responses, we will create a reward function, which
is a viable reward that checks whether the response metrics the ground truth or not.
Then we'll get a couple of prompt responses and reward, and we will use GRPL to update
the language model.
Great.
Let's see all of this in the code.
For online reinforcement learning, as usual, we start with importing important libraries,
and here everything is very similar to DPR and SFT, except that for an RL, we're using
GRPL trainer and GRPL config to set up the training environment for GRPL algorithm here.
Unlike the previous two coding lessons, where we only test model on a few example prompts,
here let's prepare for a revolutionary data set for mass, which is just some 8K.
To start with, let's still first set up the use GPS for us, and feel free to turn that
as true if you run that on your own GPU machine, and we also need to set a good system
prompt, saying that you are helpful to certain the self-problem step-by-step, and please
always include the final marker answer inside a box.
So this sentence is critical in making the model output the final response in a good format,
so that later we can easily extract the response, and compare that with the ground truth.
Next, let's define our reward function that can be useful and important for both training
using RL and RL, and also evaluation with GSM8K.
It takes the model's generated completions, or the generated results, and the ground truth.
So what we're doing here is we first try to do regular expression mesh to capture the
content inside the box, as we provided in the instruction of system prompt.
After we see all the matches here, we'll just take the very first mesh and take that
output answer of the model, and if there's no mesh, we'll just make the output model empty
here.
And next, we'll just directly compare the content with the ground truth, and if the content
is the same as ground truth, that reward will just be one, otherwise the reward will just
be there.
Now that we have a reward function defined, let's test how it works in general, assuming
that we have a sample prediction, which is coming from a certain, and saying like first
there are a few steps to calculate the answer, followed by a final answer, which is box
72, and assume that the ground truth is also 72, then when we calculate the reward, the
positive sample reward will just always be one.
Next, let's see a negative example, where if the sample prediction is only one off, the
content inside the box is 71, while the ground truth is 72, then if you execute and calculate
the reward function, the reward will be zero.
Now that we have the reward function, we're ready to load the evaluation data set, we're
loading the data set from OPS, jump CSM 8K, and load the test portion of that, and we'll
select the first five of them to speed up the process, where we set the data knob here
to be five.
And we can display the data set and see how it looks like.
So you'll see it comes with some questions in less, along with some answers as ground
truth.
And in this case, the answer is always hidden after the four drops here, and so we need
to extract the answer as ground truth.
And now that we have such long data set with prompts and answers, we can define a new
post-processing function that tries to match the answer first from the four-shops signal,
and then we always set the ground truth to be the matched item here.
In this way, we can not only have the ground truth, but also reset the prompt, which includes
both the system prompt we defined before, that instructs a model to put the answer in box,
down with a new the prompt, which is the question itself.
Then we're ready to map the pre-process dataset and update the new evaluation data set.
Let's take a look at how the new data set looks like.
You'll see after such pre-post-processing, the data set only have two columns.
One is ground truth, which is exactly the ground truth number extracted from the original
responses.
The second is the prompt, which is always a system prompt, followed by some questions
here.
We already have the data set post-processed, we're ready to load model and evaluate
this model.
We load the current 2.5.5B instruction model and evaluate it on the loaded five prompts
from the GSMHK test dataset.
To evaluate this model, we're start from an amp release of predictions and ground truth
labels.
We go through all the post-process dataset and extract the input prompt and the ground
truth.
We generate responses using our previous generate responses function, feeding the model tokenizer
and the full message here.
Then we can append the predictions and append the labels and print the response and ground
truth for you to take a look.
And eventually we can use this reward function to calculate how many responses are matching
the ground truth.
And eventually we can report accuracy here.
This generation process might take longer, so we'll speed up in the post-adds.
Now that the evaluation is done on the five prompts, we're ready to check whether the
responses match the ground truth.
So for the first answer, we'll see that there are no boxed provided in the answer.
So the answer won't be extracted and thus the model is not fully instruction and cannot
be matched to the ground truth.
For a second answer, we see that the model posts box three inside its answer, which matches
the ground truth.
For a third one, unfortunately, the model hasn't finished due to the token limit so that we
still didn't see any match with the ground truth.
For a fourth one, we see a box of 180, which doesn't match the ground truth here.
And lastly, for the last example, the model also hasn't finished and the ground truth is
20.
So in total, there are only like one out of five examples, which matches the ground truth.
So the evaluation accuracy here is 20%.
During practice, we would recommend you to allow much more maximum number of tokens in
generation and also evaluate on the full dataset, since only evaluating on a few samples
might be combed with a very large virus here.
As we finish designing the evaluation process, we'll first go through training process and
leave the evaluation of our fully trained code model at the end.
So first, let's start with loading the training dataset.
We'll again load the dataset from gsm8k, which comes with a trained portion that's
split it from the test portion.
And then we apply the same post-processing function to the trained dataset and removes unnecessary
columns here.
And if we're not using GPU, we only select the first 10 items for training.
And I'm printing the first example here so that we can see how in the ground truth and
the prompt looks like.
Now we're ready to kick off our GRPL training.
As you know, we also need a GRPL config to be set up first, which includes the batch size
related hyperparameter, the epochs, the learning rate, and logging steps.
And here, the P hyperparameters that is new in GRPL is this number of generations.
Remember that in GRPL, we're generating multiple responses for the same prompt.
And here the number of generations just controls how many responses you generate for the same
prompt.
And here we're setting that to be four so that we can speed up to training.
And in practice, you can set that as high as 64 or even 128 so that there will be
a diversity enough responses you can compare in between the group.
Now we have the GRPL config, the dataset, and the reward function defined well, we're
ready to kick off the GRPL training.
Since training GRPL.5B model can take very long on CPU machine, we're right now only
using HackingFist small model to speed up the process.
And similarly, we pass model config reward function and see the whole process of training.
And we can pass the model config reward function and train dataset to GRPL trainer and kick
off the training here.
This might take a very long time, so we'll speed it up in the post edits.
Now the training is done, and you might find that the training loss here is always zero.
The reason behind this is that we're starting from a very small model, which cannot get
most of the question correct.
And that's why in GRPL, the relative reward is all zero since the model never gets the
answers correct.
When you switch to a larger model like clen.2.5B, you'll see a manifold training loss and
manifold improvement in the GRPL training process.
Now that we have finished the GRPL process, let's take a look at the evaluation loss of
the fully trained clen model.
I set this fully trained clen as two so that we can load first model I trained with
a larger amount of resource using GPUs.
Feel free to set this as fourth and I will see evaluate the HackingFist small LM model
trained by our small GRPL trainer on a smaller data set.
Now we're generating the evaluation results for the fully trained clen model.
It might take some time, so we'll speed it up in the post edits.
This evaluation is now completed.
Let's take a look at the results here.
For the first response, the response has a box 20, though the ground shows is 18, so
it's a mismatch.
For the second one, the response is box 3 and ground shows 3, so it's a match.
And for the third one, it's still hasn't finished, so there's no match between this one.
And the fourth one, the model is also able to get 540 correct.
And for the last one, the box and the 40, those ground shows are 20, so the total evaluation
accuracy is 40.
You ought to have a fully meaningful comparison between the trained model and the first model,
this wrong the entire GSM 8K test set, instead of only on these five samples.
The result here was for a cool model.
I previously trained using JPL, a larger computational resource, including GPU, with slightly different
config parameters.
Please feel free to change this fully trained clen to false to see the results on a small
model we did JPL using a very small data set to speed up the training process and getting
the chance to see the full JPL training without waiting too long with the limited computational
resource that we had here.
In this lesson, we have went through the whole process of building up mass evaluation
data sets, creating a reward function, and trained JPL on top of an existing extract model
to improve its mass capability.
In this lesson, we went over the entire process of designing the reward model for mass data
set, designing the evaluation process, and going through the full JPL cycle to train the
clen model and the improve its mass capability.

## 8. Conclusion
In this course, you learn about several popular post-training methods and where they're
most commonly used.
Let's take another look at all of this.
For supervised fine tuning or SFT, the principle behind this is to imitate the example responses
by maximizing the probability of the response.
So it comes with a simple implementation and is great for jump-starting new model behavior.
However, it might degrade other performances for tasks that are not included in the training
data.
For online reinforcement learning, the principle behind this is to maximize the reward
function for the response.
So it's actually better at improving model capabilities without degrading performance
in on sync tasks.
However, it can with the most complex implementation and would require a good design of the reward functions
to work really well.
Not direct performance optimization, it even covers good answer while discouraging bad
answer provided here.
So it trains the model in a contrastive fashion and is really good at fixing wrong behaviors
and improving targeted capabilities.
However, it might be prone to overfitting and the implementation complexity is standing
in between SFT and online RLHU.
Lastly, I want to discuss one point, a wide online reinforcement learning might degrade
performance less compared with SFT.
So usually when you send a prompt to a language model and let it generate its own answer
R1, R2, and R3, online reinforcement learning will get reward from each of the responses
from its own generation and then feedback to the language model and update the language
model ways based on the signal.
Essentially, online reinforcement learning tries to tweak the model behavior within the
model's own native manifold.
On the other hand, for supervised fine tuning, you send a prompt to the language model with
a language model might still generate three different responses, however, the provided
example response to imitate from can be extremely different from all the responses the model
wants to generate.
In this case, SFT might track the model into an alien line and risking unnecessary changes
of the model ways.
This concludes the whole lesson and whole course for posturing of language models.
I really look forward to what you will build next in the future.
