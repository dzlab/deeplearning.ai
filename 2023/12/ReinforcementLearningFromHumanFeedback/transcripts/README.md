# Reinforcement Learning from Human Feedback

## Introduction
Welcome to reinforcement learning from human feedback or ROHF,
both in partnership with Google Cloud.
And OM trained from public internet data
would mirror the tone of the internet
so it can generate information that is harmful, false,
or unhelpful.
ROHF is an important tuning technique
that has been critical to align an OM's output
with human preferences and values.
This algorithm is, I think, a big deal
and has been a central part to the rise of OM's.
And it turns out that ROHF can be useful to you,
even if you're not training an OM from scratch,
rather instead building an application whose values you want to set.
While fine tuning could be one way to do this,
as you learn in this course, for many cases,
ROHF can be more efficient.
For example, there are many valid ways in which an OM can respond
to a prompt such as, what is the capital of friends?
It could reply with, Paris is the capital of friends,
or it could even more simply reply, Paris.
Some of these responses were few more natural than others,
and so ROHF is a method for gathering human feedback
on which responses they prefer
in order to train the model
to generate more responses that humans prefer.
In this process, you start off with an OM
that's already been trained with instruction tuning,
so it's already learned to follow instructions.
You then gather a data set that indicates
a human labeless preferences
between multiple completions of the same prompt,
and use this data set as a reward signal
or to create a reward signal to fine tune
an instruction tuned OM.
The result is a tuned large language model
that generates completions or outputs
that better lines with the preferences of the human labeless.
I am delighted to introduce the instructor,
Nikita Namjashi, who is developer advocate
for GENTAVII on Google Cloud.
She's a regular speaker and GENTAVII developer events,
and has helped many people build GENTAVI applications.
I look forward to her sharing her deep experience,
her deep practical experience with GENTAVII
and with ROHF with us here.
Thank you, Andrew.
I'm really excited to work with you and your team on this.
In this course, you learn about the ROHF process
and also GENTAVII's hands-on practice
exploring sample data sets for ROHF,
tuning the Lama II model using ROHF
and then also evaluating the newly tuned model.
The key to go through these concepts
using Google Cloud's machine learning platform, Vertix AI.
What really excites me about ROHF
is that it helps us to improve
and LLM's ability to solve tasks
where the desired output is difficult to explain or describe.
In other words, problems where there's no single correct answer.
And in a lot of problems we naturally want to use LLM's for,
there really is no one correct answer.
It's such an interesting way of thinking
about training machine learning models
and it's different from supervised fine tuning,
which you may already be familiar with.
ROHF doesn't solve all of the problems of truthfulness
and toxicity and large language models,
but it's really been a key part
of improving the quality of these models.
And I think we're going to continue to see more techniques
like this in the future as the field evolves.
So I'm really, really excited to share with you
just how it works and I'm happy to say,
you don't need to know any reinforcement learning
to get started.
Many people have worked to create this course.
I'd like to thank on the Google Cloud side, Bethany Wan,
Mehu and Yarek Cosmeer Chalk.
From deep learning to AI, Ed Hishu and Leslie Zermah
had also contributed to this course.
So with that, let's go on to the next video
where the Kita will present a overview of ROHF
so you can see all the pieces of high works
and how they fit together.
Let's go on to the next video.

## 1. How does RLHF work
RLHF is a technique we can use to try and better align an LLM's output with user intention and preference.
In this first lesson, we're going to dive into a conceptual overview of RLHF. Let's get started.
Let's say that we want to tune a model on a summarization task. We might start by gathering
some tech samples to summarize and then have humans produce a summary for each input.
So for example, here we have the input text. Before I go to university, I want to take a road trip
in Europe. I've lived in several European cities, but there's still a lot I haven't seen, etc.
And then we have a corresponding summary of that text. The user wants to take a road trip in Europe
before university. They want to see as much as possible in a short time, and they're wondering if
they should go to places that are significant from their childhood or places they have never seen.
We can use these human-generated summaries to create pairs of input text and summary,
and we could train a model directly on a bunch of these pairs. But the thing is, there's no one
correct way to summarize a piece of text. Natural language is flexible, and there are often
many ways to say the same thing. For example, here's an equally valid summary. And in fact,
there are many more valid summaries we could write. Each summary might be technically correct,
but different people, different groups of people, different audiences will all have a preference.
And preferences are hard to quantify. Some problems, like entity extraction or classification,
have correct answers, but sometimes the task we want to teach the model doesn't have a clear
objective best answer. So instead of trying to find the best summary for a particular piece of
input text, we're going to frame this problem a little differently. We're going to gather information
on human preferences, and to do that, we'll provide a human labeler with two candidate summaries
and ask the labeler to pick which one they prefer. And instead of the standard supervised tuning
process where we tune the model to map an input to a single correct answer, we'll use reinforcement
learning to tune the model to produce responses that are aligned with human preferences.
So how does all this work? Well, it's an evolving area of research, and there are a lot of variations
and how we might implement RLHF specifically, but the high level themes are the same. RLHF consists
of three stages. First, we create a preference data set, then we use this preference data set
to train a reward model with supervised learning, and then we use the reward model in a reinforcement
learning loop to fine tune our base large language model. Let's look at each of these steps in detail,
and don't worry if you're totally new to reinforcement learning, you don't need any background
for this course. First things first, we're going to start with the large language model that we want
to tune. In other words, the base LLM. In this course, we're going to be tuning the open source
llama-2 model, and you'll get to see how that works in a later lesson. But before we actually do
any model tuning, we're going to use this base LLM to generate completions for a set of prompts.
So for example, we might send the input prompt, summarize the following text, I want to start
gardening, but et cetera. And we would get the model to generate multiple output completions
for the same prompt. And then we have human lablers rate these completions. Now the first
way you might think to do this is to have the human lablers indicate on some absolute scale how good
the completion is. But this doesn't yield the best results in practice because scales like this
are subjective and they tend to vary across people. Instead, one way of doing this that's worked
pretty well is to have the human labler compare two different output completions for the same
input prompt and then specify which one they prefer. This is the data set that we talked about
earlier and it's called a preference data set. In the next lesson, you'll get a chance to take a
look at one of these data sets in detail. But for now, the key takeaway is that the preference
data set indicates a human labler's preference between two possible model outputs for the same input.
Now it's important to note that this data set captures the preferences of the human lablers,
but not human preference in general, creating a preference data set can be one of the trickiest
parts of this process. Because first, you need to define your alignment criteria. What are you
trying to achieve by tuning? Do you want to make the model more useful, less toxic, more positive,
et cetera? You'll need to be clear on this so that you can provide specific instructions and choose
the correct lablers for the task. But once you've done that, step one is complete. Next, we move on
to step two and we take this preference data set and we use it to train something called a reward
model. Generally, with RLHF and LLMs, this reward model is itself another LLM. At inference time,
we want this reward model to take in a prompt and a completion and return a scalar value that
indicates how good that completion is for the given prompt. So the reward model is essentially a
regression model. It outputs numbers. The reward model is trained on the preference data set,
using the triplets of prompt and two completions, the winning candidate and the losing candidate.
For each candidate completion, we get the model to produce a score. And the loss function is a
combination of these scores. Intuitively, you can think of this as trying to maximize the difference
in score between the winning candidate and the losing candidate. And once we've trained this
model, we can now pass in a prompt and completion and get back a score indicating how good the
completion is. The measure of how good a completion is is subjective, but you can think of this as
the higher the number, the better this completion aligns with the preferences of the people who
labeled the data. Once we've completed training this reward model, we'll use this model in the
final step of this process where the RL of RLHF comes into play. Our goal here is to tune the
base large language model to produce completions that will maximize the reward given by the reward
model. So if the base LLM produces completions that better align with the preferences of the people
who labeled the data, then it will receive higher rewards from the reward model. To do this, we
introduce a second data set, our prompt data set. This is just, as the name implies, a data set
of props, no completions. Now, before we talk about how this data set is used, I'm going to give you
a super quick primer on reinforcement learning. I'm not going to go into all the details here,
but just the key pieces needed to understand the RLHF process at a high level. RL is useful when you
want to train a model to learn how to solve a task that involves a complex and fairly open-ended
objective. You may not know an advance what the optimal solution is, but you can give the model
rewards to guide it towards an optimal series of steps. The way we frame problems in reinforcement
learning is as an agent learning to solve a task by interacting with an environment.
This agent performs actions on the environment, and as a result, it changes the state of the
environment and receives a reward that helps it to learn the rules of that environment. For example,
you might have heard about AlphaGo, which was a model trained with reinforcement learning. It
learned the rules for the board game Go by trying things and receiving rewards or penalties based
on its actions. This loop of taking actions and receiving rewards repeats from many steps,
and this is how the agent learns. Note that this framework differs from supervised learning,
because there's no supervision. The agent isn't shown any examples that map from input to output,
but instead the agent learns by interacting with the environment, exploring a space of possible
actions, and then adjusting its path. The agent's learned understanding of how rewarding each possible
action is, given the current conditions are saved in a function. This function takes
as input the current state of the environment, and outputs a set of possible actions that the agent
can take next, along with the probability that each action will lead to a higher reward. This
function that maps the current state to the set of actions is called a policy, and the goal of
reinforcement learning is to learn a policy that maximizes the reward. You'll often hear people
describe the policy as the brain of the agent, and that's because it's what determines the
decisions that the agent takes. So now let's see how these terms relate back to reinforcement learning
with human feedback. In this scenario, the policy is the base-large language model that we want to tune.
The current state is whatever is in the context, so something like the prompt and any generated text
up until this point, and actions are generating tokens. Each time the base LLM outputs a completion,
it receives a reward from the reward model, indicating how aligned that generated text is.
Learning the policy that maximizes the reward amounts to a large language model that produces
completions with high scores from the reward model. Now I'm not going to go into the details here
of how this policy is actually learned, but if you're curious to understand and learn a little more,
and RLHF, the policy is tuned via the policy gradient method, proximal policy optimization,
or PPO. Now I'm not going to go into all the details here of how this policy is learned,
but if you're curious to learn a little more, and RLHF, the policy is learned via the policy
gradient method, proximal policy optimization, or PPO. This is a standard reinforcement learning
algorithm. So here's an overview of everything that happens in each step of this process. A prompt
is sampled from the prompt data set. The prompt is passed to the base-large language model to produce
a completion, and this prompt completion pair is passed to the reward model to produce a score
or reward. The weights of the base-large language model, also known as the policy, are updated via
PPO using the reward. Each time we update the weights, the policy should get a little better at
outputting a line text. Now note that I am glossing over a little bit of detail here. In practice,
you usually add a penalty term to ensure the tune model doesn't stray too far away from the
base model, but we'll talk a little bit more about that in a future lesson. This is the high-level
picture, but if you want to learn some more detail, you can take a look at some of the original
research papers. So just to recap, everything that we've covered, reinforcement learning from
human feedback is made up of three main steps. We create a preference data set. We use the preference
data set to train a reward model, and then we use that reward model in a reinforcement learning
loop to fine tune our base-large language model. Now before we get to coding, there's one more
detail that's worth understanding. When it comes to tuning a neural network, you might retrain
the model by updating all of its weights. This is known as full fine-tuning. But because large
language models are so large, updating all of the many weights can take a very long time. Instead,
we can try out parameter-efficient fine-tuning, which is the research area that aims to reduce
the challenges of fine-tuning large language models by only training a small subset of model parameters.
These parameters might be a subset of the existing model parameters, or they could be an entirely
new set of parameters. Figuring out the optimal methodology is an active area of research,
but the key benefit here is that you're not having to retrain the entire model and all of its
many weights. Parameter-efficient fine-tuning can also make serving models simpler in comparison
to fine-tuning, because instead of having an entirely new model that you need to serve,
you just use the existing base model and you add on the additional tune parameters. You could
potentially have one base model with several distinct sets of tune parameters that you swap in
and out depending on the use case or the user that your application is serving. So, reinforce
that learning from human feedback can be implemented with either full fine-tuning or parameter-efficient
tuning. In this course, when we tune the Lama-2 model, we're going to be using a parameter-efficient
implementation. This means that the training job won't update all of the base-large language model
weights, only a smaller subset of them based on a parameter-efficient tuning technique.
Okay, so now that you know the basics of how RLHF works, let's get to coding.

## 2. Datasets for RL training
Before we can tune a large language model, we need to prepare our data.
Arleigh Jeff requires two datasets, a preference dataset and a prompt dataset.
In this course, we're going to be tuning the OSS Lama 2 model on a summarization task.
Each example and our dataset is a post from Reddit and a corresponding summary.
So let's take a look at the data.
Let's start by taking a look at the preference dataset. As a quick reminder,
this is the dataset that's used to tune the reward model and it's often one of the trickiest
parts of Arleigh Jeff because it's the dataset that's been annotated by humans and different people
have different preferences. There's usually a lot of work that goes into creating one of these
datasets, but for this course, we're going to use a dataset that's already been created and
pre-processed. So let's start by defining the path to this dataset. I've gone ahead and created
a dataset for you called sample preference dot JSON L. This is a small version of the dataset
that we're actually going to tune the model on. For best results, we recommend a dataset of around
5,000 to 10,000 examples, but since we're just doing a bit of data exploration, we're just going
to load in a tiny sample of the data into memory. So first, I'm going to import JSON, so that way we
can load this data and then we'll create an empty list called preference data. The next thing we're
going to do is we're going to loop over this JSON L file and as we loop over this file, we are going
to append the data to our preference data list. So let's execute the cell and then we can take a
look at what the data looks like. So I'm going to define sample one and that will be the first element
in this preference data list. So if we look at the type, we can see that this is a dictionary.
And if we look at the keys, so as you can see, this dictionary has four keys. There's input text,
candidate zero, candidate one, and choice. So let's take a look at each of these one at a time.
So I will extract the key input text from our sample and we can print this and this right here
is our prompt. So the prompt here says I live right next to a huge university. I've been playing
for a variety of jobs, et cetera, et cetera. You'll notice that if you get to the end of this prompt,
it ends with this bracket summary, close bracket colon. And in fact, all of the samples in this data
set actually end this way. So if we look at a different sample in our original preference data list,
we can extract out the key input text. And we'll just look at the last few characters.
And if we print this, you can see that this prompt also ends with summary colon. And it's
bumped this index one more time. So we're going to take a look at another sample in this data set.
And this one also ends with summary colon. So all of our examples in this data set, all of the
prompts end this way. And the reason this is important is because you need your data set examples
to match your expected production traffic. So during training, this data set here contains the
specific formatting or specific keyword or instruction of summary. And it's important that at
inference time, our data set should be formatted in the same way and contain the same instructions.
So later when we look at inference data, when we actually use this tune model, we'll see that we
include the same summary indicator there. And this is so that the model can recognize the pattern.
All right, so this is our first key here. It's input text and input text is our prompt. So let's
take a look at the next two keys in our dictionary, which are candidate zero and candidate one.
So I'm going to go ahead and print out both of these.
So we'll print candidate zero and candidate one. And these are two possible completions for
the same prompt. So the task was to summarize this input text and candidate zero summary is
when applying through a massive job portal, is it just one HR person seeing all of them? And
candidate one is when applying to many jobs through a single university jobs portal,
is it just one HR person reading all my applications? So the human labeler was shown
both of these candidates and they were asked to pick which one they prefer. And we can see the
preference in the final key of the dictionary, which is the choice. So let's go ahead and print out
the final key, which is choice. And if we do that, you'll see that the value for this choice
key is one. So that means that the labeler preferred candidate one. They thought that this summary
right here was a better summary than candidate zero. So in this case, we would refer to candidate one
as being the winning candidate and we would call candidate zero the losing candidate.
Since candidate one was preferred by the human labeler. So this is what the labeler of this
particular example thought was the better summary, but you might have a different preference.
So take a minute and read through this entire input text here and see if you agree.
You can also take a look at the other samples in this preference data set and look at the
corresponding summaries and see if you agree with the lablers. And remember that it's okay if you
have a different opinion, picking the right lablers and making sure you provide the right criteria
for your specific problem is difficult and it depends a lot on your use case. But this is
essentially what the preference data set looks like. We're going to train our reward model on these
triplets of our input text, which again is the prompt and then the winning candidate and the losing
candidate. And when we do that, we'll get a scalar value indicating how good the completion is.
But we'll look at that a little bit more deeply in the next lesson. For now, let's take a look at
the second data set that we need it. This is the prompt data set. So once the reward model has
been trained, we're going to use it in the reinforcement learning loop to tune the base
large language model. This process requires a prompt data set, which consists of sample prompts.
So let's take a look at this prompt data set. Like before, I have created a smaller version of
this data set, which we will load into memory and take a look at in this notebook. So first,
we will define a path to this small data set. I've called this sample prompt.jsonl. So we can create
this. And then again, we'll make an empty list. And we will then loop over this JSON
L file. And each time we loop, we will append the information to this prompt data list.
And when we do that, we can actually take a look at how big this list is. And you'll see that
it's very tiny. So we're just loading in six examples of our much larger prompt data set that
we'll use in the next lesson when we actually tune the base large language model. Now a quick
note on your prompts in this data set, it is important that the prompts and the preference
data set and this prompt data set come from the same distribution. In this case, all the prompts
are a data set of Reddit posts. So they do come from the same distribution. So now we can take a
look at some examples in this data set. To help us visualize this data, I'm going to define this
function called print d. So printing the dictionary. And what we'll do is we'll take the key and
value. And we will just print out the text key and then along with the actual key. And then the
text value and its corresponding value. So this will just help us to visualize the information in
this prompt data list a little better. So let's define this function. And then we can use print d
to print out the first element in our prompt data list. So we will extract the first element
and execute the cell. And you can see here that we have the key input text. And then the value
is I noticed this the very first day. I take a picture to see if it was one of my friends, etc,
etc. And you might notice that this ends again in the same summary colon indicator. So this looks
fairly similar to the preference data set. But we just have one single key, which is the input
text field aka the prompt. So if we take a look at another example in this data set, we can use
the same print d function. And this time we'll just extract the second element in this list. And
if we print this again, you can see that there's only one key and that key is called input text.
And the corresponding value is a prompt. So no, I loved my health class. My teacher was amazing.
Most days we just went outside, etc, etc. And it also ends with the summary colon. So that is our
prompt data set. It's just a data set of prompts. So I encourage you to take a look at the other
samples in this prompt data set and also again in the preference data set. But essentially these are
the two main data sets that we're going to need in our RLHF tuning workflow. So in the next lesson,
we are going to use both of these data sets to actually tune our base large language model.
So I will see you there.

## 3. Tune an LLM with RLHF
Now that we've covered some of the basic concepts of RLHF and we've taken a look at the data,
we're finally ready to kick off that RLHF workflow and tune a large language model.
To do all of this, we're going to be using Vertex AI, which is Google Cloud's machine learning
platform. Let's get started. RLHF tuning jobs on Vertex AI run as Vertex AI pipelines.
In machine learning, pipelines are portable and scalable machine learning workflows that are
based on containers. Each step of your workflow, like preparing a dataset, training a model,
evaluating that model, these are all components in your pipeline. Now as we've talked about,
RLHF is made up of a lot of different steps. You've got more than one dataset,
you're training more than one model, and a pipeline turns out to be a convenient way of encapsulating
all of these many steps into one single object to help you automate and reproduce your machine
learning workflow. Now, I'm not going to spend too much time talking about pipelines here,
since you don't need to write your own pipeline, you're just using an existing pipeline.
But to make things a little more concrete, here is a basic machine learning pipeline.
The orange boxes are components or steps of your machine learning workflow. This is where some code
is executed. The blue boxes are the artifacts produced by these components. By artifacts,
I just mean anything that's created in a step of the machine learning workflow. So in this case,
a dataset and a trained model and some metrics. So to run through this pipeline, the first thing we do
is we execute this create dataset step. This results in a dataset indicated by the dark blue box.
And then this dataset is used in the trained model step, which outputs a trained model and some
metrics that help us to evaluate how well that model performs. A reinforcement learning from
human feedback pipeline is a little more complicated. It might look something like this. We first create
a preference dataset. That preference dataset is used to train a reward model. The reward model is
used with the prompt dataset to tune the base large language model with reinforcement learning.
And then we get a tuned large language model and some output and training curves as well.
In reality, the pipeline that we're going to execute has a lot more steps, but more on that
shortly. The RLHF pipeline exists in the OSS Google Cloud Pipelines components library.
So to run this pipeline, you'll first import it, then you'll compile it, and then execute it.
So we'll need to make sure that we have a few different libraries installed. These are installed
for you already. But if you are in your own environment, you'll run on PIP install, and then
Google Cloud Pipeline components. And you'll also need to make sure that you have the
CubeSlow Pipelines library installed as well. So that is KFP. But these are already included in
the environment for you right now. So the first thing that we're going to do is go ahead and import
the pipeline from the Cloud Pipeline components library. So note that right now this exists in preview.
That's because RLHF is currently in preview. But eventually when it moves to GA, this will probably
move out of the preview folder here. This pipeline has been written using the CubeSlow Pipelines OSS
library. So the next thing we need to do is import the compiler from the KFP or CubeSlow Pipelines
library. So we'll say from KFP import compiler. And if this seems a little confusing, don't worry,
we're going to use all of these different elements in just a minute. So compiling a pipeline,
what I mean by this is we're going to create a YAML file. So before we can create that YAML file,
we're just going to define a name for the YAML file. So let's define the path to this file.
We'll call it RLHF pipeline package path. And we're going to call this file RLHF pipeline.YAML.
So once we've defined this, we can now execute the compile function. So this uses the compiler.
This is what we imported from CubeSlow Pipelines up here. And then we call compiler and we call
the compile function. So a whole lot of compiling here. But what we're really doing here is we're
passing in two elements to this compile function. The first is RLHF pipeline. And that is the pipeline
that we imported earlier from the Google Cloud Pipeline Components library. The next thing we pass
in is the package path right here, which is the path to our YAML file. So if we execute the cell,
what happens is compiling the pipeline creates a YAML file. So we can now take a look at this
new YAML file that's been created. So I'm just going to take a look at the first few lines.
But if you wanted to look at the whole thing, you could instead of saying head, you could say
exclamation point cat. And that would show you everything in this file, but it's pretty long.
So we're just going to look at the very beginning. What you can see here is that this YAML file
includes all of the information needed to execute a pipeline. It's basically a really long
description in natural language of this pipeline. It's got a name. It's got a description of what
it does. And then it's got all of these different inputs. So what does this pipeline actually look
like? Well, Vertex AI provides you with a visualization tool where you can see all of the components
of your pipeline. And this is what the RLHF pipeline that we're going to execute actually looks at.
It's pretty difficult to see this all on one single slide here. There are a bunch of steps.
And it probably just looks like a bunch of small boxes and lines connecting them. But we can zoom in
on one specific part of the pipeline over here on the right. And if we do that, we'll see that
this section looks a little bit like this. There are these boxes with these blue cubes on them and
these are components. Again, a component is where some code is executed. And then we have these
other boxes with these yellow triangles and these are the artifacts. This is anything that is
created as a result of our pipeline. And if you look closely, you'll see that there's the
word system artifact over here and the words component over here. So this might start to look
a little bit familiar. There is this component that says reward model trainer. This is the step
of our pipeline that trains the reward model. Underneath that, you can see a component called
reinforcer. And this is the reinforcement learning loop that tunes the base language model.
You can also see that the reward model trainer component outputs some metrics, which are indicated
here in this tensor board metrics artifact. And we will take a look at that in the next lesson.
So again, this pipeline looks pretty complicated, but it's already been written for you.
So even for your own projects, you won't be editing this RLHF pipeline component or the corresponding
YAML file. The pipeline has already been authored by the Vertex team and it's optimized for the
platform and for RLHF. And the YAML file is something that's auto generated. So you don't need to
go in and edit anything in it. You just need to use it as is. So now that we have this YAML file,
we can define a Vertex AI pipeline job and I'll explain what all of that means in just a minute.
But at a high level, this will take in the YAML file and it will also take in all of the parameters
that are specific to our use case. So let's take a look at the parameters that we're going to
pass to this pipeline. The first thing I'm going to do is make a dictionary called parameter values.
Now, there are a lot of different parameters that we're going to need. So we will take a look at them
one by one. The first three parameters here are the paths to our preference data set,
our prompt data set, and then also this evaluation data set. So we didn't talk about the Eval data
set in the previous lesson, but the Vertex AI RLHF pipeline allows you to pass in an optional
evaluation data set. What this means is that once tuning is complete, this evaluation data set will
be used to perform a batch inference job where a bunch of completions will be created for a bunch of
prompts. We will take a look at that in detail in the next lesson, but for now, all you need to know
is that we have three data sets that we are passing into this pipeline. So in the previous lesson,
when we took a look at these different data sets, we were just loading in small JSONL files directly
into memory. But for this actual pipeline, our data sets are much larger and they need to exist
somewhere called Google Cloud Storage. Cloud Storage is Google Cloud's object storage. That means
that you can store images, CSV files, text files, save model artifacts, JSONL files, just about
anything. And you'll notice that these are Google Cloud Storage paths because they start with
GS colon slash slash. So anytime you see that, that means that this is a path to an object stored in
Google Cloud Storage. Cloud Storage also has the concept of a bucket, and this is just what holds
your data. Everything you store in Cloud Storage needs to be contained in a bucket, but within a bucket,
you can create additional folders to help you organize your data. And then lastly, for text AI
requires that all three of these data sets be in the JSON lines format. So if we look back at the
notebook, you can see that for all three of these data sets, they start with this GS colon slash slash,
which again indicates that these are paths and Google Cloud Storage. And then they are all in this
bucket called vertex AI. This bucket has been created for you and is a publicly accessible bucket.
Within this bucket, there are additional folders for each of the different data sets. So for your
own projects, you'll need to make sure that your three data sets are stored in a Google Cloud Storage
bucket. And if you want some more details on how to do that, you can check out the optional lab
at the end of this course, which we'll show you how to create a bucket and how to upload data
there, as well as how to figure out what the specific path is that starts with GS colon slash slash.
For now, we can just use these data sets that I have uploaded for you already. The next parameter
we're going to set is called large model reference. So in this case, we are going to set this to
llama27b. Large model reference specifies which large language model we want to tune. Again,
in this case, we're using the open source llama2 model, but there are other supported values as well,
including textbison and the t5x family of models. So as a reminder, there are two different
models that get trained in this RLHF process. The reward model train steps sets the number of steps
to use when training your reward model. The value to set here depends on the size of your preference
data set. From experimentation, we found that the model ideally should train over the preference
data set from around 20 to 30 epochs for best results. And then reinforcement learning train steps
is the parameter that sets the number of reinforcement learning steps to perform when
tuning the base model. This depends on the size of your prompt data set and from experimentation.
In this case, we found that the model should train over the prompt data set for around 10 to 20 epochs.
Now, one thing to note is that I was giving these recommendations of how many epochs to train over
these different data sets, but this parameter here takes in a number of steps. So if you need a
handy heuristic to help you go from epochs to steps, I can show you that in the notebook.
The first thing you'll do is set the size of your data set. And this could be for the preference
data set or the prompt data set. So let's say to make this a little bit easier to understand,
let's say that our data set size is 128. Then we'll need to set the size of our batches. And again,
this means sending our data in batches instead of everything all at once. And reinforcement learning
from human feedback on vertex AI currently uses a fixed batch size of 64. So you'll need to set
this number to be 64. You can't actually adjust the number of batches. But once we have both
these set, we can determine the steps per epoch by seeing how many batches it will take to complete
our full data set of 128. So we'll import math. And that's so we can use a rounding function.
And then we can say steps per epoch equals math dot seal. And this will just round up if these
numbers don't divide evenly. We'll take our size of our data set and we will divide it by our batch size.
And if we print this number, we'll see that that is two because 64 times two is 128. So we'll take
two steps with the batch size of 64 to make it through our full data set of 128.
Once we have our steps per epoch, we can then set the number of epochs that we want to train for.
So let's say that we set this to 10, we can then determine the total number of training steps
that we'll need by multiplying our steps per epoch by the number of epochs that we want to train for.
And if we do that and we print out the number, we'll see that this will be two times 10. So we'll
need to train for a total of 20 steps. You can use this handy heuristic for your own use case.
You'll just set the size of your preference data set or your prompt data set. You'll set a fixed
batch size of 64. And then you'll set the number of epochs to train over and you can use the
guidelines that I mentioned earlier. So I'm going to go ahead and update the training steps here
for both the reward model and the reinforcement learning loop to correspond to the size of my
actual data sets here. So I'm actually not using the entire reddit data set. It's a good best practice
to execute the pipeline on a smaller subset of the data. The first time around, just to make sure
that the pipeline executes correctly. These pipelines run for many hours. So running them first on a
small amount of data is just a useful thing to do. So in this case, my preference data set was size 3000
and the batch size is of course fixed at 64. So that helped me get my steps per epoch and then
I decided to train over 30 epochs. So once I had that, I knew that my number of training steps
for the reward model was 1410. The size of my prompt data set was 2000 and the batch size is again
fixed at 64 and this helped me to determine the steps per epoch and I decided to train over 10 epochs
for the reinforcement learning loop. And that is how I got to 320. So again, this is still a
smaller amount of the full reddit data set, but in the next lab, we'll take a look at results from
training on all of the data. So the next three parameters are the learning rate multipliers and
the KL coefficients. I would say that these are maybe a little bit more advanced of parameters
and maybe not something you would set on your first try with this. You can set these to the default,
but as you start really tuning the pipeline for your use case, you might want to adjust these a
little bit. And I have the defaults set here already. That's one for both of the multipliers and
0.1 for the KL coefficient. The reward model learning rate multiplier and reinforcement learning
rate multipliers are constants that you can use to adjust the base learning rate when either
training the reward model or during the reinforcement learning loop. You can't actually adjust the
learning rate itself and that's because generally you want the learning rate to match the learning
rate that was used to train the base large language model and you might not know that off the top of
your head. So the learning rate is fixed for you by the pipeline, but you can't adjust these multipliers.
So what that means is if you multiply by a number greater than one, you're going to increase the
magnitude of gradient updates applied at each training step. But if you multiply by a number less
than one, you'll decrease the magnitude of these updates. Next, we have the KL coefficient. This
is a regularization term that helps to prevent something called reward hacking. So for example,
let's say that our reward model tends to give higher rewards for completions that contain positive
words like excellent superb, great. During the reinforcement learning loop, our base large language
model might learn that if it generates completions that are filled with positive terms but don't
actually make a whole lot of sense, it will still result in higher rewards. So for example,
our base large language model might start learning to produce completions that just have all of these
positive words in it like excellent, fantastic, awesome, great. And it doesn't really make a lot of
sense to a human reading these responses, but the reward model is still giving high rewards.
This is known as reward hacking and the KL coefficient essentially helps to prevent reward hacking
by preventing the model from diverging too far from the original model. So the tuned model essentially
is penalized if it starts to diverge too far from its initial distribution and break the functionality
of the original large language model. If you set this KL coefficient to zero, there is no penalty
at all and the larger you set this coefficient, the more the tuned model will be penalized for diverging
from the original large language model. Okay, we are on to our final parameter and this is the
instruction, which I've set here to be summarize and less than 50 words. The instruction lets the
model know what task it needs to perform. So this text is going to get prepended to each prompt in
your data set, both the preference and prompt data sets. So you only want to set this parameter if
you don't already have the instruction included in your prompts. So if you recall in the previous
lesson, when we took a look at the input text keys in our data sets, none of them had an instruction
that said to summarize the text in less than 50 words. If we did include this instruction already
in our data set, we wouldn't need to set this instruction parameter because these base models have
been trained over a large variety of different instructions. You can make this instruction parameter
a simple and intuitive description of the task that you want the model to complete. But with that,
we have wrapped up all of the parameter values that we need and we are ready to actually execute
this pipeline. So in this example, we are summarizing Reddit posts, but given the information that you
have about RLHF, can you think of some other tasks and instructions that would be well suited to
reinforcement learning from human feedback? For example, write a response to the following text,
and in that case, your text that you might have could be the Reddit post. Now that we have all
of our parameter values defined, we are ready to create a pipeline job. What this means is that this
reinforcement learning from human feedback pipeline is going to execute on vertex AI. So it's not
going to run locally here in our notebook, but it's going to run on some server on Google Cloud.
In order to do this, we first need to authenticate to Google Cloud and initialize the vertex AI
Python SDK. For this course, we've done that setup for you. But if you want to learn how to do this
for yourself and your own projects, you can take a look at the optional lab included at the end of
this course. So I'm importing an authenticate function that we have already written in this Udl
file. When I run this authenticate function, it's going to return the credentials, which is how we
communicate with vertex AI, and then the name of our project where all of these services are running,
as well as the name of a bucket, where we can store some generated artifacts from our pipeline.
The last variable we'll need to set is the region, and this is the location of the data center
where we're actually going to run this pipeline. Some services are only available in a certain set
of regions, and this reinforcement learning from human feedback pipeline is available in this
Europe West for region. So that's why I've set that here as this value. Next, we need to import and
initialize the vertex AI pipeline SDK, and if you're writing this in your own environment,
you will need to pip install Google Cloud AI platform, but we have done that already in this
environment here. So I'm just going to import that library now, and then once I've done that,
we can initialize AI platform, so we'll call the initialization function.
And this is just something you need to do anytime you want to use this AI platform SDK.
So we'll set a couple of different variables here. We'll set the project ID, which we loaded
earlier in the authenticate function. We'll set the location to be the Europe West for region
that we just set in the previous cell, and then we will specify our credentials.
And if we execute the cell, we would have initialized the Python SDK. So once we've done that,
our second to last step here is to create our pipeline job. And so I'm going to call this
job, and we'll call AI platform dot pipeline job. And to this pipeline job, I'm going to pass
in a few key parameters. So the first thing we'll pass in is a display name, and this is just any
string name for what you want to call this pipeline job. So here I'm calling it tutorial,
RLHF tuning, but you could change this to be anything you like. After we've done that, we need to
pass in a staging bucket. And so this is the pipeline root parameter. And basically what this
means is that our RLHF pipeline is going to create a bunch of artifacts along the way. It's going
to output some different files, and we just need some central location to store all of those things
that are going to be created. So I'm just going to store all of that in a Google Cloud storage bucket
that is saved in this variable staging bucket. Now we set the template path. And if you recall,
at the very beginning of this lesson, we created a YAML file. So if I print this, this is the path
to our YAML file. And that was the YAML file that defined all of the information about our pipeline
that we want to execute. The very last parameter we'll pass in is parameter values. And this is that
big dictionary we created up here of all of the parameters that were specific to our reinforcement
learning from human feedback that were specific to our Reddit use case. So once we have defined all
of these parameters here, we can create this job. Now we take a look at that. This creates this
AI platform pipeline jobs object. And the very last step here is to run the job. And we do that by
calling job.run. Now this job is going to take several hours and it's going to require a lot of
hardware. So for the purposes of this online classroom, you're not going to actually run this pipeline.
But in the next lesson, you'll take a look at the results of a pipeline that's been executed already.
So if you did want to run this in your own projects, what you would do is call job.run. And this
will create and execute a pipeline for you. In the next lesson, we're going to take a look at the
results of a pipeline that's already been executed for you. So these were some results run by my
teammate, Bethany, who ran an RLHF tuning job on the full giant Reddit data set. And that job
took over a day to finish running. So I'll see you in the next lesson where we'll take a look at the
results.

## 4. Evaluate the tuned model
RLHF involves a lot of different steps,
but we don't just want to train a model.
Our ultimate goal is to create a new large language model
that performs the task we care about better
than the original large language model.
So in this final lesson of this course,
we're going to discuss some different strategies
for evaluation and take a look at results
from the newly tuned model.
Let's get started.
There are a few different things we can look at
when evaluating large language models.
Though I should mention that LLM evaluation
is still very much a developing area of research,
and there's a lot more to say
than what we can fit in this single lesson.
But at a high level, here's what we might look at.
First, we can look at the training curves,
like loss produced during the training process
to see if the model is actually learning.
You might have done something like this
in the past when training neural networks
or other machine learning models.
Second, we can look at automation metrics.
These are measures of performance
that can be calculated using algorithms
or mathematical formulas that require ground truth.
So this might include some familiar metrics
like accuracy or F1 or some metrics
more common to generative tasks
like the Rouge family of metrics,
which help you to determine how similar
a piece of generated text is
to a human generated reference text.
Third, we can do side-by-side evaluation
where we compare the performance of two models
against each other using one set of input prompts.
This allows you to calculate the win rate,
which tells you what percent of the winning responses
were produced by a particular model.
In the case of RLHF, researchers have found
that the training curves and side-by-side evaluation
have been most useful.
So if you're familiar with the Rouge metric
and you're wondering why that's not as valuable,
even though it's often used for summarization tasks,
it turns out the score might not be a suitable measurement
for RLHF because it's not really the objective
that RLHF aims for.
In other words, the Rouge score does not
describe the alignment with human preferences very well.
It simply tells you how close the generated text is
to some reference text.
So some research has even shown that the more severely
we optimize for Rouge, the worse the model performance is
in the case of RLHF.
So we're going to start by taking a look
at some of the training curves.
The Vertex AI RLHF pipeline that we created
in the previous lesson, outputs some training curves
to TensorBoard.
TensorBoard is an open source project
for machine learning experiment visualization.
And you can install TensorBoard with pip install TensorBoard.
But again, this is already installed
in this environment for you.
So we're going to examine these curves
to see how well the model is learning.
After we've loaded the TensorBoard extension,
we can launch TensorBoard.
So we'll do that again with the percent sign.
This time we'll type TensorBoard
and then you'll type dash dash log der.
And then you'll need to provide a folder
that has your TensorBoard log files.
So in this case, I've gone ahead
and I've uploaded the TensorBoard log files
for the reward model training
to this directory called reward logs.
So we can actually take a look
at what is in this directory.
And you'll see there's one file
and it ends with this very long string
of numbers here, 1, 1, 0, V2.
And this is the log file that was created
during the training process.
So in a minute, I will show you
how you can find these log files for your own training jobs.
But before we do that, let's just go ahead
and take a look at what's in this file
and visualize it with TensorBoard.
So if we execute the cell right here,
we will see that this launches TensorBoard.
So I'm gonna go ahead and I'm gonna scroll down
to rank loss, which is the metric
that we care about right now.
This is a loss function that was used
to train the reward model.
So like other loss functions,
generally what you wanna see is this curve
decreasing over time and then converging.
So starting to plateau here,
which is exactly what you see here.
And in fact, it looks like it converged
and we kept training for quite a while after that.
So if you were gonna run another tuning job
with the same data,
you might want to train for even fewer steps.
So this actually looks pretty good.
So let's go ahead and take a look
at the curves produced during the RL loop.
So again, I'm going to call the command TensorBoard
and we'll say logder.
And this time we will pass in a different directory
and this is a directory I've created
that has a log file for the reinforcement learning step.
And like I mentioned earlier,
I'm gonna show you how to find these files in just a minute
but let's first take a look at what they look like.
So this will launch TensorBoard again
and we can scroll up here.
And what we wanna take a look at here
are two particular metrics.
The first is the KL loss.
This tells us how much the model is deviating
from the original base model.
So what you want this to look like is you wanna see a curve
that's increasing and then eventually it starts to plateau.
That's not quite what's happening here.
It looks like the KL loss is kind of all over the place
and it starts off higher, it starts to decrease.
Doesn't quite look like it's converging.
And in fact, if we collapse this
and we take a look at the reward,
we also, we'll see that this is also all over the place.
So ideally the reward curve should also increase over time
as the model learns the reward gets higher and higher
and at some point it will plateau.
So ideally this is what we would want both of these curves
to look like.
You can see this KL loss continues to increase
and at some point it sort of plateaus
and the same thing for the reward.
The reward keeps climbing higher and higher
until at some point it plateaus.
But we're not really seeing that for either the KL loss
or the reward here and these TensorBoard files.
And so that's a pretty good indication
that your model isn't really learning.
In fact, in this case, it kind of seems like it's underfitting
because there's no real trend here
and either the curves from the KL loss and the reward.
But in this particular case, that wasn't too surprising.
These were log files I pulled from tuning the model
on a small subset, around 1% of the total data set.
So next, let's take a look at some logs
that were produced when we trained on the full data set.
So we'll call this TensorBoard command one more time.
And again, we'll say logger and we'll pass in one last directory,
which is called reinforcer full data logs.
And if we launch TensorBoard here,
we should see that our curves look a little bit closer
to what we're expecting.
So you can see that the KL loss here
continues to increase and increase and increase.
And at some point, it sort of starts to plateau.
The same thing for the reward.
We can see that again, this is increasing,
which is exactly the kind of behavior we want to see.
We want to see that the reward increases over time
until at some point it stabilizes.
So these were some training curves that were generated
from a large scale tuning job run by my teammate, Bethany.
She actually ran a bunch of experiments
with this reddit data set and the Lama 2 model.
And I can show you what parameters she used specifically
to achieve these results.
So here is the dictionary parameter values
that we created in the previous lesson.
And for starters, for the preference data set,
the prompt data set, and the evaluation data set,
she trained on the full data set
instead of the smaller, sub-symboled version of the data set.
So if I adjust this path here,
this is the Google Cloud Storage Path
that leads to the full data set for all three of these.
So instead of text small, the directory here
is just called text.
She fine-tuned the Lama 2 model.
And the reward model train steps were set to 10,000
as well as the reinforcement learning train steps.
Reward model learning rate multiplier was 1.0.
And the reinforcement learning rate multiplier was 0.2.
The KL coefficient was set to the default of 0.1
and the instruction was the same as before,
summarize, and less than 50 words.
So now let me show you how you can access
these tensorboard files for yourself in your own projects.
So currently, we've just been interacting with Google Cloud
in a notebook via the Python SDK.
But if you go to console.cloud.google.com
and go to your Google Cloud project,
under the vertex AI section, you'll
see a little button that says Pipelines.
So if you select Pipelines in the console,
it will open up all of the pipelines
in a particular region that you've run.
And so under runs, you can go ahead
and select your pipeline here.
It should be somewhat easy to find.
It should have the same name that we gave it earlier,
which was Arleigh Jeff train template.
And when you click on this pipeline,
it will open up that visualization
that I showed you earlier of all of the boxes
and all of the lines.
Once you've opened your pipeline,
you can zoom into the top right corner
where it says reward model trainer.
As a reminder, this is the component
that executes training of the reward model.
You can see that this component produces an artifact
called tensorboard metrics.
So if we were to click on this tensorboard metrics box,
it will pop up on the right-hand side
with this URI over here, which is a path in Google Cloud Storage.
And if you click on this path right here,
it will open up the tensorboard logs for you.
If you want to find the specific file within that directory
yourself, you should see something called events out TF events
that will end in W110V2.
But that's how you find your specific tensorboard logs
for the reward model trainer.
For the reinforcement learning loop, it's pretty similar.
You'll just click on the reinforcer component
and then open up the corresponding tensorboard metrics
artifact that is produced.
And again, that will also open up a URI,
which is a path in Cloud Storage.
And if you click on that path, you'll
be able to find your tensorboard logs.
The training curves that we looked at
can help us get a sense of whether or not
our model is learning.
But at the end of the day with these large language models,
sometimes the best way to evaluate them
is just to look at the completions
that they produce for a set of input prompts.
So you might remember that in the previous lesson,
when we created our pipeline job,
we passed in an evaluation data set.
This is a data set of prompts, no completions,
just summarization prompts.
We're calling this an evaluation data set,
but it might differ from how you are used
to using evaluation data sets with machine learning
in the past.
This data set is just passed to the tuned model
for a bulk inference job.
So what that means is that once our model has been tuned,
we generate completions for all of the prompts
in this evaluation data set.
We don't calculate any metrics.
We're just calling the model and producing
some kind of text output.
So to make that all a little bit more concrete,
let's take a look at some of these evaluation results.
I've got ahead and loaded in a small subset
of the evaluation results here for you to examine.
And this is also a JSON L file.
So we'll start by importing JSON.
Then we will define the path to where these results are.
So we'll call this evalTunedPath
equals evalResultsTuned.JSONL.
Next, we'll define an empty list like we did before.
We'll call this evalDataTuned.
And then we will loop over this JSONL file
and append the data to our empty list.
Next, I'm going to use that printD function
that we defined in the second lesson,
which just helped us to visualize the keys
and values for addictionary.
So we will import printD.
And once we've done that, we can use it.
So I'll call printD.
And let's take a look at the first element in this list.
So this first element here is a dictionary
that has a key called inputs.
The value for inputs is itself another dictionary
that has a key called inputs pre-tokenized.
And if we look at that value here,
we'll see that this is a prompt.
It's got our instruction summarized in less than 50 words.
This was the instruction that we set in the instruction
parameter when we kicked off the pipeline.
And it's been prepended to our evaluation data set.
After that, we have the Reddit post.
So before anything, not a sad story or anything,
my country's equivalent of Valentine's Day is coming up
and I had this pretty simple idea to surprise my girlfriend
and it would involve giving her some roses, et cetera.
This prompt also ends with summary colon and brackets,
which we saw in the second lesson.
So this prompt here was sent to the tuned model
and the tuned model produced this prediction result down here,
which says, my country's equivalent to Valentine's Day
is coming, want to surprise my girlfriend with roses,
but I don't know if she would like getting some.
Any ideas on how to get that information out of her
without spoiling the surprise?
So this is the summary that our tuned model produced
for this input prompt.
So next, let's do some side-by-side evaluation.
What this means is we're going to look at some completions
on the same set of prompts for our Lama-2 model
before and after we've run this tuning job.
Next, I'm going to load in a file
that has inference results from the base model.
This is the Lama-2 model before we executed tuning.
So first we'll define the path to this data set
and it has the exact same input prompts
that our evaluation data set we were just looking at as well.
And again, we'll create a new empty list.
These are the results from the untuned model
and we can loop over this file and each time we do that,
we will append the data to this list here.
So now we have two lists.
We have a data set that has results
from the tuned Lama-2 model
and then we have a data set that has results
from the untuned Lama-2 model.
If we look at the first example in this untuned data set,
what you'll see is that the prompt is the same,
but the completion is going to be different
because it came from the model
before we ran our RLHF tuning job.
You can see that it's the same prompt as before
about Valentine's Day and roses,
but the prediction is different.
The untuned model produced a summary,
the author wants to surprise his girlfriend
with roses on Valentine's Day,
but he doesn't know if she likes roses.
He wants to find out without spoiling the surprise.
So if we scroll back up to the completion produced
by the tuned model, you can see that it is in fact different.
And one difference you might notice
is that the tuned model produced a summary in first person.
So in the same voice as the original Reddit poster,
while the untuned model refers to the author,
instead of saying it in the same voice
as the person posting on Reddit.
But take a look and see if you can find any other differences
and which of the two responses you prefer.
To make it easier to compare all of the results,
so we don't have to keep printing each element
and scrolling up and down like this.
I'm gonna put everything into a data frame
so we can do some real side-by-side evaluation.
The first thing I'll do is make a list of the prompts.
So we'll call this prompts.
Okay, so what I'm doing here is I'm looping
over the data set of results from the tuned model.
And for each sample in that data set,
I'm extracting the value for inputs pre-tokenized.
So that's gonna correspond to the prompts.
So if we execute this,
we can then take a look at what we've created here
and that is a list of all of the prompts in this data set.
So you can say here's one prompt,
here's another, et cetera.
So now that we've extracted all the prompts,
we're going to extract the completions
for both the untuned base model and the tuned model.
So first we'll extract the completions
from the model before it went through tuning.
Call this untuned completions.
And this time, we'll be extracting the value
for the prediction key.
And if we do that, we will have a list of completions.
And these are the completions from the untuned model.
We'll do this one more time
and extract the completions for the model
after it went through tuning.
So this is gonna look really similar to the previous cell.
We're just going to be looping over the data set
with completions from the model after tuning.
And if we look at that, you can see that this is also
a list of completions and this from the model
after it's gone through tuning.
So lastly, we're going to put everything together
in one big data frame.
So to do that, we will first import pandas
and then I'm going to make a data frame called results.
And for the data and this data frame
we'll make one column called prompt
and here we will pass in that list of prompts.
Then we will make another column called base model
and this is going to be the completion
for that specific prompt generated by the model
before it was tuned with RLHF
and then we'll make a final column called tuned model
and this is the completion for the prompt
that was generated after the model was tuned with RLHF.
And just so we can visualize all of this on the screen here,
I'm going to set a pandas option
to display max call width and this will just help us
to visualize this a little nicer.
All right, so let's take a look at our results data frame.
There are, we scroll up to the top here, three columns.
We've got our prompt and this prompt here
is the same one we looked at earlier about roses
and Valentine's Day and then we've got
the completion generated by the model before tuning
and the completion generated by the model after tuning.
And there are a few more prompts in here.
This one is about a senior in high school
who wants to study computer science in college.
This one here is about applying to jobs
and we've got one at the bottom here
about applying for credit cards.
So you can take a look at all of the data
in this data frame here and do your own side-by-side evaluation
where you try and identify
which of the two completions you prefer,
the response from the model before tuning
or the response from the model after tuning.
But that's essentially how you would do
a side-by-side evaluation.
Now if you're wondering for your own RLHF tuning jobs,
how do you access the batch evaluation results?
Well, you'll do this again by going into the cloud console
and opening up your pipeline,
but this time you will zoom in on the component
that says perform inference.
Under perform inference, you'll see a component
called bulk infer.
This is the component that just performs a bulk inference job,
meaning it takes in our JSON L file of prompts
in our evaluation data set
and then calls the model to produce completions
for each one of those prompts.
If you click on that component,
you'll see on the right-hand side a box popup
that says output parameters
and specifically the parameter output prediction GCS path
will point to a location and Google Cloud storage
that has the JSON L file.
So you can click on that link and then
download the JSON L file and take a look at the results.
So to finish off today,
I want to talk about two new interesting techniques
in the world of RLHF.
The first is RLAIF or scaling reinforcement learning
from human feedback with AI feedback.
This is a really interesting technique
where we are actually creating preference data sets
that are labeled by an off-the-shelf large language model.
So previously, when we looked at the preference data set,
it was labeled by human laborers,
but actually in the research area,
they're now looking at different ways
to use a large language model
to actually create that preference data set.
So this is a pretty interesting paper
that I would recommend taking a look at
if you're curious to see how we might use an AI model
to help generate a preference data set.
And then similarly in the topic of using LLMs
to help us in this RLHF process,
another interesting technique is called autoside by side.
And this is where you perform side-by-side evaluation,
like we did in the notebook,
but instead of having a human being
look at the results before and after tuning,
you actually use a third arbitrary model,
which is itself a large language model
instead of a human laborer.
So what this means is that this third large language model
looks at the responses from both the untuned model
and then the tuned model,
and determines which one it likes better,
and it often also provides an explanation.
So you can see a screenshot here in the slides
from the autoside by side service in Google Cloud,
where we have a prompt,
we have the response from the untuned model
and the response from the tuned model,
which looks pretty similar to the pandas data
frame that we created.
But after that, there's a column
that indicates which of the two completions
be third large language model preferred,
as well as an explanation provided
by that third large language model
as to why it preferred that specific result.
But these are just some interesting new areas of research
that hopefully give you an idea
for how this field is evolving.
So that wraps up our lesson on evaluating
the results from RLHF.
So I'll see you in the next video
where we will conclude the course
and wrap up everything that we've learned.

## Conclusion
Congratulations on finishing the short course on RLHF.
We started off with a conceptual overview
of how RLHF works and the different data sets involved.
Then you saw how to tune the OSS Lama 2 model
using an ML pipeline and how to evaluate the results.
I hope this prepares you for when
you're ready to tune your own models with RLHF,
and I am so excited to see what you build.
