# DSPy: Build and Optimize Agentic Apps

## 1. Introduction
Welcome to DSPY, Build and Optimize Ajectic Apps built in partnership with Databricks.
In this course, you learn how to build and optimize your GENIAR application using DSPY.
When you start building configs in your apps, one per challenge is writing good prompts
for NLR.
Our often tried dozens of prompts tweaking whereas changing formats hoping to get better
results.
This process takes a lot of time and the prompts often even break when you change the underlying
ON.
DSPY's streamlines and authorizes this whole process.
You define what inputs your model needs and what outputs it should return and also provide
a dataset of inputs and design outputs.
DSPY can then optimize your AR programs to get better performance with much less manual
work.
I'm delighted to teach the UC instructor, Jenschen, who is a software engineer at Databricks
and copies the developments of DSPY.
To exchange you, in this course, you will learn how to build AI programs with DSPY.
DSPY has two main building blocks, signature and a module.
When you are trying to build a component of an application, specifying the signature
tells the system where inputs and outputs should expect from them all component.
For example, a sentiment analysis program will have a string text has an input and an
integer representing a sentiment has the output.
A module then uses those signatures to actually call a language model and get results.
Sometimes an app doesn't work, but we are not sure why.
In this course, you will also learn to use an workflow tracing to help you see exactly
what's going on in each step of the application.
What data was used, what tools were called, what the model returned, and where things
broke.
With just one line of code, you can turn on this tracing feature for your app.
When we use a DSPY, it's an optimizing agent to work those.
It will have a complex workflow that takes an input and uses multiple steps of processing
to generate the outputs, maybe using an RM on multiple of these steps.
When DSPY's optimizer can take your agents as well as an evaluation data sets and metric,
and based on that, automatically search for better prompts for all steps.
Our scene is sometimes though really high quality fuel shop prompts on your data, that
the way that it's far better than I as a human could likely have achieved by hand.
In this course, you use DSPY's optimizer to optimize prompts in a rag app that answers
Wikipedia questions.
Many people have worked to create this course.
I'd like to thank Omar Katal, Kha'i Yin, Trishster Opsalon, and Tobu Hirata for Databricks.
From T-Blend.ui, Ashmeu Gagari and Brendan Brown also contributed to this course.
The first lesson will be an introduction to DSPY.
One surprising thing about DSPY is how fuelizer code it takes to implement and added essentially
autobase the prompt engineering process.
If you've gone to the next video to see how this works.


## 2. Introduction to DSPy
In this lesson, you will learn about what DSP is,
and more importantly, what is special about DSP,
and why many developers are using DSP
to build their Gen A apps, let's type it.
So, what is DSP?
At a very high level,
DSP is in Gen A authoring framework that
simplifies the development of Gen A applications.
You can use it to build a rock system to agent.
But as a user, I would ask this very natural question,
are we reinventing the wheels?
We already have so many options over there.
The answer is no, let's explore together.
Before we learn about the specialty of DSP,
let's talk about the problems we're facing.
Since late 2022, you do the success of Gen A models,
there came the rise of Compound AI system.
Here, you can see a question answering system
and also a co-generator.
Compound AI system is a system that has multiple modules.
Each module handles a sub-task,
which can make how I'm cause,
or could just be a regular tool colleague.
Combining all these modules together,
we form a big and powerful AI system
that can handle complex tasks.
Like retrieval augmented generation,
we're co-generator as you see in the picture.
When building a Compound AI system,
we are all facing a problem from prompt engineering.
We do prompt engineering because we know that
with a better prompt, we'll get a better result
from the language model.
But prompt engineering could be very messy
because we are tweaking the strings
and we don't really know what changes make actual difference.
In practice, that usually means
we end up iterating over 50, 100, or even more prompts.
And each prompt could be very long
like tens of thousands of words.
Making that even worse, if we switch the model,
then we need to start over
because prompt engineering is heavily biased
towards the language model.
As a summary, prompt engineering is both brittle
and time-consuming.
The other problem comes from the framework itself.
Frameworks are very powerful because they simplify
and standardize the experience of building things
like agents or rack and so on.
But there have been more and more complaints
that I don't like the user frameworks
because instead of value, I'm seeing more trouble.
One very solid concern is that I'm being forced
to learn the framework contract,
which sometimes means on a necessary overhead for me,
rather than keeping me focused on the logics.
And it's really hard to migrate off the framework
on your existing code if I decide to switch over to another one.
To address both problems, there came the DSPI.
Now let's answer the question,
what is DSPI again with more details?
DSPI is a flexible lightweight framework
that simplifies the interaction with our own
and provides automatic program optimization,
including prompt optimization,
and our own weights fine 20 through the DSPI optimizer.
DSPI also provides seamless productization support,
extreme async and so on.
We provide you with a building blocks
for AI applications in DSPI,
but we don't restrict the way you build applications.
It's both easy to migrate to DSPI and migrate off DSPI.
We have three special things inside DSPI.
And the first one is our own agnostic programming
instead of our own bias prompting.
Basically, instead of the impromptu engineering,
DSPI interacts with our own
by defining input fields and the output fields.
In DSPI context, you can treat our own endpoint
as a well engineered rest for API,
but different from traditional APIs,
the input and output are defined as client side.
Don't worry if this is not fully clear yet.
We'll cover this right in the next lesson.
The second part is a seamless productization
through both native DSPI features,
extreme cache and also through the closed integration
when the mouthflow.
MLflow is an ML and AI ops tool
that streamlines end-to-end development of AI applications.
For example, helps debugging AI programs
by MLflow tracing,
tracking development by MLflow experiments
and deployed application with MLflow deployments
or learn more about it in lesson three.
And in lesson four,
you will learn about automatic program optimization
after you build your AI applications with DSPI.
You can create a DSP optimizer
and apply it on your program
and automatically get quality improvements.
DSPI is trusted by industry.
We have managed successful integration
with lots of enterprise users.
Please check it out more at DSPI.AI slash community
slash use cases.
That's all about this lesson.
In next lesson, you will learn
how to build your DNA applications in DSPI.
See you there.

## 3. DSPy Programming - Signatures and Modules
In this lesson, you will learn how to build AI applications with DSP.
Let's dive into the code. This lesson comes with a lab.
In the lab, you will learn DSP fundamentals by building a simple sentiment analysis program
with DSP building modules, and you will also learn how to build a custom-imagined with DSP,
with a demo of building the name of the celebrity game.
It's a very simple game where the player one thinks about celebrity name,
and player 2 starts asking you a sooner question until finding the name or use up all the question code.
DSP programming has two important abstractions, signature and module.
Signature is where you define the input and output contract of your alarm cause,
and DSP module is the interface of talking to the alarm with custom logic.
Let's look at them one by one. As we mentioned in lesson one, that in DSP context,
interaction with the alarm is similar to calling a REST4 API that has a wild defined input and output format.
But this data format definition happens on client side instead of the server side, like a normal REST4 API.
The definition is through DSP signature, ensure language.
Signature defines input and output fields along with types and annotations.
There are two ways of defining a DSP signature, class-based signature and stream-based signature.
The first way is class-based signature. All you need to do is sub-classing from DSP title signature,
and marker input fields by DSP title input field, and output fields by DSP title output field.
And optionally, you can provide type and annotation to his field.
There are five important parts of a class-based signature.
The red box, which is the dog string of the signature class,
is a signature instruction which defines the purpose of the alarm call.
Like a brief overview of your task, this usually only requires a few sentences,
but if you already have an existing prompt and don't want to simplify,
you can paste the whole prompt into the dog string.
The orange box is the field name.
You will use this name to carry input data and access the output data.
The blue box simply tells a program if that's an input field or output field.
And the purple box carries actual information about the field,
and it's useful when the field name is not self-explanable.
Lastly, the green box carries the type information,
which could be unimprimited Python types,
or be adding custom class with identity models.
This is mostly useful for output fields.
If you specify the type, when you access the output fields,
the value will automatically be of the type you desire.
DSP also supports a lighter way of signature definition,
costume-based signature.
You just write the input fields before the arrow,
and write the output fields after the arrow, and separate fields by comma.
This is good for prototyping, but for general usage,
we recommend go with a class-based signature for flexibility and more powerful support.
Now we have defined input and upper format,
but we still only have static information,
and we need to define a way to utilize the signature to talk to the L1.
That is the purpose of DSP module.
Module is a minimal building block of a DSP program,
and in most situations, has signature attached.
The simplest module is DSP at a predict,
which formats a user query to L1 prompt,
and parses L1 response according to the attached signature.
Modules have configurable attributes assigned from the signature.
For example, attribute demo carries a future examples.
DSP modules can be customized for implementing custom logic,
and that module can consist of submodules.
DSP provides a list of building modules
that makes it easy for users to get started,
and the most important one is DSP predict,
which simply does our interaction,
and it is a building block for all the complex modules.
Channel thought is also commonly used,
aside from the plan response,
that also asks for reasoning behind the answer.
React, stand for reasoning and act,
will be used in a next lesson,
which is a common standard for building AI agents.
DSP program of thought, similar to React,
but the tool calling is just a code, and DSP refine.
With that, users can set a reward function and threshold,
if the threshold is on met, which is issue-richwise
with our own feedback.
That's a definition of DSP module,
as talk about usage right now.
To use the building modules,
simply pass a signature to the building modules,
and invoke the building modules through input fields
by setting the keyword argument.
For example, in a slide,
you create a channel after our instance,
which has a single input field called question,
then when you invoke the module,
you just pass a value into the question.
We have a complete list of building modules
available on documentation site,
please read out more there.
But more commonly,
you have complex logic that cannot be fully covered
by previous modules.
In those case,
you will need to write a custom module
to define a custom logic.
This is very similar to PyTorch.
You subclass from the DSPada module class,
and implement the forward method
with your custom logic.
You create a module instance,
and the instance is a callable
similar to previous modules.
This is very flexible.
You can call adding Python function
within your adding frameworks,
a LAN chain, or Lama in-depth,
or adding tools like SQL or file system handler.
As long as that is the Python program,
you are allowed to put that in the forward method.
That's a lot of contacts.
Let's do some coding to get a better sense
of how this module signature system works.
Let's get started with the coding.
So before we start,
when you set up the API key,
and don't worry,
in this lab,
we have set up the key for you.
Simply call our helper function
can open the API key
and set it as an environment variable
that you are good to go.
The first step of DSPada programming
is choosing your LAN.
In this lab,
or I used a GPT-4 mini,
and to change our own,
I'm simply changing the string here.
It is following a form
at a provider name,
followed by a model name.
Provider name is like openAI,
anthropic,
and the model name is actual model name,
like GPT-4 or that's mini,
or GPT-4L.
Now let's start building a sentiment classifier
to see how the module signature system works.
As we're mentioning a slide,
we recommend going with a class-based signature,
and here we start from DSPada signature,
and write the task description in a dog string.
An input will be a single string
called text of type string
with some actual information,
marked as DSPada input field.
An output field will just be a sentiment of type integer,
marked as DSPada output field,
some actual information,
and restricted in the range of 0 to 10.
This part is called pendante constraints,
which is fully supported in DSPada.
We can also use a string-based signature,
just to write an input.
Before the arrow,
an output after the arrow,
we can see that this is clean,
but losing some information
from class-based signature.
Now we have the signature,
which is the static information about input and output,
we need to create a module
to actually interact with EL1.
Let's use the very basic DSP module,
DSP predict,
which is a feed-to-signature
into DSPada predict,
and create an instance.
Now it's time to invoke an instance,
and we only have one input field text
to specify the value for a text,
and call that,
and let's see the print.
The output is a DSPada prediction,
which is similar to a dictionary,
but allows both keyword accessor and dog accessor,
has only one value
with maps to our output field sentiment
of type integer in the runs of 0 to 10.
Let's see how we access the sentiment field,
we can use both keyword accessor,
word dog accessor,
which will give you the same value.
Let's see how to change the arrow I'm behind the scene.
To do that,
simply call DSPada configure,
and change the arrow I'm instance,
here let's change that to be GPT40,
and let's feed the same input
to see if that can generate a different value,
we'll give the same value.
Alright, that gives us the same value.
Now let's change back to the arrow I'm to be for a minute.
Now a lot of people may ask this question,
where is my prompt?
This looks a clay,
but there is definitely some prompt,
somewhere when we talk to the arrow I'm.
To answer this question,
in DSP,
we provide an API call inspect history,
and the end number here determines how many entries
you want to pull from the memory,
let's run that.
The console output is a pretty print
of the multi turn message,
and our response.
The system message maps to the
system roles message,
has the sequential information,
like the input fields,
output fields,
patented constraints,
and also determines the input output format
we talked to the LM.
And the user message carries actual user input,
and is formatted according to the format we define above.
And the response section is our response,
which is also formatted according to the format we defined.
Let's now try a different building modules.
We use the channel of our module,
and see what will happen for the output.
You can see that in addition to a sentiment field,
we solve the same values before,
we also include a reasoning part in the output.
And let's see why we get a different output.
We can check the LLM interaction history
by calling the DSP in Spark history.
And we can see that the output fields,
in addition to the sentiment,
include a reasoning field,
and has a specific format.
And the response can tend to field,
reasoning, and get parsed by the module.
Let's see what's happening behind the scene,
and explain by the minimum DSP module,
DSP at our predict.
DSP at our predict has signature
and other information like demos attached to itself,
when we receive the user input.
In the fourth method,
we can add all the information
to something called DSP adapter,
which talks to the LLM.
After receiving all the information,
including signature,
user queries, and other attributes,
the adapter formats the actual prompt,
combining all this information,
and sends the prompt to the LLM.
The prompt tells the LLM about response format,
according to the adapter type we use.
We will show in a lab how to change the adapter
based on a language model,
you can write DSP automatically selected for you.
Let's type deep into the prompt.
We can see the fields information are at the top.
Then we define input and output data format,
where the default adapter is a section header followed by the value.
Then the user question is formatted according to its defined format
in the user message.
The output format is very important,
because only if we know the data format,
we can extract the field's value automatically.
The output flow is just a reverse.
The LLM gives back response in our defined format.
Then the adapter parses it into the output fields,
and send back to the module.
Because the response comes in the format,
we define in the prompt.
The adapter knows how to parse it into the required fields.
The parser result is wrapped in the spada prediction,
which is similar to dict,
but allows both dot accessor and key accessor.
To summarize in concise language,
DSP combines the signature and the module information
and actual inputs into a multi-term prompt,
and parses the LLM response according to the signature.
So the LLM functions like a restable API
with a well-defined inputs and outputs.
Now let's build a complex program
by customizing DSP module.
Let's first talk about how to use a different adapter.
To do that, simply call PSPodelConfigure
and send an adapter to be your favorite adapter class.
And here we use JSON adapter,
which is a good adapter if your model supports struct output,
like GPT-40, GPT-4 whole meaning.
Let's invoke the same channel after all instance
with the same input,
and let's see what happens with a different adapter.
We can see that in addition to the original ones,
we ask the outputs to be in a JSON format
instead of the section highly followed by the value.
And the response will be formatted as a JSON object
and so that adapter can parse it.
Now let's start building the name of the celebrity game
by customizing a DSP module.
Let's recap what the game is doing.
Player 1 is us, thinking about celebrity name,
and the player 2 is the LOM.
Starts ask a useful known question
until find the name or use of all the coders,
we give 20 for this game.
Now let's build a build a module.
Before we talk about what happened inside module
and how we build that,
let's play with that to get a sense of how it works.
We think about celebrity name
and type the name, use a brown James.
It's starting to ask a yes or no question.
And we just answer that, not an actor,
not a musician, sports figure,
possible yes,
current player yes,
lakeers yes,
brown James yes.
So get to the answer.
Let's take a look of how we write this module.
So we start class from DSP out of module
and we have two sound modules.
The first one is a question generator,
which generates a yes or no question.
And it is a channel thought module
with this question generator signature.
Let's take a look at the signature.
The signature has two input fields,
which is passive questions and passive answers,
and starts with empty list.
And it has two output,
a new question for the yes or no question,
and gas made indicating if that's a generic gas,
or let's say the record gas on the name.
The second sub module is a reflection module.
After we wrap up the game,
we want to do a self-reflection,
which takes in the correct celebrity name,
and the final gasser,
and the passive question and answers,
and helper will be a single string
for the reflection process,
what's going on, good, what's going on.
We define the customer logic in a forward method.
We first get a user to enter a name.
And we're starting a cumulative question
and in the for loop,
we'll keep generating a question
and keep asking user for the answer
and keep that in the record.
Until we reach the correct gas,
or we use up all the quotas,
when the process has started doing self-reflection,
and which maps to the log of this running process.
The demo is just a four-funk.
But I want to explain with this demo
how flexible it is to use DSPY module.
Basically, you can write an input function
inside this for a method.
We don't have adding restriction,
so it's easy to migrate to DSP
and migrate off DSPY.
And we can also see how the signature system
makes it easy to interact with our arm.
Because we have the signature system,
and explicitly for the output new question and gas made,
we don't need to worry about parsing the fields
out of the arm response.
And we don't need to worry about if the gas made
can indicate that like,
indicate if that's a generic answer,
general question,
or direct a name gas robustly.
The last thing I want to show with this lab
is how to save and load a DSPY module.
DSPY provides two ways of saving and loading.
The first way is state-only saving,
only saves internal state after DSPY module.
And to do that,
set a path to adjacent file
and set the flag of save program equal to false.
In order to load that back,
you need to recreate the instance,
or if you have an existing instance,
and call the load on your module to load that back.
DSPY also supports whole program saving
through cloud pickle.
You can worry about dependency
until you recreate the instance.
To do that,
give data directory as a path
and set the same program equal to two,
then call save.
To load that back,
use DSPY to load
and give the same path,
and it will be loaded as a new instance.
After you load back the program,
you can call that as if that's an original program,
and that will restart the game process.
That's all about this lesson.
You have learned how to program with DSPY.
In the next lesson,
you will use MLflow tracing
to debug your DSPY program.
See you there.

## 4. Debug Your DSPy Agent with MLflow Tracing
In this lesson, you'll learn about how to use MLflow tracing to help debug the DSP
program.
All right, let's go.
This lesson comes with a lab.
In a lab, you will go into build a DSP agent with the help of MLflow tracing.
As a demo, we'll build an airline customer service agent that helps users book and manage
the flights.
Let's dive in.
So what is tracing and why do you need tracing?
Tracing basically means recording the inputs and outputs of the intermediate function inside
your AI program.
And the capture is a hierarchical calling stack, like module A cos module B.
Generous applications can be very complex internally, while only the final output is exposed.
So when something is going off, it's hard to trace back to the root cause.
For example, if you have built a DSP program consisting of five cell modules, then if one
non-arm cause fails, due to not being able to understand the prompt, even though DSP
provides in Spark history to check out our own cause, it's hard to trace back.
Tracing provides an easy way for interpretability and debugging.
And what is MLflow?
MLflow is an open source AI ops package that streamlines your GNI app development.
MLflow helps with a full-life cycle of building GNI applications, ensuring that each
phase is traceable and reproducible.
Both MLflow server and clients have fully open source.
You can easily set it up.
To get started with tracing a DSP program with MLflow, you only need to add one line to
a program, which is MLflow.DSP.autolog, or simply MLflow.autolog.
After that, your program will be automatically traced and the trace will be saved in the
MLflow server that you can access anytime after being generated.
Let's talk about what is getting traced.
In MLflow, which has four things inside your DSP program, which has every module call,
no matter if that's a top module or internal modules, we also trace how the cause to the
adapter, from which you can see how the adapter formats the user query, and process the
alarm response.
We also trace the cause to the alarm so that you can inspect the actual prompt and the actual
alarm response.
DSP.autol, which is the wrapper over DSP.autolog calling, is also traced.
With MLflow tracing, not only we can interpret the input and output of some module of the
DSP program, we can also see the hierarchy of these modules, together with some other
important information like time consumption.
When something is going wrong, there will be a cross mark that points out to the program
module.
The screenshot in the slide is a real trace of a React module.
Let's map it to the components we trace.
React is our top-level module, which causes a few submodules represented as predict.
And we can see both React and the predicts are getting traced.
Inside the predictor call, the chat adapter format and parse carries the trace for the
adapter.
And the alarm trace in the middle has the actual prompt and response from BLM.
At the bottom, we can also see the trace of the tool calling, and in this case, tool
calling files, so there is a red mark next to that.
In the previous lesson, we have seen how to use DSP inspired history API to check out
the actual prompt.
With MLflow tracing, now it will be even easier.
Click on the alarm trace, and you can see the actual prompt and alarm response.
Let's get started with coding, similar to the previous lab, when to set up the API key.
And additionally, when to set up the MLflow environment, to import MLflow, we need to
pull that to the MLflow tracking UI.
In this lab, we have set up the MLflow tracking server for you, so you can get a tracking
UI and set it up with our helper functions.
And now let's give the experiment a unique identifier, just call it DSPY lesson 3.
And as we call in the slides, we can turn on the tracing feature, but one line, MLflow
dot DSP dot auto log.
Then we can choose how I am, as we did in the previous lab.
Now let's get started with building an airline customer service agent with DSPY React,
and use MLflow tracing to help with the process.
So this agent will be able to take user requests and book fly for the users, and can also modify
the artillery for the users.
We first need to define the data in real production, that will be a database schema.
It will have user profile, we have fly information, antenna or information, and customer service
support ticket.
Now we have the dummy data and data format.
Let's define a few tools in the module we use.
We need some of the faster fly information, based on date, origin, destination, with the
fashion and ternary, and we need to be able to pick the flight out of a few candidates.
And we need to be able to book an artillery on behalf of the user, and we need to be able
to cancel a ternary and pull user information, or fill a customer ticket if we cannot resolve
the issue automatically.
To define tools or functions with DSPY contacts, you need to specify a dog string to describe
what this function can do, and provide typings for input arcs so that the alarm can help
set the arguments.
Now we have the tools and data.
Let's define a signature so that we know what is input and output to expect from this
program.
The input will be a single string, representing a user request, and the output is a process
result, which is a message 18 tells the user.
If that's a successful booking, now we'll have the confirmation number, or like a number
to a ticket, if we cannot resolve the issue.
Now we have the signature and the tools, we can combine them into a DSPY or React.
So what is React?
React stands for reasoning and act.
So basically, we give the alarm the signature, which is the end goal of this program, along
with the list of tools, and the alarm can decide if that wants to call the tools to get
actual information and answer user questions.
If it doesn't need actual information, it can just answer the question.
Now we have built a React instance.
Let's invoke it.
We can simply put the user request into user request, can book a flight from a several
to JFK on a certain date, which we have a flight there, and tell the agent the name.
Let's run that.
Cool.
Now we'll get back the result, along with the MLflow tracing.
Here is the MLflow trace UI.
We can see what has been traced and the attributes of the trace.
So let's take a look at the attributes of the trace.
We capture the inputs and outputs, along with the attributes, which is the arguments to
a function call.
In the alarm, there will be the temperature, max token, and other configurable attributes.
The event tab will hold the error message if there is any error.
And we can see that the top module React is being traced, and the input output is our
end input and end output.
Can see that the inputs is our user query.
And the final output, the final representing as our output field process result, says,
has successfully booked a flight, and here is a confirmation number.
And we can look into some modules to see what happens in the process.
So when we talk to the alarm, we'll give that a user request and ask for the next
thought.
Could be a tool calling, could be the end of the process.
At the origin, they will just say, I want to fast-fly information, because I don't have
any information right now.
And it also determines arguments of the function calling to give the provided date we want
to fast-fly it.
After receiving that, we go ahead and call the tool, fast-fly information, which has the
inputs decided by the alarm, has a date, origin, destination, and then we have faster list
of flights matching the request.
We get two out there.
When we grab all the information, like the two candidate flights, along with the tool
calling information, send them back to the alarm to decide the next tool calling, or we
can wrap up the process.
So we represent all the things in the trajectory field.
We have the last tool calling arguments, and the result.
And the alarm says, okay, I have a bunch of flights, we need to pick the best flight.
If the argument has a flight candidates, and we call it to over-pick the flight, we have
pretty fine logic for this paid flight.
We always pick the shortest flight, if that's the same duration, we pick the cheapest one.
So here we get the flights information and inputs, and we pick a flight heart out of that.
And we grab all the information to call the alarm again.
The alarm, this time, see, okay, we have the flights, and we can pull the user information
so that I can book the flight on behalf of the user.
Then we get the user information, and send all the things to the alarm again.
And next, the thought will be, have all the information, I can just call a book of flight.
Then we call book the flight.
Then we write things into a database, and we recall the alarm.
And this time, the alarm says, okay, this seems to be complete.
So we can call the finish with a dummy tool in DSP, marking the end of the reactor tool
calling.
And after all the process, we still need to find a way to fulfill the process result,
which is output filled.
So we just call a channel of that, give that all the tool calling history, and the user
request to form the final output.
Then after that, we wrap up the process.
We can see clearly with MLflow tracing, how we interpret what's happening inside
a reactor module with a very complex multi-help calling.
And if there's anything going wrong, we can click on that certain module, and
find the input output to decide how we debug.
By the way, MLflow is also integrated with a LAN chain, Lama index, and other frameworks.
You can get the auto tracing feature with other frameworks as well.
In the lab, we set up an MLflow server for you, and in actual development, you will
need to do it by yourself.
If you don't have time to set up your own MLflow server, or want to explore more features,
Databricks provides managed MLflow service, and you can simply connect it to it to
get started.
You can try now through Databricks Lighthouse, which provides free trial.
Give it a try, and sign up at dot databricks.com.
In this lesson, you have used MLflow tracing to build a complex reactor module for a airline
customer service.
In the next lesson, you will learn how to use DSP Optimizer to automatically optimize
the program's quality.
See you there.

## 5. Optimizing Agents with DSPy Optimizer
In this lesson, you will learn about how to use the DSP optimizer to automatically improve your DSP program's quality.
Let's get coding.
This lesson comes with a lab.
In a lab, you will get a hands-on experience of using DSP optimizer to automatically improve the quality of an energetic rock,
which uses Wikipedia as a data source.
After the optimization, you will see our rock has a big quality boost.
Let's dive in.
Before talking about DSP optimizer, let's think about what optimization means when we talk about generate applications.
Mostly, it can be three things.
It can mean optimizing a prompt template could also mean building high-quality future examples.
These two parts both belong to the concept of prompt optimization or prompt engineering.
Optimization could also mean fine-tune LOM weights in DSP with support all of the three parts.
Now let's take an overview at how to use the DSP optimizer.
First, you need to pick an optimizer.
For how to pick the optimizer, please refer to our documentation site at DSP.AI.
For this lesson and lab, we will use the MiPRO video optimizer,
which is a good optimizer for prompt template optimization and building future examples.
After you pick an optimizer, you need to tell the optimizer the evaluation metric function
and also the training and the validation data set.
The core idea of the requirement is the optimizer has to know what is a good program and what is a bad program.
Different from normal machine learning job, the data set can be as small as 20 records.
If you don't have a validation data set, the optimizer will split part of the data to be the validation data set.
Let's talk about how optimizers work.
In this lesson, we'll focus on prompt engineering and demonstrate by MiPRO video optimizer.
For how to do fine training with DSP optimizer and how other optimizers work,
please read on our documentation page at DSP.AI slash tutorials.
So from a very high level, we first build multiple sets of future examples.
We will talk about how to build that in the next slide.
Then based on the future examples and your program information,
we let our arm generate multiple prompt template candidates,
which map to the instruction of the signature if you use a class-based signature.
If you use a class-based signature, that means the dog string of it.
Then we sample from both directions the future example set and the instruction set
to form the candidate program and run evaluations on the candidate program.
The evaluation is based on a metric function and validation data set.
We pick data from validation data set and run against the program,
compare the program output against the golden labels using a user defined metric function.
The final program score is the average over all pick data.
Along the process, we continuously pick the candidate of the highest score.
And importantly, in our optimizer, when neither do you prove for search,
nor try all the combos, instead, we use a statistical way called Bayesian sample length
to intelligently sample towards optimal combo.
Now we have seen optimization flow at a high level.
Let's talk about how we generate the future examples and the instruction candidates.
Future examples are generated through the process couple-strapping, which is very simple.
We grab data from a training data set
and fit that into a DSP program, which can be one module or multiple modules.
Then, if according to the metric function, that's a score over the threshold set by the user.
We grab the trace, which is the input and output of its module,
to make that as a future example candidate for that module.
Please note that one data can generate multiple traces
because we are randomness to each call via certain temperature to non-zero.
Now let's talk about how we build the instruction candidates.
We grab the program code and description along with the future examples
and some arbitrary tips like being comprehensive or being concise
and send all of them to the LOM through something called DSP proposal
and generate a bunch of instruction candidates.
Now we have the candidates of both instruction, which is prompt template,
and future examples.
We can start generating candidate programs by picking one future example candidate
and one instruction candidate and combine them.
And then we can evaluate the candidate program
and keep simply for the number of trials user-specified.
We have done thorough experiments on the performance of the MiPROVITY optimizer.
We see that MiPROVITY outperforms the original prompt by large margin in multiple tasks.
For more details, please check out the paper,
optimizing instructions and demonstrations for multi-stage language model programs.
We can also utilize MLflow to interpret optimization process.
Simply turn on the autolog by setting a few more flags to two in MLflow.dsPilot autolog,
then the optimization process will be tracked.
As we said, when we try the candidate program, we do evaluation on that
and all these evaluations along with the candidate program's information
like what's the instruction, what's the future example, or be saved to MLflow
so get a full track of what's been tried across the process.
Now let's get some coding to see how optimizer works.
As we did in previous labs, we do set up a API key
and we need to set up a MLflow truck and server.
And we need to give data a unique identifier.
Let's call that DSPiCourse2.
And we can set up the autolog in and turn on a few more flags than before
so that we can track the optimization process.
And we need to specify the MLflow, let's continue using the GPT-4L meaning.
For this ROG agent, it will be based on Wikipedia data
and then we use an agentic ROG instead of like fixed a number of knowledge source code.
Agentic ROG basically means we'll let the MLflow decide if we still need more data
from knowledge source before we get to the final answer.
Let's first define our tool, which is such a Wikipedia.
We use co-birth V2, which is public available interface for quantum Wikipedia data.
And the return value will be the chunk faster from Wikipedia data source.
And we still use the DSPiL React as our program.
And this time, the input output is very simple, just question answer.
So we use a stream-based signature and has only one tool is a search Wikipedia.
Now let's recap what we need to have to optimize it wrong.
We need to have data set, training data set, and validation data set.
So, let's load data set.
We have prepared the data set for you, which is the subset of the hop-up QA data set,
which is the question and answer in data set based on Wikipedia data.
After loading the data, let's take a look at what the data looks like.
So it's very simple.
It has a question, has an answer, and input case will be the question.
Okay, it's time to create our optimizer.
We'll use a MiproV2 optimizer.
And, as we said, we need to prepare a metric function so that the optimizer knows
how to evaluate our candidate program.
And for this task, we'll just use the answer exact match.
And to configure the optimizer, we'll recommend you use the auto mode,
which has three modes, light, medium, and heavy, which are carefully trained for good performance.
But if you want to customize a MiproV2 optimizer, you can find our available options on our documentation side
and in the search box search for DSPID on MiproV2.
Now you will go and find all the available configurations.
The optimization process can take quite a while.
So in order to speed up the process, we have recorded the cache for you.
So the optimization will hit the cache to be much faster.
In real production, you don't need to require any cache.
Just make our own calls.
Now, let's kick off the optimization process.
We use the optimizer.compile function and send the program to it,
specify a training set and validation set.
All right, the optimization is done.
We can take a look at the login.
So basically, we're continuously getting a candidate program
and run evaluation over that.
And we finally picked the basic program out of the process.
Let's take a look of what is getting changed along with the process.
So if you remember the original signature,
it's just a very simple input to a question to answer
and the react module will not have any instructions.
But after the optimization process,
the reaction module will have a very comprehensive instruction populated.
And to also have a few future examples built into that,
represented as a list in the demo attributes.
Let's first evaluate the non-optimizer react, our RIG application.
We get a score of 31,
and we can see some example input and output gear in the table.
Cool, let's now evaluate the optimizer react and get our score.
Okay, the score of 54.
You can see that without adding human interaction
just by simply using the optimizer,
we get a score boost from 31 to 54.
That is the power of DSP optimizer.
As we mentioned in slides,
we track the optimization process with MLflow
and you can view that in the MLflow UI.
Go into the UI.
The optimization run shows up as a natural run.
And each child maps to a evaluation of the candidate program.
Click into the run.
You can see the attributes of the candidate program,
like future examples, instruction,
along with other attributes,
and the evaluation score of the candidate.
So you keep track of the full record of the optimization process.
In this lesson, you'll have learned how to use DSP optimizer
to optimize your DSP program.
And we have seen how powerful the DSP optimizer is
through optimizing the ROG application with Wikipedia data.

## 6. Conclusion
Congratulations on completing this course.
In this course, you'll learn that DSP is a lightweight,
flexible, GNI and authoring framework that
simplifies interaction with our arms and aging development.
It also provides automatic program optimization through DSP optimizers,
and has native MLflow tracing integration for easy development.
You use the DSPIDAL signature to define inputs and outputs,
and DSPIDAL module to wrap the customer logic.
After building a program, you'll learn how to prepare a small data set,
and a metric function to use the DSP optimizer
to improve the quality of the AI program.
And finally, you use the MLflow tracing
by adding MLflow.DSPID.autolog.
I'm looking forward to seeing what you have built on your own.
