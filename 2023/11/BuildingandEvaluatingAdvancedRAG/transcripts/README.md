# Building and Evaluating Advanced RAG

## 01. Introduction
Retrieval Augmented Generation or RAG has become a key method for getting LMS answered questions
over a user's own data.
But to actually build and productionize a high quality RAG system, it costs a lot to have
effective retrieval techniques to give the LMS highly relevant context to generate this
answer, and also to have an effective evaluation framework to help you efficiently iterate and
improve your RAG system both through initial development and during post-deportment maintenance.
This course covers two advanced retrieval methods, sentence window retrieval and auto-emerging
retrieval that deliver a significantly better context to the LMS than simpler methods.
It also covers how to evaluate your LMS question answering system with three evaluation metrics,
contact relevance, drowdenness, and answer relevance.
I'm excited to introduce Jerry Liu, co-founder and CEO of Larmatex, and Anupang Data, co-founder
and chief scientist of Truera.
For a long time, I've enjoyed following Jerry and Larmate index on social media and getting
tips on evolving RAG practices, so look forward to him teaching this body of knowledge
more systematically here.
Add Anupang has been a professor at CMU and has done research for over a decade on trustworthy
AI and how to monitor evaluates and optimize AIF effectiveness.
Thanks Andrew, it's great to be here.
Great to be with you Andrew.
Sentence window retrieval gives an LLN better context by retrieving not just the most relevant
sentence, but the window of sentences that occur before and after it in the document.
Auto-emerging retrieval organizes the document into a tree-like structure where each parent
notes text is divided among its child notes.
When none of the child notes are identified as relevant to a user's question, then the
entire text of the parent note is provided as context for the outline.
I know this sounds like a lot of steps, but don't worry, we'll go over it in detail
on code later, but the main takeaway is that this provides a way to dynamically retrieve
or coherent chunks of text than simpler methods.
To evaluate RAG-based LLM apps, the RAG Triad, a triad of metrics for the three main steps
of a RAG's execution, is quite effective.
For example, we'll cover in detail how to compute context relevance, which measures how
relevant the retrieved chunks of text are to the user's question.
This helps you identify and debug possible issues with how your system is retrieving
context for the LLM in the QA system.
But that's only part of the overall QA system.
We'll also cover additional evaluation metrics such as groundedness and answer relevance
that let you systematically analyze what parts of your system are or are not yet working
well, so that you can go in in a targeted way to improve whatever part needs the most
work.
If you're familiar with the concept of error analysis and machine learning, this has
similarities, and I've found that taking this sort of systematic approach helps you be
much more efficient in building a reliable QA system.
The goal of this course is to help you build production ready RAG-based LLM apps.
And important parts of getting production ready is to iterate in a systematic way on
the system.
In the later half of this course, you gain hands-on practice iterating using these retrieval
methods and evaluation methods, and you also see how to use systematic experiment tracking
to establish a baseline and then quickly improve on that.
We'll also share some suggestions for tuning these two retrieval methods based on our
experience assisting partners who are building RAG apps.
Many people have worked to create this course.
I'd like to thank on the LLM index side, Logan Makowicz, and on the true eraside, Shaixen,
Joshua Rainey, and Barbara Lewis.
The next lesson will give you an overview of what you'll see in the rest of the course.
You'll try out question answering systems that use sentence window retrieval or auto-merging
retrieval and compare their performance on the RAG triad.
Contacts relevance, groundedness, and answer relevance.
Sounds great.
Let's get started, and I think you'd be really clean up with this RAG stuff.

## 02. Advanced RAG Pipeline
In this lesson, you'll get a full overview of how to set up both a basic and advanced
rag pipeline with Lama Index.
We'll load in an evaluation benchmark and use true lengths to find a set of metrics so
that we can benchmark advanced rag techniques against the baseline or basic pipeline.
In the next few lessons, we'll explore each lesson a little bit more in depth.
Let's first walk through how a basic retrieval augment at generation pipeline works, or rag
pipeline.
Incess of three different components, ingestion, retrieval, and synthesis.
Going through the ingestion phase, we first load in a set of documents.
For each document, we split it into a set of text shrunks using a text splitter.
Then for each chunk, we generate an embedding for that chunk, using an embedding model.
And then, for each chunk within embedding, we offload it to an index, which is a view
of a storage system such as a MACDAR database.
Once the data is stored within an index, we then perform retrieval against that index.
First, we launch a user query against the index, and then we fetch the top came with similar
chunks to the user query.
Afterwards, we take these relevant chunks, combine it with the user query, and put it into
the prompt window of the ALM in the synthesis phase.
And this allows us to generate a final response.
This new book will walk you through how to set up a basic and advanced rag pipeline with
Lama Index.
We will also use true error to help set up an evaluation benchmark so that we can measure
improvement against the baseline.
For this quick start, you will need an OpenAI API team.
Note that for this lesson, we'll use a set of helper functions to get you SAP and running
quickly, and we'll do a deep dive into some of these sections in the future lessons.
Next we'll create a simple ALM application using Lama Index, which internally uses an
OpenAI ALM.
In terms of the data source, we'll use the how to build a career in AI PDF written by
Android.
Note that you can also upload your own PDF file, if you wish.
And for this lesson, we encourage you to do so.
Let's do some basic sanity tracking of what the document consists of, as well as the
length of the document.
We see that we have a list of documents.
There's 41 elements in there.
Each item of that list is a document object.
And we'll also show a snippet of the text for a given document.
Next we'll merge these into a single document, because it helps with overall text blending
accuracy when using more advanced retrieval methods, such as the sentence window retrieval,
as well as an on-emerging retrieval.
The next step here is to index these documents, and we can do this with the vector store index
within Lama Index.
Next we define a service context object, which contains both the ALM.
We're going to use, as well as the embedding model we're going to use.
The ALM we're going to use is GPT 3.5 Turbo from OpenAI.
And then the embedding model that we're going to use is the hugging phase BGUD small
model.
These few steps show this ingestion process right here.
We've loaded in documents, and then in one line, vector store index off from documents,
we're doing the chunking, embedding, and indexing under the headed with the embedding model
that you specified.
Next we obtain a query engine from this index that allows us to send user queries that
do retrieval in synthesis against this data.
Let's try out our first request.
And the query is, what are steps to take when finding projects to build your experience?
Let's find out.
Store small and gradually increase the scope and complexity of your projects.
Great, so it's working.
So now we've set up the basic drag pipeline.
The next step is to set up some evaluations against this pipeline to understand how well
it performs, and this will also provide the basis for defining our advanced retrieval methods
of a sentence window retriever, as well as a auto merging retriever.
In this section, we use true lines to initialize feedback functions.
We initialize a helper function, get feedbacks to return a list of feedback functions to
evaluate our app.
Here we've created a rag evaluation triad, which consists of pairwise comparisons between
the query, response, and context.
And so this really creates three different evaluation models.
Answer relevance, context relevance, and groundedness.
Answer relevance is the response relevant to the query.
Context relevance is the retrieved context relevant to the query.
And groundedness is the response supported by the context.
We'll walk through how to set this up yourself in the next few notebooks.
The first thing we need to do is to create a set of questions on which to test our application.
Here we've pre-written the first 10, and we encourage you to add to the list.
And now we have some evaluation questions.
What are the keys to building a query in AOT?
How can team work contribute to success in AOT, et cetera?
The first thing we need to do is to create a set of questions on which to test our application.
Here we've pre-written the first 10, but we encourage you to also add to this list.
Here we specify a fun new question.
What is the right AI job for me?
We add it to the Eval questions list.
Now we can initialize the trulence modules to begin our evaluation process.
We've initialized the trulence module, and now we've reset the database.
We can now initialize our evaluation modules.
The trulence are growing as a standard mechanism for evaluating generative AI applications on scale.
Rather than relying on expensive human evaluation or set benchmarks,
all of us allow us to evaluate our applications in a way that is custom to the domain in which
we operate, in dynamic to the changing demands for our application.
Here we've pre-built a shulence recorder to use for this example.
In the recorder, we've included the standard triad of evaluations for evaluating rags.
In this context of relevance, an answer relevance.
We'll also specify an ID so that we can track this version of our app.
As we experiment, we can track new versions by simply changing the app ID.
Now we can run the query engine again with the trulence context.
So what's happening here is that we're sending each query to our query engine.
And in the background, the trulence recorder is evaluating each of our queries against these three metrics.
If you see some morning messages, I don't worry about it.
Some of it is system-to-end it.
Here we can see a list of queries as well as our associated responses.
You can see the input, output, the record ID, tags, and more.
You can also see the answer relevance, context relevance, and ground-inness for each rub.
In this dashboard, you can see your evaluation metrics like context relevance, answer relevance, and ground-inness,
as well as average latency, total cost, and more, and then a UI.
Here we see that the answer relevance and ground-inness are decently high,
but context relevance is pretty low.
Now let's see if we can improve these metrics.
With more advanced retrieval techniques, like sentence window retrieval,
as well as on-emergent retrieval.
The first advanced technique we'll talk about is sentence window retrieval.
This works by embedding and retrieving single sentences to more granular chunks.
But after retrieval, the sentences are replaced with a larger window of sentences
around the original retrieved sentence.
The intuition is that this allows for the LLAB to have more context for the information retrieved
in order to better answer queries while still retrieving on more granular pieces of information,
so ideally improving both retrieval as well as synthesis performance.
Now let's take a look at how to set it up.
First, we'll use OpenAI GPT 3.5 Turbo.
Next, we'll construct our sentence window index over the given document.
Just a reminder that we have a helper function for constructing the sentence window index,
and we'll do a deep dive in how this works under the hood in the next few lessons.
Similar to before, we'll get a query engine from the sentence window index.
And now that we've set this up, we can try running an example query.
Here, the question is, how do I get started on a personal project in AI?
And we get back a response.
The guest started on a personal project in AI.
It is first important to identify a scope of the project.
Great.
Similarly to before, let's try getting the truelines evaluation context
and try benchmarking the results.
So here, we import the true recorder sentence window,
which is a prebuilt truelines recorder for the sentence window index.
And now, we'll run the sentence window retriever on top of these evaluation questions,
and then compare performance on the rag triad of evaluation modules.
Here, we can see the responses come in as they're being run.
Some examples of questions or responses, how can teamwork contribute to success in AI?
Teamwork can contribute to success in AI by allowing individuals leverage the expertise
and insights of their colleagues.
What's the importance of networking in AI?
Now, looking is important in AI because it allows individuals to connect with others
who have experience and knowledge in the field.
Great.
Now that we've run evaluations for two techniques,
the basic rag pipeline as well as the sentence window retrieval pipeline,
let's get a leader word of the results and see what's going on.
Here, we see that general only groundedness is
8% of points better than the baseline rag pipeline.
Answer relevance is more or less the same.
Context relevance is also better for the sentence window prairie engine.
Latency is more or less the same and the total cost is lower.
Since the groundedness and context relevance are higher by the total cost is lower,
we can intuit that the sentence window retriever is actually giving us more relevant
context and more efficiently as well.
When we go back into the UI, we can see that we now have a comparison between the
direct query engine and the baseline as well as the sentence window.
And we can see the metrics that we just saw in the notebook
displayed in the UI as well.
The next advanced retrieval technique we'll talk about is the auto merging retriever.
Here, we construct a hierarchy of larger parent nodes with smaller child nodes that reference
the parent node. So for instance, we might have a parent node of chunk size 512 tokens.
And underneath, there are four child nodes of chunk size 128 tokens that link to this parent node.
The auto merging retriever works by merging retrieve nodes into larger parent nodes,
which means that during retrieval, if a parent actually has the majority of its children
nodes retrieved, then we'll replace the children nodes with the parent node.
So this allows us to hierarchically merge our tree nodes.
The combination of all the child nodes is the same text as the parent node.
Similarly to the sentence window retriever and the next few lessons,
we'll do a bit more of a deep dive on how it works.
Here, we'll show you how to set it up with our helper functions.
Here, we've built the auto merging index.
I'm using GBT 3.5 turbo for the Allen as well as the BGE model for the embedding model.
We get the query engine from the auto merging retriever.
And let's try running an example query.
How do I build a portfolio of the AI projects?
In the logs here, you actually see the merging process go on,
or merging nodes into a parent node to basically retrieve the parent node as opposed to the child node.
To build a portfolio of AI projects, it is important to start with simple underkinkings
and gradually progress some more complex ones.
Great, so we see that it's working.
Now let's benchmark results with trulence.
We get a pre-built trulence recorder on top of our auto merging retriever.
We then run the auto merging retriever with trulence on top of our evaluation questions.
Here, for each question, we actually see the merging process going on,
such as merging three nodes into the parent node for the first question.
If we swole down just a little bit, we see that for some of these other questions,
we're also performing the merging process.
Merging three nodes into the parent node, merging one node into the parent node.
An example, a question response pair is, what is the importance of networking in AI?
Networking is important in AI because it helps in building a strong,
professional networking community.
Now that we've run all three retrieval techniques,
the basic rag pipeline, as well as the two advanced retrieval methods,
we can view a comprehensive leaderboard to see how all three techniques stack up.
We get pretty nice results for the auto merging query engine.
On top of the evaluation questions, we get 100% in terms of ground in this.
94% in terms of answer relevance.
43% in terms of context relevance, which is higher than both the sentence window and the baseline
rag pipeline. And we get roughly equivalent total costs to a sentence window query engine,
implying that the retrieval here is more efficient with equivalent latency.
And at the end, you can view this in the dashboard as well.
This lesson gives you a comprehensive overview of how to set up a basic and advanced rag pipeline,
and also how to set up evaluation modules to measure performance.
In the next lesson, on a problem, we'll do a deep dive into these evaluation modules,
specifically the rag triad of grab in this, answer relevance, and context relevance.
And you'll learn a bit more about how to use these modules and what each module needs.

## 03. RAG Triad of metrics
In this lesson, we do a deep dive into evaluation.
We'll walk you through some core concepts on how to evaluate RAG systems.
Specifically, we will introduce the RAG Triad, a triad of metrics for the three main steps
of a RAG's execution, context relevance, groundedness, and answer relevance.
These are examples of an extensible framework of feedback issues,
programmatic evaluations of LLM apps.
We then show you how to synthetically generate an evaluation data set,
given any unstructured corpus.
Let's get started.
Now I'll use a notebook to walk you through the RAG triad,
answer relevance, context relevance, and groundedness,
to understand how each can be used with trulence to detect hallucinations.
At this point, you have already peep installed trulence eval and LLM indexed.
So I'll not show you that step.
The first step for you will be to set up an OpenAI API key.
The OpenAI key is used for the completion step of the RAG
and to implement the evaluations with trulence.
So here's a code snippet that does exactly that.
And you're now all set up with the OpenAI key.
The next section, I will quickly recap the query engine construction with LLM index.
Jerry has already walked you through that in lesson one.
In some detail, we will largely build on that lesson.
The first step now is to set up a true object.
From trulence eval, we are going to import the true class.
Then we'll set up a true object and instance of this class.
And then this object will be used to reset the database.
This database will be used later on to record the prompts, responses, intermediate results
of the LLM index app, as well as the results of the various evaluations.
We will be setting up with trulence.
Now let's set up the LLM index reader.
So this snippet of code reads this PDF document from a directory
on how to build a career in AI written by Andrew Eng
and then loads this data into this document object.
The next step is to merge all of this content into a single large document
rather than having one document bar page, which is the default setup.
Next, we set up the sentence index leveraging some of the LLM index utilities.
So you can see here that we are using open AI GPT 3.5 turbo, set at a temperature of 0.1
as the LLM that will be used for completion of the rag.
The embedding model is set to BGE small n version 1.5.
And all of this content is being indexed with the sentence index object.
Next, we set up the sentence window engine.
And this is the query engine that will be used later on to do retrieval effectively
from this advanced rag application.
Now that we have set up the query engine for sentence window-based rag,
let's see it in action by actually asking a specific question.
How do you create your AI portfolio?
This will return a full object with the final response from the LLM,
the intermediate pieces of retrieved context, as well as some additional metadata.
Let's take a look at what the final response looks like.
So here you can see the final response that came out of this sentence window-based rag.
It provides a pretty good answer on the surface to this question of how do you create your AI portfolio.
Later on, we will see how to evaluate answers of this form against the rag triad
to build confidence and identify failure modes for rags of this form.
Now that we have an example of a response to this question that looks quite good on the surface,
we will see how to make use of feedback functions, such as the rag triad,
to evaluate this kind of response more deeply, identify failure modes,
as well as build confidence or iterate to improve the LLM application.
Now that we have set up the sentence window-based rag application,
let's see how we can evaluate it with the rag triad.
We'll do a little bit of housekeeping in the beginning.
First step is this hiss of code snippet that lets us launch a stream lid dashboard from inside
the notebook. You'll see later that we'll make use of that dashboard to see the results of the
evaluation and to run experiments, to look at different choices of apps, and to see which one
is doing better. Next up, we initialize OpenAI and GPT 3.5 turbo as the default provider
for our evaluations, and this provider will be used to implement the different feedback functions
or evaluations, such as context relevance, sponsor relevance, and groundedness.
Now let's go deeper into each of the evaluations of the rag triad, and we'll go back and forth a bit
between slides and the notebook to give you the full context. First up, we'll discuss answer
relevance. Recall that answer relevance is checking whether the final response is relevant
to the query that was asked by the user. To give you a concrete example of what the output of
answer relevance might look like, here's an example. The user asked the question,
how can altruism be beneficial in building a career? This was the response that came out of the
rag application, and the answer relevance evaluation produces two pieces of output. One is a score
on a scale of 0 to 1. The answer was assessed to be highly relevant, so it got a score of 0.9.
The second piece is the supporting evidence or the rationale or the chain of thought reasoning behind
why the evaluation produced this score. So here you can see that supporting evidence found in
the answer itself, which indicates to the LLAM evaluation that it is a meaningful and relevant
answer. I also want to use this opportunity to introduce the abstraction of a feedback function.
Answer relevance is a concrete example of a feedback function. More generally, a feedback function
provides a score on a scale of 0 to 1 after reviewing an LLAM app's inputs, outputs and intermediate results.
Let's now look at the structure of feedback functions using the answer relevance feedback function as a
concrete example. The first component is a provider, and in this case, we can see that we are using
an LLAM from OpenAI to implement these feedback functions. Note that feedback functions don't have
to be implemented. Necessarily using LLAMs, we can also use BERT models and other kinds of mechanisms
to implement feedback functions that I'll talk about in some more detail later in the lesson.
The second component is that leveraging that provider will implement a feedback function.
In this case, that's the relevance feedback function. We give it a name, a human readable name
that'll be shown later in our evaluation dashboard. And for this particular feedback function,
we run it on the user input, the user query, and it also takes as input the final output or response
from the app. So given the user question and the final answer from the rag, this feedback function will
make use of an LLAM provider, such as OpenAI, GPT 3.5, to come up with a score for how relevant
the responses to the question that was asked. And in addition, it'll also provide supporting
evidence or chain of thought reasoning for the justification of that score. Let's now switch back
to the notebook and look at the code in some more detail. Now let's see how to define the question
answer relevance feedback function in code. From TrueLensival, we will import the feedback class.
Then we set up the different pieces of the question answer relevance function that we were just
discussing. First up, we have the provider that is OpenAI, GPT 3.5, and we set up this
particular feedback function where the relevant score will also be augmented with the chain of
thought reasoning, much like I showed in the slides. We give this feedback function a human
understandable name. We call it answer relevance. This will be show up later in the dashboard,
making it easy for users to understand what the feedback function is setting up.
Then we also will give the feedback function access to the input, that is the prompt,
and the output, which is the final response coming out of the drag application.
With this setup, later on in the notebook, we will see how to apply this feedback function
on a set of records, get the evaluation scores for answer relevance as well as the chain of thought
reasons for why, for that particular answer, that was the judged score to be appropriate for
as part of the evaluation. The next feedback function that we will go deep into is context relevance.
Recall that context relevance is checking how good the retrieval processes,
that is given a query, we will look at each piece of retrieved context from the vector database
and assess how relevant that piece of context is to the question that was asked.
Let's look at a simple example. The question here or the prompt from the user is how can
altruism be beneficial in building a career? These are the two pieces of retrieve context
and after the evaluation with context relevance, each of these pieces of retrieve context
gets a score between 0 and 1. You can see here the left context got a relevant score of 0.5,
the right context got a relevant score of 0.7, so it was assessed to be more relevant to this
particular query, and then the mean context relevance score is the average of the relevant scores
of each of these retrieved pieces of context that gets also reported out.
Let's now look at the structure of the feedback function for context relevance.
Various pieces of this structure are similar to the structure for
answer relevance, which we reviewed a few minutes ago. There is a provider, let's open AI,
and the feedback function makes use of that provider to implement the context relevance feedback
function. The differences are in the inputs to this particular feedback function. In addition to
the user input or prompt, we also share with this feedback function a pointer to the retrieve
context, that is, the intermediate results in the execution of the rag application.
We get back a score for each of the retrieved pieces of context,
assessing how relevant or good that context is with respect to the query that was asked,
and then we aggregate and average those scores across all the retrieve pieces of context
to get the final score. Now you will notice that in the answer relevance feedback function,
we had only made use of the original input, the prompt, and the final response from the rag.
In this feedback function, we are making use of the input or prompt from the user,
as well as intermediate results, the set of retrieve contexts to assess the quality of the
retrieval. Between these two examples, the full power of feedback functions is leveraged by making
use of inputs, outputs, and intermediate results of a rag application to assess its quality.
Now that we have the context selection set up, we are in a position to define the
context relevance feedback function in code. You'll see that it's pretty much the code segment
that I walked through on the slide. We're still using OpenAI as the provider, GPT 3.5 as the
evaluation LLAM. We are calling the question statement or context relevance feedback function.
It gets the input prompt, the set of retrieved pieces of context. It runs the evaluation function on
each of those retrieve pieces of context separately, gets a score for each of them, and then
averages them to report a final aggregate score. Now one additional variant that you can also use,
if you like, is in addition to reporting a context relevance score for each piece of retrieve
context, you can also augment it with chain of thought reasoning so that the evaluation LLAM
provides not only a score, but also a justification or explanation for its assessment score,
and that can be done with QS relevance with chain of thought reasoning method.
And if I give you a concrete example of this in action, you can see here's the question,
are the user prompt, how can altruism be beneficial in building a career?
This is an example of a retrieved piece of context that takes out a chunk from Andrew's article
on this topic. You can see the context relevance feedback function gives a score of 0.7 on a
scale of 0 to 1 to this piece of retrieved context. And because we have also
invoked the chain of thought reasoning on the evaluation LLAM, it provides this justification
for why the score is 0.7. Let me now show you the code snippet to set up the groundedness feedback
function. We kick it off in much the same way as the previous feedback functions,
leveraging LLAM provider for evaluation, which is if you recall open AI, GPT 3.5.
Then we define the groundedness feedback function. This definition is structurally
very similar to the definition for context relevance. The groundedness measure comes with
chain of thought reasons, justifying the scores, much like I discussed on the slides.
We give it the name groundedness, which is easy to understand. And it gets access to
the set of retrieved contexts in the rag application, much like for context relevance,
as well as the final output or response from the rag. And then each sentence in the final response
gets a groundedness score. And those are aggregated, averaged to produce the final groundedness score
for the full response. The context selection here is the same context selection that was used for
setting up the context relevance feedback function. So if you recall that just gets the set of
retrieved pieces of context from the retrieval step of the rag and then can access each node within
that list, recover the text of the context from that node and proceed to work with that to do
the context relevance, as well as the groundedness evaluation. With that, we are now in a position
to start executing the evaluation of the rag application. We have set up all three
feedback functions, answer relevance, context relevance and groundedness, and all we need
is an evaluation set on which we can run the application and the evaluations and see how they're
doing and if there are opportunities to iterate and improve them further. Let's now look at the workflow
to evaluate and iterate to improve LLM applications. We will start with the basic LLM index rag
that we introduced in the previous lesson and which we have already evaluated with the true
lens rag triad. We'll focus a bit on the failure modes related to the context size.
Then we will iterate on that basic rag with an advanced rag technique, the LLM index sentence
window rag. Next we will re-evaluate this new advanced rag with the true lens rag triad,
focusing on these kinds of questions. Do we see improvements specifically in context relevance?
What about the other metrics? The reason we focus on context relevance is that often
failure modes arise because the context is too small. Once you increase the context up to a certain
point, you might see improvements in context relevance. In addition, when context relevance goes up,
often we find improvements in groundedness as well because the LLM in the completion step
has enough relevant context to produce the summary. When it does not have enough relevant context,
it tends to leverage its own internal knowledge from the pre-training dataset to try to fill those
gaps, which results in a loss of groundedness. Finally, we will experiment with different window
sizes to figure out what window size results in the best evaluation metrics. Recall that
if the window size is too small, there may not be enough relevant context to get a good
score on context relevance and groundedness. If the window size becomes too big on the other hand,
irrelevant context can creep into the final response, resulting in not such great scores in
groundedness or answer relevance. We walk through three examples of evaluations or feedback functions,
context relevance, answer relevance and groundedness. In our notebook, all three were implemented
with LLM evaluations. I do want to point out that feedback functions can be implemented in different
ways. Often we see practitioners starting out with ground truth evils, which can be
expensive to collect, but nevertheless a good starting point. We also see people
leverage humans to do evaluations. That's also helpful and meaningful, but hard to scale in practice.
Ground truth evils just to give you a concrete example. Think of a summarization use case where
there's a large passage and then the LLM produces a summary. A human expert would then give that
summary a score indicating how good it is. This can be used for other kinds of use cases as well,
such as chatbot like use cases or even classification use cases. Human evils are
similar in some ways to ground truth evils in that as the LLM produces an output or a rag application
produces an output, the human users of that application are going to provide a rating for that
output, how good it is. The difference with ground truth evils is that these human users may not be
as much of an expert in the topic as the ones who produce the curated ground truth evils.
It's nevertheless a very meaningful evaluation. It'll scale a bit better than the ground truth evils,
but our degree of confidence than it is lower. One very interesting result from the research
literature is that if you ask a set of humans to rate a question, there's about 80% agreement,
and interestingly enough when you use LLM for evaluation, the agreement between the LLM
evaluation and the human evaluation is also about the 80-25% mark. So that suggests that LLM
evaluations are quite comparable to human evaluations for the benchmark, data sets to which they have
been applied. So feedback functions provide us a way to scale up evaluations in a programmatic manner.
In addition to the LLM evils that you have seen, feedback functions also provide can implement
traditional NLP metrics such as ROOSH scores and blue scores. They can be helpful in certain scenarios,
but one weakness that they have is that they are quite syntactic. They look for overlap between
words in two pieces of text. So for example, if you have one piece of text that's referring to a
river bank and the other to a financial bank, syntactically they might be viewed as similar and
these references might end up being viewed as similar references by a traditional NLP evaluation,
whereas the surrounding context will get used to provide a more meaningful evaluation when you're
using either large language models such as GPT-4 or medium-sized language models such as bird
models and to perform your evaluation. While in the course we have given you three examples of
feedback functions and evaluations, answer relevance, context relevance, and groundedness.
TrueLens provides a much broader set of evaluations to ensure that the apps that you're building
are honest, harmless, and helpful. These are all available in the open source library and we encourage
you to play with them as you are working through the course and building your LLM applications.
Now that we have set up all the feedback functions, we can set up an object to start recording,
which will be used to record the execution of the application on various records.
So you'll see here that we are importing the TrueLama class, creating an object to recorder of
this TrueLama class. This is our integration of TrueLens with LLM index. It takes in the sentence
window engine from LLM index that we had created earlier, sets the app ID and makes use of the three
feedback functions of the RagTriad that we created earlier. This TrueRecorder object will be used
in a little bit to run the LLM index application as well as the evaluation of these feedback
functions and to record it all in a local database.
Let us now load some evaluation questions. In this setup, the evaluation questions are
set up already in this text file and then we just execute this code snippet to load them in.
Let's take a quick look at these questions that we will use for evaluation.
You can see what are the keys to building a career in AI and so on.
And this file you can edit yourself and add your own questions that you might want to get
answers from Andrew or you can also append directly to the Eval questions list in this way.
Now let's take a look at the Eval questions list and you can see that this question has been
added at the end. Go ahead and add your own questions.
And now we have everything set up to get to the most exciting step in this notebook. With this
code snippet, we can execute the sentence window engine on each question in the list of Eval questions
that we just looked at. And then with TrueRecorder, we are going to run each record against the RagTriad
we will record the prompts, responses, intermediate results and the evaluation results in the True database.
And you can see here as each as the execution of the steps are happening for each record,
there is a hash that's an identifier for the record as the record gets added.
We have an indicator here that that step has executed effectively.
In addition, the feedback results or answer relevance is done and so on for context relevance and so on.
Now that we have the recording done, we can see the logs in the notebook by executing
by getting the records and feedback and executing this code snippet.
And I don't want you to necessarily read through all of the information here.
The main point I want to make is that you can see the depth of instrumentation
in the application. A lot of information gets logged through the TrueRecorder.
And this information around prompts, responses, evaluation results and so forth can be quite valuable
to identify failure modes in the apps and to inform iteration and improvement of the apps.
All of this information is available in a flexible JSON format.
So they can be exported and consumed by downstream processes.
Next up, let's look at some more human readable format for prompts, responses and the feedback
function evaluations. With this code snippet, you can see that for each input, prompt or question,
we see the output and the respective scores for context relevance,
groundedness and answer relevance. And this is run for each and every entry in the list of
questions in evaluation, underscore questions dot text. You can see here the last question is
how can I be successful in AI was the question that I manually appended to that list at the end.
Sometimes in running the evaluations, you might see an end that likely happens because of
API call failures, you'll just want to rerun it to make sure that the execution successfully completes.
I just showed you a record level view of the evaluations, the prompts, responses and
evaluations. Let's now get an aggregate view in the leaderboard, which aggregates across
all of these individual records and produces an average score across the 10 records in that database.
So you can see here in the leaderboard, the aggregate view across all the 10 records,
we had said that by due to app 1, the average context relevance is 0.56. Similarly,
their average scores for groundedness, answer relevance and latency across all the 10 records
of questions that were asked of the rag application. And then the cost is the total cost in dollars
across these 10 records. It's useful to get this aggregate view to see how well your app is
performing and at what level of latency and cost. In addition to the notebook interface,
TrueLens also provides a local streamlit app dashboard with which you can
examine the applications that you're building. Look at the evaluation results,
drill down into record level views to both get aggregate and detailed evaluation views
into the performance of your app. So we can get the dashboard going with the TrueDotron dashboard
method and this sets up a local database at a certain URL. Now once I click on this,
this might show up in some window, which is not within this frame.
Let's take a few minutes to walk through this dashboard. You can see here the
aggregate view of the apps performance. 11 records were processed by the app and evaluated
the average latency is 3.55 seconds. We have the total cost, the total number of tokens
that were processed by the LLAMs and then scores for the rag triad. For context relevance,
it's 0.56 for groundedness 0.86 and answer relevance 0.92. We can select the app here to get a
more detailed record level view of the evaluations.
For each of the records, you can see that the user input, the prompt, the response,
this metadata, the timestamp and then scores for answer relevance, context relevance and
groundedness that have been recorded along with latency, total number of tokens and total cost.
Let me pick a row in which the LLAM indicates, evaluation indicates that the LLAM,
the rag application has done well. Let's pick this row. Once we click on a row,
we can scroll down and get a more detailed view of the different components of that row from
the table. The question here, the prompt was, what is the first step to becoming good at AI?
The final response from the rag was, has to learn foundational technical skills. Down here,
you can see that the answer relevance was viewed to be one on a scale of 0 to 1. It's a
relevant, quite a relevant answer to the question that was asked. Up here, you can see that
context relevance, the average context relevance score is 0.8. For the two pieces of context
that were retrieved, both of them individually got scores of 0.8. We can see the chain
of thought reason for why the LLAM evaluation gave a score of 0.8 and to this particular
response from the rag and in the retrieval step. Down here, you can see the groundedness
evaluations. This was one of the clauses in the final answer. It got a score of 1. Over here is
the reason for that score. You can see this was the statement sentence and the supporting evidence
backs it up and so it got a full score of 1 on a scale of 0 to 1 or a full score of 10 on a
scale of 0 to 10. So previously, the kind of reasoning and information we were talking about
through slides and in the notebook. Now, you can see that quite neatly in this
streamlit local app that runs on your machine. You can also get a detailed view of the timeline
as well as get access to the full JSON object. Now, let's look at an example where the rag did not do
so well. It says, I look through the evaluations. I see this row with a low grounded score of 0.5.
So let's click on that. That brings up this example. The question is how can altruism be beneficial
in building a career? There's a response. If I scroll down to the groundedness evaluation,
then both of the sentences in the final response have low grounded score. Let's speak one of these
and look at why the groundedness score is low. So you can see this overall response got broken down
into four statements and the top two were good, but the bottom two did not have good supporting
evidence in the retrieve pieces of context. In particular, if you look at this last one,
the final output from the LLAM says additionally practicing altruism can contribute to personal
fulfillment in a sense of purpose, which can enhance motivation and overall well-being ultimately
benefiting one's career success. While that might very well be the case, there was no supporting
evidence found in the retrieved pieces of context to ground that statement, and that's why our
evaluation gives this a low score. You can play around with the dashboard and explore some of
these other examples where the LLAM, the final rag output does not do so well to get a feeling
for the kinds of failure modes that are quite common when you're using rag applications.
And some of these will get addressed as we go into the sessions on more advanced rag techniques,
which can do better in terms of addressing these failure modes. Lesson two is a wrap with that.
In the next lesson, we will walk through the mechanism for sentence window-based retrieval
and advanced rag technique, and also show you how to evaluate the advanced technique
leveraging the rag-triad and trulence.

## 04. Sentence-window retrieval
In this lesson, we'll do a deep dive into an advanced rag technique, our sentence window
retrieval method.
In this method, we retrieve based on smaller sentences to better match the relevant context,
and then synthesize based on an expanded context window around the sentence.
Let's check out how to set it up.
First in context, the standard rag pipeline uses the same text shrunk for both embedding
and synthesis.
The issue is that embedding based work tree vol typically works all with smaller chunks,
whereas the ALLEM needs more context and bigger chunks to synthesize a good answer.
What sentence window retrieval does is decouple the two a bit.
We first embed smaller chunks or sentences and store them in a vector database.
We also add context of the sentences that occur before and after to each chunk.
During retrieval, we retrieve the sentences that are most relevant to the question with
a similarity search, and then replace the sentence with a full surrounding context.
This allows us to expand the context that's actually fed to the ALLEM in order to answer
the question.
This notebook will introduce the various components needed to construct a sentence window
retriever with LOML index.
The various components will be covered in detail.
At the end, ONIFOM will show you how to experiment with parameters and evaluation with true
error.
This is the same setup that you've used in the previous lessons, so make sure to install
the relevant packages, such as LOML index and true lines evo.
For this quick start, you'll need an open AI key similar to previous lessons.
This open AI key is used for embeddings, ALLEMs, and also the evaluation piece.
Now we set up and inspect our documents to use for iteration and experimentation.
Similar to the first lesson, we encourage you to upload your own PDF file as well.
As with before, we'll load in the how to build a career in AI ebook.
It's same document as before, so we see that it's a list of documents.
There are 41 pages.
The object schemas are document object, and here's some sample text from the first page.
The next piece is we'll merge these into a single document because it helps with overall
text blending accuracy when using more advanced retrievers.
Now let's set up the sentence window retrieval method, and we'll go through how to set
this up more in depth.
We'll start with a window size of 3 and a top k value of 6.
First we'll import, we'll be called a sentence window node parser.
The sentence window node parser is an object that will split a document into individual
sentences and then augment each sentence trunk with a surrounding context around that sentence.
Here we demonstrate how the node parser works with a small example.
We see that our text which has three sentences gets split into three nodes.
Each node contains a single sentence with the metadata containing a larger window around
the sentence.
We'll show what that metadata looks like for the second node right here.
You see that this metadata contains the original sentence, but also the sentence that occurred
before an effort.
We encourage you to try out your own text too.
For instance, let's try something like this.
For this sample text let's take a look at the surrounding metadata for the first node.
Since the window size is 3, we have two additional adjacent nodes that occur in front and of course
none behind it because it's the first node.
So we see that we have the original sentence or hello, but also Fubar and CatDark.
The next step is to actually build the index and the first thing we'll do is to set up
an L on.
In this case we'll use OpenAI, specifically a GPT 3.5 turbo with a temperature of 0.1.
The next step is to set up a service context object, which as a reminder is a wrapper object
that contains all the context needed for indexing, including the L on embedding model and
the node parser.
Note that the embedding model that we specify is the BGE small model and we actually download
and run it locally from hugging face.
This is a compact, fast and accurate for its size and embedding model.
We can also use other embedding models.
For instance, a related model is BGE large, which we have in the commented out code.
The next step is to set up a vector store index with a source document.
Because we've defined the node parser as part of the service context, what this will do
is it will take the source document, transform it into a series of sentences, augmented with
surrounding context, and embed it and load it into the vector store.
We can save the index to disk so that you can load it later without re-dulling it.
If you've already built the index and saved it and you don't want to rebuild it, here
is a handy block of code that allows you to load the index from the existing file if
it exists, otherwise it will build it.
The index is now built.
The next step is to set up and run the query engine.
First what we'll do is we'll define what we call a metadata replacement post processor.
This takes a value stored in the metadata and replaces the node text with that value.
And so this is done after retrieving the nodes and before sending the nodes to the outline.
We'll first walk through how this works.
Using the nodes we created with the sentence window node parser, we can test this post processor.
Note that we made a backup of the original nodes.
Let's take a look at the second node again.
Great.
Now let's apply the post processor on top of these nodes.
If we now take a look at the text of the second node, we see that it's been replaced with
a full context including the sentences that occurred before and after the current node.
The next step is to add the sentence transformer rewrank model.
This takes the query and retrieve nodes and reorder the nodes in order of relevance using
a specialized model for the task.
Generally, you would make the initial similarity top k larger and then the rewranker will
rescore the nodes and return a smaller top end so it will filter out a smaller set.
An example of a rewranker is BGE rewranker base.
This is a rewranker based on the BGE embeddings.
This string represents the models named from hugging face and you can find more details
on the model from hugging face.
Let's take a look at how this rewranker works.
We'll input some toy data and then see how the rewranker can actually rewrank the initial
set of nodes to a new set of nodes.
Let's assume the original query is I want a dog and the initial set of scored nodes is
this is a cat with a score of 0.6 and then this is a dog with a score of 0.4.
Intuitively, you would expect that the second node actually has a higher score so matters
the query more and so that's where the rewranker can come in.
Here we see the rewranker properly surfaces the node about dogs and gave it a high score
of relevance.
Now let's apply this to our actual query entered.
As mentioned earlier, we want a larger similarity top-k and the top-end value we chose for
the rewranker in order to give the rewranker a fair chance of surfacing the proper information.
We set the top-k equal to 6 and top-end equals to 2, which means that we first fetched
the six most similar chunks using the sentence window retrieval and then we filter for the
top two most relevant chunks using the sentence rewranker.
Now that we have the full query entrance set up, let's run through a basic example.
Let's ask a question over this dataset.
What are the keys to building a query in AI?
And we get back in response.
We see that the final response is that the keys to building a query in AI are learning
foundational technical skills, working on projects, and finding a draw.
Now that we have the sentence window query entered in place, let's put everything together.
We'll put a lot of code into this notebook cell, but note that this is essentially the
same as the function in the utils API file.
We have functions for building the sentence window index that we showed earlier in this
notebook.
It consists of being able to use the sentence window node parser to extract out sentences
from documents and augment it with surrounding context.
It contains setting up the sentence context or using the service context object.
It also consists of setting up a vector store index, using the source documents and the
service context, containing the L1 embedding model and node parser.
The second part of this is actually getting the sentence window query entered, which
we showed consists of getting the sentence window retriever, using the metadata replace
my post processor to actually replace a node with the surrounding context, and then finally
using a rewriting model to filter for the top end results.
We combine all of this using the as query enter module.
Let's first call build sentence window index with the source document, the L1, as well
as the save directory, and then let's call the second function to get the sentence
but a query enter.
Great.
Now you're ready to experiment with sentence window retrieval.
In the next section, on your problem, we'll show you how to actually run evaluations using
the sentence window retriever so that you can evaluate the results and actually play
around the parameters and see how that affects the performance of your engine.
After running through these examples, we encourage you to add your own questions and then
even define your own evaluation benchmarks, just to play around with this again a sense
of how everything works.
Thanks, Jerry.
Now that you have set up the sentence window retriever, let's see how we can evaluate
it with the rag triad and compare its performance to the basic rag with experiment tracking.
Let us now see how we can evaluate and iterate on the sentence window size parameter to make
the right trade-offs between the evaluation metrics or the quality of the app and the
cost of running the application and evaluation.
We will gradually increase the sentence window size, starting with one, evaluate these
successive app versions with true lens and the rag triad, track experiments to pick the
best sentence window size.
And as we go through this exercise, we will want to know the trade-offs between token usage
or cost.
As we increase the window size, the token usage and cost will go up, as in many cases will
context relevance.
At the same time, increasing the window size in the beginning, we expect will improve
context relevance and therefore will also indirectly improve groundedness.
One of the reasons for that is when the retrieval step does not produce sufficiently relevant
context, the LLAM in the completion step will tend to fill in those gaps by leveraging
its pre-existing knowledge from the pre-training stage rather than explicitly relying on the
retrieved pieces of context.
And this choice can result in lower groundedness scores because recall groundedness means components
of the final response should be traceable back to the retrieved pieces of context.
Consequently, what we expect is that as you keep increasing your sentence window size,
context relevance will increase up to a certain point, as will groundedness and then beyond
that point, we will see context relevance either either flatten out or decrease and groundedness
is likely going to follow a similar pattern as well.
In addition, there is also a very interesting relationship between context relevance and
groundedness that you can see in practice when context relevance is low, groundedness
tends to be low as well.
This is because the LLAM will usually try to fill in the gaps in the retrieved pieces
of context by leveraging its knowledge from the pre-training stage.
This results in a reduction in groundedness even if the answers actually happen to be
quite relevant.
As context relevance increases, groundedness also tends to increase up to a certain point,
but if the context size becomes too big, even if the context relevance is high, there
could be a drop in the groundedness because the LLAM can get overwhelmed with contexts
that are too large and fall back on its pre-existing knowledge base from the training phase.
Let us now experiment with the sentence window size.
I will walk you through a notebook to load a few questions for evaluation and then gradually
increase the sentence window size and observe the impact of that on the rag-triad evaluation
metrics.
First, we load a set of pre-generated evaluation questions.
You can see here some of these questions from this list.
Next, we run the evaluations for each question in that reloaded set of evaluation
questions and then with the true recorder object, we record the prompts, the responses,
the intermediate results of the application as well as the evaluation results in the
true database.
Let us now adjust the sentence window size parameter and look at the impact of that on the
different rag-triad evaluation metrics.
We will first reset the true database.
With this code snippet, we set the sentence window size to 1.
You'll notice that in this instruction, everything else is the same as before.
Then we set the sentence window engine with the get sentence window query engine associated
with this index and next up, we are ready to set up the true recorder with the sentence
window size set to 1.
This sets up the definition of all the feedback functions for the rag-triad, including answer
relevance, context relevance and groundedness.
And now we have everything set up to run the evaluations.
For the set up with the sentence window size set to 1.
And all the relevant prompts, responses, intermediate results and the results of the evaluation
of these feedback functions will get logged into the true database.
Okay, that ran beautifully.
Now, let's look at it in the dashboard.
You'll see that this instruction brings up a locally hosted stream lit up and you can
click on the link to get to the stream lit up.
So the app leaderboard shows us the aggregate metrics for all the 21 records that we ran
through and evaluated with true lands.
The average latency here is 4.57 seconds.
The total cost is about 2 cents.
Total number of tokens processed is about 9,000.
And you can see the evaluation metrics, the application does reasonably well in answer
relevance and groundedness, but on context relevance it's quite poor.
Let's now drill down and look at the individual records that were processed by the application
and evaluated.
If I scroll to the right, I can see some examples where the application is not doing so well
on these metrics.
So let me pick this row and then we can go deeper and examine how it's doing.
So the question here is in the context of project selection and execution, explain
the difference between ready and fire and ready fire aim approaches, provide examples
where each approach might be more beneficial.
You can see the overall response here in detail from the rag.
And then if we scroll down, we can see the overall scores for groundedness, context relevance
and answer relevance.
Two pieces of context were retrieved in this example.
And for one of the pieces of retrieved context, context relevance is quite low.
Let's drill down into that example and take a closer look.
What you'll see here with this example is that the piece of context is quite small.
Remember that we are using a sentence window of size one, which means we have only added
one sentence extra in the beginning and one sentence extra at the end around the retrieve
piece of context.
And that produces a fairly small piece of context that is missing out on important information
that would make it relevant to the question that was asked.
Similarly, if you look at groundedness, we will see that both of these pieces of retrieved
the sentences in the final summary, the groundedness scores are quite low.
Let's pick the one with the higher groundedness score, which has a bit more justification.
And if we look at this example, what we will see is there are a few sentences here in
the beginning for which there is good supporting evidence in the retrieved piece of context.
And so the score here is high, it's a score of 10 on a scale of 0 to 10.
But then for these sentences down here, there wasn't supporting evidence.
And therefore, the groundedness score is 0.
Let's take a concrete example.
If this one, it's often used in situations where the cost of execution is relatively low
and where the ability to iterate and adapt quickly is more important than upfront planning.
This does feel like a plausible piece of text that could be useful as part of the response
to the question.
However, it wasn't there in the retrieved piece of context.
It's not backed up by any supporting evidence in the retrieved context.
This could possibly have been part of what the model had learned during its straining
phase where either from the same document, Andrew's document here on career advice for
AI or some other source talking about the same topic, the model may have learned similar
information.
But it's not grounded in that it is not the sentence is not supported by the retrieved
piece of context in this particular instance.
So this is a general issue when the sentence window is too small that context relevance
tends to be low and as a consequence, groundedness also becomes low because the LLM starts
making use of its pre-existing knowledge from its straining phase to start answering questions
instead of just relying on the supplied context.
Now that I have shown you a failure mode with sentence windows set to one, I want to walk
through a few more steps to see how the metrics improve as we change the sentence window size.
For the purpose of going through the notebook quickly, I'm going to reload the evaluation
for questions, but in this instance, just set it to the one question where the model
had problem, this particular question, which we just walked through with the sentence window
size set at one, and then I want to run this through with the sentence in sentence window
size set to three.
This code snippet is going to set up the the rag with sentence window size set up three
and also set up the true recorder for it.
We now have the definition of the feedback function set up in addition to the rag with
the sentence window set at size three.
Next up, we are going to run the evaluations for that particular evaluation question that
we have looked through in some detail with the sentence window set through one where we
observe the failure mode.
It has run successfully.
Let's now look at the results with sentence window engine set to three in the true lens dashboard.
You can see the results here.
I ran it on the one record.
That was the problematic record when we looked at sentence window size one.
And you can see a huge increase in the context relevance.
It went up from 0.57 to 0.9.
Now if I select the app and look at this example in some more detail, let's now look at the
same question that we looked at with sentence window set at one.
Now we are at three.
Here's the full final response.
Now if you look at the retrieved pieces of context, you'll notice that this particular
piece of retrieved context is similar to the one that we had retrieved earlier with
sentence window set at size one.
But now it has the expansion because of the bigger sentence window size.
And if you look at the score for this section, we'll see that this context got a context
relevance score of 0.9, which is higher than the score of 0.8 that the smaller context had
gotten earlier.
And this example shows that with an expansion in the sentence window size, even reasonably
good pieces of retrieved context can get even better.
Once the completion step is done with these significantly better pieces of context, the
groundedness score goes up quite a bit.
We'll see that by finding supporting evidence across these two pieces of highly relevant
context, the groundedness score actually goes up all the way to 1.
So increasing the sentence window size from 1 to 3 led to a substantial improvement in
the evaluation metrics of the rag triad.
Both groundedness and context relevance went up significantly as did answer relevance.
And now we can look at sentence window set to 5.
If you look at the metrics here, a couple of things to observe.
One is the total tokens has gone up.
And this could have an impact on the cost if we were to increase the number of records.
So that's one of the tradeoffs that I mentioned earlier.
As you increase the sentence window size, it gets more expensive because more tokens
are being processed by the LLAMs during evaluation.
The other thing to observe is that while context relevance and answer relevance have remained
flat, groundedness has actually dropped with the increase in the sentence window size.
And this can happen after a certain point because as the context size increases, the LLAM
can get overwhelmed in the completion step with too much information.
And in the process of summarization, it can start introducing its own pre-existing knowledge
instead of just the information in the retrieved pieces of context.
So to wrap things up here, it turns out that as we gradually increase the sentence window size
from 1 to 3 to 5, the size of 3 is the best choice for us for this particular evaluation.
And we see the increase in context relevance and answer relevance and groundedness as
we go from 1 to 3, and then a reduction or degradation in the groundedness step with
a further increase to a size of 5.
As you are playing with the notebook, we encourage you to rerun it with more records in
these two steps, examine the individual records which are causing problems for specific
metrics like context relevance or groundedness and get some intuition and build some intuition
around why the failure modes are happening and what to do to address them.
And in the next section, we will look at another advanced rack technique, auto merging to address
some of those failure modes.
Irrelevant context can creep into the final response resulting in not such great scores
in groundedness or answer relevance.

## 05. Auto-merging retrieval
In this lesson, we'll do a deep dive into another advanced rag technique, autovergrain.
An issue with the naive approach is that you're retrieving a bunch of fragmented context
chunks to put into the LLM context window.
And the fragmentation is worse, the smaller your chunk size.
Here, we use an auto-merging heuristic to merge smaller chunks into a bigger parent
chunk to help ensure more coherent context.
Let's check out how to set it up.
In this section, we'll talk about auto-merging retrieval.
What's interesting with the standard rag pipeline is that you're retrieving a bunch
of fragmented context chunks to put into the LLM context window.
And the fragmentation is worse, the smaller your chunk size.
For instance, you might get back two or more retrieved context chunks and roughly the
same section, but there's actually no guarantees on the ordering of these chunks.
This can potentially hamper the LLM's ability to synthesize over this retrieved context
within its context window.
So what auto-merging retrieval does is the following.
First, define a hierarchy of smaller chunks linking the bigger parent chunks, for each parent
chunk can have some number of children.
Second, during retrieval, if the set of smaller chunks linking to a parent chunk exceeds
some percentage threshold, then we merge smaller chunks into the bigger parent chunk.
So we retrieve the bigger parent chunk instead to help ensure more coherent context.
Now let's check out how to set this up.
This notebook will introduce the various components needed to construct an auto-merging retriever
with LLM index.
The various components will be covered in detail.
And similar to the previous section, at the end, Audubon will show you how to experiment
with parameters and evaluation with Trader.
Similar to before, we'll load in the OpenAI API key, and we'll load this using a convenience
helper function in our utils file.
As with the previous lessons, we'll also use the how to build a career in AI PDF.
And as before, we also encourage you to try out your own PDF files as well.
We load in 41 document objects, and we'll merge them into a single large document, which
makes this more amenable for text blending with our advanced retrieval methods.
Now, we're ready to set up our auto-merging retriever.
This will consist of a few different components, and the first step is to define what we call
a hierarchical node parser.
In order to use an auto-merging retriever, we need to parse our nodes in a hierarchical fashion.
This means that nodes are parsed in decreasing sizes and contained relationships to their
parent node.
Here we demonstrate how the node parser works with a small example.
We create a toy parser with small chunk sizes to demonstrate.
Notice that the chunk sizes we use are 248, 512, and 128.
You can change the chunk sizes to any sort of decreasing order that you'd like.
Here we do it by a factor of 4.
Now let's get the side of nodes from the document.
What this does is this actually returns all nodes.
This returns all leaf nodes, intermediate nodes, as well as parent nodes.
There's going to be a decent amount of overlap of information and content between the leaf
intermediate and parent nodes.
If we only want to retrieve the leaf nodes, we can call a function within LOM index called
GAT leaf nodes, and we can take a look at what that looks like.
In this example, we call GAT leaf nodes on the original set of nodes, and we take a look
at the 31st node to look at the text.
We see that the text chunk is actually fairly small, and this is an example of a leaf node,
because a leaf node is the smallest chunk size of 128 tokens.
Here's how you might go about strengthening your math background to figure out what's
important to know, etc.
Now that we've shown what a leaf node looks like, we can also explore the relation trips.
We can print the parent of the above node and observe that it's a larger chunk containing
the text of the leaf node, but also more.
More concretely, the parent node contains 512 tokens while having four leaf nodes that
contain 128 tokens.
There's four leaf nodes, because the tug sizes are divided by a factor of 40's shine.
This is an example of what the parent node of the 31st leaf node looks like.
Now that we've shown you what the node hierarchy looks like, we can now construct our index.
We'll use the OpenAI, LLN, and specifically GPT 3.5 turbo.
We'll also define a service context object containing the LLN embedding model and the
hierarchical node parser.
As with the previous notebooks, we'll use the BG, Small, and Embedding model.
The next step is to construct our index.
The way that index works is that we actually construct a vector index on specifically
if the leaf nodes.
All other intermediate and parent nodes are stored in a docStore and are retrieved dynamically
during retrieval, but what we actually fetch during the initial top-k embedding lookup
is specifically the leaf nodes, and that's what we embed.
You see in this code that we define a storage context object, which by default is initialized
with an in-memory document store, and we call storagecontext.docStore.addDocuments to
add all nodes to this in-memory docStore.
However, when we create our vectorStore index, called auto-merging index right here,
we only pass in the leaf nodes for vector indexing.
This means that the specifically the leaf nodes are embedded using the embedding model and
also indexed, but we also pass in the storage context as well as the service context.
And so the vector index does have knowledge of the underlying docStore that contains
all the nodes.
And finally, we persist this index.
If you've already built this index and you want to load it from storage, you can just
copy and paste this block a code, which will rebuild the index of it doesn't exist or
load it from storage.
The last step now that we've defined the auto-merging index is to set up the retriever
and run the query engine.
The auto-merging retriever is what controls immersion logic.
If a majority of children nodes are retrieved for a given parent, they are swapped out for
the parent instead.
In order for this immersion to work well, we set a large top-k for the leaf nodes.
Remember, the leaf nodes also have a smaller chunk size of 128.
In order to reduce token usage, we apply a re-ranker after the immersion has taken place.
For example, we might retrieve the top 12, merge and have a top 10, and then re-rank into
a top 6.
The top end for the re-ranker may seem larger, but remember that the base chunk size is
only 128 tokens, and then the next parent above that is 512 tokens.
We import a class called auto-merging retriever, and then we define a sentence transformer
of re-rank module.
We combine both the auto-merging retriever and the re-rank module into our retriever query
engine, which handles both retrieval and synthesis.
Now that we've set this whole thing up and to add, let's actually test what is the
importance of networking in AI as an example question.
We get back a response.
We see that it says networking is important in AI because it allows individuals to build
a strong professional network and more.
The next step is to put it all together, and we'll create two high-level functions,
build auto-merging index, as well as get auto-merging query engine.
And this basically captures all the steps that we just showed you.
And the first function, build auto-merging index, will use the hierarchical node parser
to parse out the hierarchy of child-to-parent nodes, will defy the service context, and
will create a vector store index from the leaf nodes, but also linking to the document
store of all the nodes.
The second function, get auto-merging query engine, leverages our auto-merging retriever,
which is able to dynamically merge leaf nodes into parent nodes, and also use our re-rank
module, and then combine it with the overall retriever query engine.
So we build the index using the build auto-merging index function, using the original source
document, the lm set to GPT 3.5 turbo, as well as the merging index as a save directory.
And then for the query engine, we call get auto-merging query engine, based on the index,
as well as we set a similarity top-k of equal to 6.
As a next step, on the problem we'll show you how to evaluate the auto-merging retriever,
and also iterate on parameters using TrierRot.
We encourage you to try out your own questions as well, and also iterate on the parameters
of auto-merging retrieval.
For instance, what happens when you change the trunk sizes, or the top-k, or the top-end
for the re-ranker?
Play around with it, and tell us what the results are.
That was awesome, Jerry.
Now that you have set up the auto-merging retriever, let's see how we can evaluate it with
the rag triad, and compare its performance to the basic rag with experiment tracking.
Let's set up this auto-merging new index.
You'll notice that it's two layers.
The lowest layer chunk, the leaf nodes, will have a chunk size of 512, and the next layer
up in the hierarchy, is a chunk size of 2048, meaning that each parent will have four
leaf nodes of 512 tokens each.
The other pieces of setting this up are exactly the same as what Jerry has shown you earlier.
One reason you may want to experiment with the two layer auto-merging structure is that
it's simpler.
Less work is needed to create the index, as well as in the retrieval step, there is less
work needed, because all the third layer checks go away.
If it performs comparably well, then ideally we want to work with a simpler structure.
Now that we have created the index with this two layer auto-merging structure, let's set
up the auto-merging engine for this setup.
I'm keeping the top K at the same value as before, which is 12, and the re-ranking step
will also have the same N equal 6.
This will let us do a more direct head-to-head comparison between this application setup and
the three layer auto-merging hierarchy app that Jerry had set up earlier.
Now let's set up the true recorder with this auto-merging engine, and we will give this
an app ID of app 0.
Let's now load some questions for evaluation from the generated questions.text file that
we have set up earlier.
Now we can define the running of these evaluation questions.
For each question in about questions, we are going to set things up so that the true
recorder object, when invoked, with the runnivels, will record the prompts, responses, and
the evaluation results, leveraging the query engine.
Now that our evaluations have completed, let's take a look at the leaderboard.
We can see that app 0 metrics here, context relevance seems slow, the other two metrics
are better.
This is with art, two level hierarchy with 512 as the leaf node chunk size and the parent
being 2048 tokens, so for leaf nodes per parent node.
Now we can run the true dashboard and take a look at the evaluation results at the record
level at the next layer of detail.
Let's examine the app leaderboard.
You can see here that after processing 24 records, the context relevance at an aggregate
level is quite low, although the app is doing better on answer relevance and groundedness.
I can select the app, let's now look at the individual records of app 0 and see how
the evaluation scores are for the various records.
You can scroll to the right here and look at the scores for answer relevance, context relevance
and groundedness.
Let's pick one that has low context relevance.
So here is one, if you click on it, you'll see the more detailed view down below.
The question is discussed the importance of budgeting for resources and the successful
execution of AI projects and the right here is the response and if you scroll down further
you can see a more detailed view for context relevance.
There were six pieces of retrieve context, each of them has a score to be particularly
low in their evaluation scores between 0 and 0.2.
And if you pick any of them and click on it, you can see that the response is not particularly
relevant to the question that was asked.
You can also scroll back up and explore some of the other records.
You can pick ones, for example, that the scores are good, like this one here and explore
how the application is doing on various questions and where it strings are, where it's failure
modes are to build some intuition around what's working and what's not.
Let's now compare the previous app to the auto merging setup that Jerry introduced earlier.
We will have three layers now in the hierarchy, starting with 128 tokens at the leaf node
level, 512 one layer up and 2048 at the highest layer.
So at each layer, each parent has or children.
Now let's set up the query engine for this app setup, the true recorder, all identical
steps as the one for the previous app.
And finally, we are in a position to run the evaluations.
Now that we have app 1 set up, we can take a quick look here in the leader board.
You can see that relative to app 0, the number of tokens processed in app 1 for the same
number of records is about half.
And the total cost is also about half, and that's because recall this has three layers
in the hierarchy and the chunk size is 128 tokens instead of the 512, which is the smallest
leaf node token size for app 0.
So that results in a cost reduction.
It is also that context relevance has increased by about 20%.
And part of the reason that's happening is that the merging is likely happening a lot
better with this new app setup.
We can also drill down and look at app 1 in greater detail like before.
We can look at individual records.
Let's pick the same one that we looked at earlier with app 0.
It's the question about the importance of budgeting.
And now you can see context relevance is doing better.
Groundedness is also considerably higher.
And if we pick a sample example, a response here, you'll see that in fact it is talking
very specifically about budgeting for resources.
So there is improvement in this particular instance and also at an aggregate level.
Let me now summarize some of the key takeaways from lesson 4.
We walked you through an approach to evaluate and iterate with the auto retrieval advanced
drag technique.
And in particular, we showed you how to iterate with different hierarchical structures,
the number of levels, the number of child nodes and chunk sizes.
And for these different versions, you could evaluate them with the drag triad and track
experiments to pick the best structure for your use case.
One thing to notice is that not only are you getting the metrics associated with the
drag triad as part of the evaluation, but the drill down into the record level can help
you gain intuition about hyper parameters that work best with certain dark types.
For example, depending on the nature of the documents, such as employment contracts
versus invoices, you might find that different chunk sizes and hierarchical structures work
best.
Finally, one other thing to note is that auto merging is complementary to sentence window retrieval.
And one way to think about that is, let's say you have four child nodes of a parent with
auto merging, you might find that child number one and child number four are very relevant to
the query that was asked and these then get merged under the auto merging paradigm.
In contrast, sentence windowing may not result in this kind of merging because they are not
in a contiguous section of the text.
That brings us to the end of lesson four.
We have observed that with advanced rack techniques such as sentence windowing and auto merging
retrieval augmented with the power of evaluation and experiment tracking and iteration,
you can significantly improve your drag applications.
In addition, while the course has focused on these two techniques and the associated drag
triad for evaluation, there are a number of other evaluations that you can play with in order
to ensure that your LLM applications are honest, harmless and helpful.
This slide has a list of some of the ones that are available out of the box in true lens.
We encourage you to go play with true lens, explore the notebooks and take your learning to the next level.

## 06. Conclusion
Congrats on finishing the course.
Hopefully, you have picked up some skills on how to build, evaluate, and iterate on your
RAG application to make a more production ready.
Regardless of whether you come from a data science machine learning background or
traditional software background, you'll need to learn some of these or development principles
so that you can be a rockstar AI engineer, we can build robust Alan software systems.
Reducing a LLM hallucination is going to be the top priority for every developer
as the field evolves. We are excited to see the base models get better and larger scale evaluations
become cheaper and more accessible for everyone to set up and run.
As a next step, I'd recommend looking more deeply into understanding your data pipeline,
retrieval strategy, and LLM prompts to help improve RAG performance.
These two techniques we showed were just the tip of the iceberg. You should look into everything
from chunk sizes, to retrieval techniques like hybrid search, to Alan-based reasoning like
chain of thought. The rack triad is an excellent place to start with evaluating your RAG-based
LLM apps. As a next step, I encourage you to dig deeper into the area of evaluating LLM's
and the apps that they power. This includes topics such as assessing model confidence,
calibration, uncertainty, explainability, privacy, fairness, and toxicity in both benign
and adversarial settings. We look forward to seeing what you'll build next.
