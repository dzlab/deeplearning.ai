# LLMOps

## Introduction
Welcome to LLMOps, built in partnership with Google Cloud and taught by
Irvin Hauserna. Say you design and deploy one LLM-based use case,
such as summarizing customer emails, and it takes you a few ways to do that.
That probably took a bunch of work to select the large language model to build on,
which by me trying a few to see what works best,
then tuning the prompts and set the evaluation framework,
deploying and then monitoring performance.
As you build an application, having automated ways to deploy and monitor will make your life easier.
And if you need to update this application,
maybe because the LLM provider is deprecating the model you had built on,
and so you need to switch to a new LLM,
having tools that help you migrate and test will also simplify your workflow.
Now that you've built your first application,
what if a team is so impressed with your summarization application that they come back
and asked for five more use cases, like classifying emails, routing emails, and so on,
where you have to do five times the work of the original project.
OBS, which is an extension of MLOps or machine learning operations
to applications built on large language models,
destroyed the processes and tools to automate the cycle of
data preparation, model tuning, deployment, maintenance, and monitoring.
Take just the example of prompt management.
When designing prompts, it can be very helpful to have an automated way of
evaluating and comparing the performance of different prompts,
and also updating the prompt if needed,
such as if the underlying LLM changes.
In addition, if your application uses multiple LLM calls,
for example, chaining multiple steps of processing together,
maybe using an orchestration framework like LLM chain,
then managing the dependencies also as additional complexity.
In this course, you learn the latest best practices for how to manage all this complexity.
Delighted to introduce the instructor for this course,
Irvin Hausner, who is a developer advocate in machine learning at Google Cloud,
and has deep experience with LLMOps.
Thanks, Andrew. I am excited to work with you and your team on this.
In this course, you learn how to build and end-to-end workflow
for your large language model based application.
Specifically, you learn how to prepare data for tuning a large language model,
say using some sort of path or parameter efficient fine tuning.
You also learn how to automate and orchestrate the LLM tuning workflow
for multiple use cases, including when you have huge text datasets that might be too large
to fit in memory. You learn how to deploy your model as a REST API and call the API.
If you don't know where the REST API is, don't worry about it. Irvin will cover that too.
You also learn how to build an overall LLMOps workflow. Irvin will go through how to use
SQL for data preparation, open source tools for orchestration and automation,
and vertex AI to deploy your model for production.
The rapid advancements in generative AI and large language models have ignited a search of
creativity and innovation with Tendi developer community. This means more and more use cases that
developers have to build and manage. Effective management and operational strategies become
really important if you want to ship your code to real users and do so efficiently.
Building AI applications has become easier thanks to developments like foundation models as APIs
and open source LLMs. Now it's possible to build many different use cases quickly,
so rather than carefully planning out something, you might even build multiple use cases.
But this complexity or dare I say pain of managing and scaling multiple
projects can be quite high unless you have good LLMOps tooling and processes to make them efficient.
So through this course I hope you also be able to manage your AI production use cases more
efficiently. No prior LLMOps knowledge needed. Many people have worked to create this course.
I'd like to thank on the Google Cloud side, Nikita Namjushi, who's taught other short courses with us
and Dave Elliott. And from the deep line with the AI team, Ed Issue has also contributed to this course.
In the next video, Irvin will give an overview of the key ideas of LLMOps. If you've heard of
MLOps before and it's fine if you haven't, you also see what the key differences are and why
LLMOps is pushing us beyond what machine learning had previously done before large language models.
Let's go on to the next video to get started.

## The Fundamentals
In this lesson, you will learn about the concepts and ideas of LLM ops and how it's similar
and different to ML ops.
In this first lesson, you will learn about concepts like data management, automation and
deployment.
Let's dive into it.
Let's start with what is machine learning operations.
ML ops is an ML engineering culture and practice that aims at unifying ML development,
and ML operations ops.
Automation and monitoring at all steps of the ML system is crucial.
With automating meaning, maybe if you have a use case where you want to deploy a large
language model, you want to automate the process of data engineering, training or tuning
our model, and deploying it as an API in production.
Once you have an introduction, of course, you want to monitor how your LLM performs
in production.
At all of these steps of constructing an ML system, you need to think about things like
how do we integrate these steps?
So how do I go, for example, from data engineering to tuning a model and to deploying it?
How do I test not only my end to end workflow, but also each of these steps?
How do I release a model or my code?
How do I deploy it as an API?
And if you want to train or tune my own custom model and I want to use specific hardware,
a GPUs or a GPUs, how do I manage this?
Let's dive a bit deeper into what does a machine-long operations workflow look like.
Let's have a look at the ML ops framework that you might be familiar with.
When building an ML use case, maybe being a large language model into production,
you have to go through different steps.
First, you might want to think about data ingestion.
How do I take data from my application databases or other databases that I have?
How do I validate this data?
Maybe check if there's any missing data or anything wrong with my data that I have to fix
in the data transformation step.
Data transforming, meaning maybe I'll take my text data and convert it into a format
that the large language model can understand or deal with missing values.
Then you might want to train your model or tune your model.
You need to do model analysis.
So you need to evaluate your model and understand how it performs on your data.
Then you need to think through how to do model serving and what fits for your use case.
And of course, you want to lock all the metrics related to your modeling production.
We'll focus on the ones in blue in this course.
You might think now when you build a workflow like this,
that you don't want to run through each of these steps manually,
meaning that you don't want to execute each step manually yourself,
and then go to the next step and execute that one and so on, so forth.
That is very time consuming and not very efficient.
So how can we help with this?
So this is where automation and orchestration can help.
Orchestration, meaning you telling the system which step needs to be executed first,
which one after and so on, so forth.
So meaning that you go from data ingestion to data validation
all the way to serving your model,
auto-waining, meaning that you automate this process.
You can understand that this is a very one-dimensional process.
So per use case, so let's say you want to build a large language model
that does summarization.
For that use case, you will build a workflow like this.
So it's very one-dimensional.
How are things different when it comes to ML ops for large language models?
Well, there are two things that are different.
First, ML ops for LLMs, also called LLM ops,
is focused more narrowly on everything related to the LLM development
and how to get it running smoothly in production.
LLM system design is looking at the entire application,
which may include things that the front end, back end, data engineers,
and mobile app developers may build.
Let's go through a few examples of what are these difference
between ML ops for LLMs and LLM system design.
When dealing with ML ops for LLMs,
you might have to think through how do you do experimentation.
There's many great models out there.
There's many great foundation models out there,
like Palm or Lama.
You might want to experiment with multiple foundation models
to understand which one is a good fit for your use case.
Let's say summarization.
Then you might design multiple prompts,
and then you have to think through how do you manage these prompts
during experimentation, but also when you use prompts in production.
You might want to use supervised tuning.
So let's say you want to tune your own model
for your summarization use case.
Maybe the Palm API works out of the box for you,
but you want to improve it by using supervised fine tuning on summaries.
Of course, we also need to think through monitoring our LLMs in production,
and how do we evaluate LLMs?
Evaluating LLMs is still a topic where there's a lot of research.
So the way you evaluate your LLM might change over time.
When we look at LLM system design,
there's other things that we have to think about.
Like if we're building a summarization use case,
how are we going to chain multiple steps together?
So let's say we have a lot of documents we want to summarize.
Too much for the LLM to process at once.
So we have to summarize in batches.
So we have to change multiple steps together.
Maybe we also want to do grounding to make sure our LLM has additional information
to get you the output that you desire.
And maybe we want to track history.
What are some of the summaries we created in the past and how did they perform?
All of this is related to LLM system design.
Let's look at a high level example of a LLM driven application.
Everything starts with a user interface.
So how does a user interact with the application?
The user input goes to your backhand.
So what happens behind the scenes?
So what you can do is you might take the user input and do some preprocessing.
Let's take the example of summarizations here.
Let's say it's a lot of information you need to summarize.
You might want to chunk it in smaller pieces and send it bird chunk into your LLM.
This is where you can use preprocessing.
When you're done your preprocessing,
you might want to use some grounding.
So grounding your LLM in some facts.
These facts you can include in your prompt.
And your prompt goes into your model.
You might want to use a foundation model out of the box like the PALM API or a LOMA model.
When you get the response of your LLM,
you might want to do some grounding again.
So check the response against the facts that you have.
Once we have done our grounding, you might want to do some post-processing.
You maybe want to clean up the response and give it a structure that is user-friendly.
You might also want to include and think about responsible AI.
You want to make sure that we build AI responsible.
Maybe you want to check for toxicity or any bias in the response of the LLM.
Anything that's important to you and your use case.
Once we have output we're happy with,
this is what goes back to the user and the user sees the final up.
Maybe you want to tune your own custom model.
So you want to do model customization.
When we do model customization,
you have to go through the process of data preparation, tuning a model,
and of course, evaluating and understanding
how your tuned model is performing on your use case.
And of course, this is an iterative process.
You might want to do this a couple of times
until you have a summarization model that you're happy with.
Once you have a model you're happy with,
you can deploy that model into your production environment
and you have a fine-tuned model you can use
in your LLM driven application.
This is an example workflow.
And of course, depending on your use case,
maybe you have something else in summarization.
Depending on your requirements and your use case,
you might have a different use case than summarization.
You can take a different approach.
So this can look totally different for your use case.
In this course we'll focus on the ones that are in green.
So we'll focus on how do we operate
and build a model customization workflow
and how do we deploy that into production
and how do we send a prompt
and consume the output of that model.
Let's look at that in a bit more detail.
Let's look at a LLM ops pipeline
and let's use the example that we're going to build in this course.
Of course, everything starts with data.
We're going to start with preparing our data sets
and we're going to version our data sets
so that we can keep track of the data sets
that we created.
Next, we're going to design a pipeline
that's going to do supervised tuning
of a large language model for us.
And this might look like a simple box.
A lot of great things happen in this pipeline
and that is automated for us.
Next, we're going to generate an artifact.
Just see this as a file.
And this file has information on our configuration and workflow.
Basically, the workflow is the steps of our pipeline.
First step A and step B and step C, so on, so forth.
The configuration is the configuration
or the parameters that we're going to use
to execute the workflow.
For example, which data set are we going to use
for this supervised fine tuning?
This could be one of our configuration parameters.
Once we have generated our artifact,
we can go and execute a pipeline.
This pipeline will also deploy a model for us.
It will deploy the LLM.
Once we have the LLM deployed, we can get predictions from it.
So we can use prompts to get a response from the LLM.
Once we have the response of the LLM,
we can use responsible AI to check the safety
using safety scores.
Two essential topics for LLM hubs are orchestration.
So we're going to talk about how you can orchestrate a pipeline,
meaning maybe you want to do data preparation
and prepare a dataset first before you do supervised tuning
before you deploy a model.
So you're going to orchestrate the steps in your pipeline.
Now, once we have that pipeline,
secondly, we want to talk about automation
and how we can automate our pipeline
to make our life easier as developers.
This LLM ops pipeline is a simplified diagram,
depending on your use case, on your requirements,
this can differ.
LLM topics beyond the scope of this course.
There's things we can't cover in this course,
things like prompt design and prompt management.
So how do you deal with your prompts
when you're experimenting
and how do you deal with prompts in production?
How do we evaluate a large language model
when we're experimenting or when we have any production?
We'll briefly talk about model evaluation,
but we'll not go in extensive detail.
How do we monitor our LLM in production
or how do we do testing of our LLM system or application?
These are some of the topics you can look into after this course.
As mentioned, we're going to start with data preparation.
So the next lesson, we're going to prepare our datasets.
See you in the next lesson.

## Data Preparation
One of the key steps of LIDM apps is dealing with text data and lots of it.
In this lesson, you learn how to retrieve Stack Overload text data from a data warehouse,
dealing with data that is too large to fit in memory using SQL and modify the data to tune a model to be more task specific.
In order to run this lab, we have to go through some setup code.
We have the setup authentication. We have to say,
this is who I am and I have permissions to access this service and the data in the cloud.
We also have to set up a project.
So a project organizes all of your cloud resources and all of the data lives inside of our project.
So let's get our authentication code.
And in this authentication code, the credential variable is what authenticates us.
Next, we have to set up our region.
The region is where you will run your resources.
So let's say you have to run in a data center in Singapore.
Then you'll select Singapore as your region.
In this example, we're going to use US Central 1 as our default region.
Okay, next we're going to import the Vertex AI library.
This is SDK.
And this SDK lets us interact with the Vertex AI services in the cloud.
We're going to initialize the Vertex SDK using the project at the region and credentials that we just set up.
Now we should be good to go.
So let's dive into the data.
So I want to stay as close to reality as possible.
So in this example, we're going to use a data warehouse.
A data warehouse is a central repository for storing and analyzing large amounts of data from various sources.
It is used to gain insights and make informed decisions.
You can take your data from application database and bring it to your warehouse and make it available for your teams to use.
We'll be using BigQuery.
BigQuery is our data warehouse and it's serverless.
So we don't need to manage service.
Plus it lets us use SQL.
Wait, what?
SQL?
Isn't that very old school?
Well, let me explain this.
So we're using SQL because SQL queries can efficiently process large amounts of data.
Making a good choice for tasks that require high performance.
Plus it's great at cleaning and doing data preparation.
You might have used pandas.
So SQL versus pandas.
So pandas is great if the data fits in memory on your computer.
And SQL is great if your data sits in a data warehouse and you might want to process that skill.
SQL plus a data warehouse is a powerful tool as we discussed when building ML ops systems for large language models.
We want to make sure it's scalable.
Okay, let's get started.
In order to interact with our data warehouse, we again have to initialize a library.
We're going to use the BigQuery client and we're going to initialize it in the same way as with the vertex AI client.
So we're going to import the BigQuery library and then we're going to initialize the project and make sure we're using our credentials.
In this lab, we're going to use the Stack Overflow public data set.
So Stack Overflow is a data set that has questions and answers and metadata related to Stack Overflow questions.
The hierarchy is like this.
You have a data set, the Stack Overflow data set.
And within this data set, we have tables with different types of data.
Tables in a data warehouse are organized collections of data.
So the first question you might have is what data is available for us to use, right?
If you're going to do parameter efficient fine tuning for a model later on, what data is there for us to use?
Okay, let's explore the data set.
As mentioned, we're going to use SQL.
So we're going to use SQL to query the data warehouse.
And what we can do is we can use SQL also to go through the data set and to return all of the tables available within this data set.
We can select the table names from BigQuery Public Data Stack Overflow and we can return the information schema and the table names.
So we only want to return the table names.
In order to send this query to the data warehouse, we're going to use the client that we just initiated.
So the BigQuery client takes in the SQL query, sends it to the API of the data warehouse and we'll return the data set in our notebook instance.
So next what we're going to do, we're going to go through the results and we're going to print each of the row, which is a table name, so that we can see which tables are available.
So for row in the query job that we have here, we're going to print every value in the that we have.
If we run that, we can see all the tables that are available.
So there's a whole lot of them.
So we have answers, we have comments, we have questions, we're going to do parameter efficient fine tuning and we're going to do supervised fine tuning.
So we want to create a data set that has a question and has a answer.
Okay, next let's retrieve some data from a table, meaning let's fetch some data from the data warehouse and visualize it and print it in our notebook.
We'll load the result of a query in a pandas data frame so that we can look at the data.
Again, we have to write a query from this query.
We're going to read all the columns from our stack overflow data set and we're going to read from the table post questions.
So you see the table up here post questions and I'm going to limit it as well.
We don't want to retrieve all of the data.
I'm going to talk about that more, but first let's run this query.
As mentioned, we're going to load it into a pandas data frame.
So pandas, who doesn't love pandas and then we're going to execute the query again just like we've done before.
We're going to take this equal and send it to our data warehouse.
The next code will take the result of our query in a data warehouse and we'll create a pandas data frame.
This approach is valuable because it allows you to work with the results of a query job in a pandas data frame,
after which you can use all the pandas magic to explore your data.
So the result of a query from a data warehouse, we're going to load into an arrow table, which is part of the Apache framework.
This makes sure that the data is in a format that makes it easier to read for the pandas data frame and that lets us use the pandas data frame to explore our data further.
That was pretty quickly.
Here we see three rows, three data rows from our tables.
We can also see the columns.
Go ahead, have a look at the data that's available here.
This is only one table and I did a limit, so I didn't retrieve all of the data.
Dealing with large data sets and this is not uncommon working with large language models.
You often have to deal with these large data sets and they often don't fit into memory, especially if you're using a local notebook.
So let's say you want a query and return all of the columns and rows of the table post questions.
So we're not going to limit this time, so we're going to use the same query, but we removed the limit.
So we're going to retrieve all of the data available in the table.
So we're going to send this again to our data warehouse and let's see what is going to happen.
So the next code will create a pandas data frame just like we've done before.
Yes, this will take a bit longer.
Oops, we're getting a 403 response.
This data, if you read here, is too large to return.
So what happens is that you can fit a lot of data in your data warehouse, but not always will fit that data into the memory of your local machine or your virtual machine or your container.
So how are we going to deal with these large data sets?
Okay, so when data is large, we can have an issue with memory.
So what is a best practice or what is a blueprint we can follow when dealing with large data sets?
When your data sits in a data warehouse and when you have two large tables and you want to do things like joining or filtering,
it's best to do this processing through something like SQL in your data warehouse in order to deal with these large data sets instead of exporting all of the data and then doing the work in bandas.
Another thing to think about is that once you've selected your data and you know what you want to use for training,
is that you can export that result into something like a solid state drive or a cloud storage bucket.
Why is this? Is that if you do tuning or training at skill, it's important that you can access your data, your training data quickly and fast.
Meeting this data can be done faster when using an SSD or a cloud storage bucket or something similar.
You want to make sure your accelerators, if you're using something like GPUs, are not under utilized and waiting for your training data to come in.
Of course, also keep track of your data limits.
You want to be able to track where your data comes from and which transformations it has gone through.
Next, we need a solution to deal with larger data.
So when working with large data, you have to optimize your query to save resources and time.
In the following example, you will combine two tables to get a question and the answer that we need for tuning data.
We're going to use a wear clause that allows us to filter the results based on a specific condition, ensuring that only the relevant data that we need is returned.
This can significantly improve performance, especially when dealing with these large data sets.
Let's get our query.
I'll talk you through it in more detail so you understand what we're actually returning from the two tables.
First, a select statement.
We're going to select data and we're going to select two columns.
We're going to need a question and we're going to need a answer.
So we're going to take the title of the question plus the question itself and we're going to name our column input text.
We're going to take the answer and we're going to call this output text.
As you probably know by now is that we're going to read our data from the public data set stack overflow.
And we're going to read from post questions and we're going to join that with our answers.
So adding the answers and we're going to join on accepted answer ID.
So in our table, there's a unique identifier and we're going to use this to join the data from the two tables.
We're matching our join on a unique identifier.
Our table with questions has a unique ID.
So each question has a unique ID and that same unique ID is with the also in the table with the answers.
So the question and the answer can be joined using this unique ID.
And then we're getting to the where clause.
So we're not going to use all of the data that's in our data set.
So we want to make sure there's an answer in the row.
We're also only going to use questions that are about Python.
We love Python and let's focus on Python for each use case.
And we're going to use a in-aware clause a date.
So when you're dealing with data that's over time, you often also want to use that time variable to filter and select data from a certain time period.
And we're going to limit. We don't need all of the data in order to do tuning for a model.
Let's go with 10,000 examples.
So we have a where clause.
We have a limit.
And we're joining the data from two tables.
This query will return a result that lets us create a data set that we can use for our tuning.
Okay.
So we've written our query.
Let's send it to the data warehouse just like you've done before.
And the result, as mentioned, will be a pandas data frame.
We now have a data set with questions and answers.
Fintuning language models on a collection of data sets phrased as instructions has been shown to improve model performance and generalization to unseen tasks.
An instruction refers to a specific direction or guideline that conveys a task or action to be executed.
Basically, you're telling the large language model what to do.
These instructions can be expressed in various forms.
You can set a rule.
You can say step by step what the large language model needs to do.
You can have a procedure or an example.
When we don't use the instruction, it will only be a question and an answer in our data set.
The model might not be sure what to do with the question and answer.
So the instruction tells the large language model what to do and what we expect from them.
We want them to answer the question.
We have to give them a hint about the task we want to perform.
Let's extend this data set with an instruction.
We're going to use an instruction template and this instruction template has an instruction for the large language model.
So what we have here is we say please answer the following stack overflow question on Python and answer it like you are a developer answering stack overflow questions.
And then after the stack overflow question piece, we want to have the actual question from our data set, the input text.
So we want to combine the instruction and the question to include it in our data set.
We'll create a new column that will combine the instruction template and the question input text.
We're going to use a new column because we don't want to override the existing one which you might want to use later.
So we're going to take the instruction template and combine it with our input text and create a new column called input text instruct.
Next you will divide the data into a training and evaluation set where evaluation will be used as unseen data during tuning to evaluate performance.
We're going to use scikit learn, train test split in order to split the pandas data frame into a train and evaluation set.
The data is divided into training and evaluation with 80 20 split by default.
We want to have a bit more data for our tuning play a bit with this and you can change it to whatever you like.
We also going to use a random state parameter to initialize the random number generator.
We want to use random sampling to make sure that we do a fair comparison of our model.
Let me make an important point.
So keep your parameters as consistent as possible across your experiments so that you're able to do a fair comparison with experiment meaning when you train your model or when you run your end to end workflow from your data to know your model tuning.
If you change too many parameters or let's say you change your training evaluation split from an 80 20 to a 60 40 and you update a whole bunch of hyper parameters and maybe your new experiment has a better model or a model that's worse.
You don't know what has impacted this.
If you change one parameter or maybe two, it's easier to keep track of what impacts your model performance.
We're not going to calculate something like accuracy.
Why we are not calculating accuracy is because we're using text and text is very ambiguous and we can't calculate an accuracy over this text use case.
As we talked about before, if you have a large data set in your data warehouse, one of the best practices is to get this data out of your data warehouse and store it in files on either an SSD or something like a cloud storage bucket so that you can read it efficiently during your training or tuning.
Now the question might arise.
File format, which file format I'm going to use for my training and evaluation data.
Well, there's many options and I want to talk about a few of the key ones.
So this example we're going to use a JSON line format.
JSON L format is a simple text based format where each question and answer will be a row.
It's very human readable and it's an ideal choice for a small to medium sized data set.
If you have larger data sets, you can use a binary file format like a T of record or a part K file where a T of record, a tensed flow record is easier to read for computers, making it ideal for efficient training.
K files are also efficient for reading when doing training or tuning and it's a good choice for large and complex data sets as well.
So basically these two, it's up to your preference and what model framework you're using.
When building an ML ops workflow, it's also important that you think about versioning your artifacts.
You could imagine when dealing with tables in a data warehouse and files for training that we just talked about, you have all these different artifacts in your ML ops workflow.
It will be important that you keep track of the different artifacts and you also do versioning of your data.
So one example you might want to know from which data set from your data warehouse you generated your data file.
So that you have traceability, but it's also important for having reproducibility and maintainability.
And of course you want to make sure your colleagues can also understand what's going on if they have to take over some of the work or have to help you.
So in this case, we're also going to version Jason L file that we're going to generate.
So I'm using a description where I'm using the type of data set.
So if it's a training or evaluation, when you use the name of the data set and we're going to use a time stamp.
So when did we generate the data?
So I'm going to import the library date time and I'm going to use today's date and time as a time stamp.
So now it's time to generate our Jason L file that we're going to use for our training in our Jason lines file.
We're going to only use two columns input text instruct so do column we just generated an output text, which is the answer.
What we're going to do is we're going to take the data frame converted into a Jason L file where each record question answer is on a separate line.
This is controlled through the orient, which is set to records.
I mentioned that it's important to do versioning and keep track of the files that we generate.
We're going to use the name of the data set stack overflow and that it's a question and answer problem.
And we're going to use the date to make sure that we can keep track of when this file was generated.
Then it's time for us to generate the file.
We're going to first generate a file for our training data.
We're going to use the file name that we just described.
We're going to write it to the local directory.
That's practices, of course, is that you write your file into the SSD environment or a cloud storage bucket if you're using large files or if you generate many files that you want to keep track of.
But for this example, we're going to sort locally.
You can now find the file in your local workspace, go to file and then open.
You can have a look at the file.
It's okay to now pause the video and make some changes to the lab yourself.
Maybe you want to do versioning in a different way.
Maybe you want to go from year to month, today and hour.
So that you're able to sort of the files on year.
Also, if you had a look at the JSON file and you see how it's structured with each question and answer on a line, you might want to remove the orient equals records and lines set to true and see how the file changes and how the structure of the JSON changes where not each question and answer is on a separate line.
In the next lab, we're going to talk about automation, orchestration and parameter efficient fine tuning.

## Automation and Orchestration with Pipelines
Now that you have your data, it's time to tune a LLM.
Keep in mind that you might need to tune a model multiple times to develop one that works for your use case.
Experimentation is more critical when it comes to LLM's because we are still learning wall works.
You want to automate the process of running experiments to make your life easier.
In this lesson, you will learn how to build a machine learning pipeline to tune and deploy your model using an open source framework.
Okay, before we go into the code, let's talk a bit about ML ops workflows for large language models.
I've talked about system design for LLM's and I've also talked about these workflows that are more one-dimensional.
Let's say you want to train or tune a model.
The process you typically go through is you get some training data, you do training, you do evaluation,
you do evaluation during the training or tuning process as we talked about in the previous lab.
And then as a result, you will have a trained model which could be something like a TensorFlow saved model format.
So once you have a trained model, what you will do is you will take your trained model and you will put it in your production environment.
Because in the end we want to ship our model and integrate it with our use case.
Once you have your trained model in your production environment, this could be a rest API or a batch process is that you take production data
and you generate predictions.
And again, you do evaluation as often as needed.
And based on your evaluation, you can of course update your training data and run your workflow again.
There's a few key things here that I want to talk about.
There's orchestration, automation and deployment.
Orchestration meaning is that you specify which step needs to be run first and then what is the next step, so on, so forth.
Automation meaning is that you automate this workflow.
This helps you, for example, that if you want to train a new model, you can rerun the end to end workflow again.
And deployment I talked about, this means it's taking your trained model and putting it into your production environment.
Okay, so just about orchestration, it's about orchestrating the sequence of steps.
Where automating is about making sure that you as a person don't need to run the script yourself, you automate the execution of the code.
For example, if you have multiple Python files, you don't want to manually go in and execute maybe your data preparation and then the file for your model training.
No, you want to orchestrate them. You want to say first the data preparation and then the model training and you want to automate that process so that you don't need to execute one file after another.
Let's dive into the code. For orchestration and automation, you can use different frameworks, for example,
for example, a bunch of airflow and Q-fold pipelines are both popular for building machine learning workflows that help you with this orchestration and automation.
This notebook we're going to use Q-fold pipelines. So we're going to import the Q-fold pipelines package.
Q-fold pipelines is a open source framework and you use it for constructing like a kit for building machine learning pipelines to make it easy to do orchestration and automation.
So first I'm going to import the DSL. So the DSL we're going to use say we're going to use like a drawing board to sort of design our pipeline.
And then also we're going to use a compiler. I will talk about that later.
Okay, so let's say we want to orchestrate a very simple example first because I want to teach you about orchestration and automation.
So let's say we're going to take a Python function that is say hello, that will take a name, which is going to be a string.
And then it's going to take a name as inputs and going to add hello, so hello, and then my name, so hello, and then this is going to return hello text.
Okay, so important here is that Q-fold pipelines has two key concepts.
There's components and there's pipelines. A pipeline is like a self-contained set of code like this code that we see here.
And this is one step or could be one step in your workflow to map this to machine learning workflow.
So the first step could be data processing or data engineering. The second step could be model training. So each component is a step in your workflow.
In the role of machine learning, we often talk about pipelines, but building and managing these pipelines can become complex and time consuming.
That's where Q-flow pipelines comes in. A powerful tool that helps us automate and orchestrate our ML ops workflow, but Q-flow doesn't speak plain English.
Also, it doesn't speak Dutch. It uses a specialized language called a DSL DSL stands for domain specific language.
Think of it as a set of instructions tailored specifically for building ML ops pipelines.
The open source framework Q-flow pipelines provides its own DSL library, which lets you define your pipeline steps and execution logic, also referred to configuration in a clear and concise way.
No more juggling code in different languages. With the KFP pipeline DSL, you can focus on the what of your pipeline.
The tasks you need to perform and let Q-flow take care of the how Q-flow pipelines DSL we can use for building ML and pipelines.
So how do we make sure that this code, this pipeline function, that we know that this is a component in our machine learning workflow?
This is where the DSL comes in. So we're going to use the DSL to tell the system that this is a component.
So this code will run in a containerized environment. You may have not heard about a container.
So containers are like a contained environment that have your dependencies and your software code.
The advantage of containers is that you don't need to manage a server, install your OS, install your dependencies there, and your software on the server.
It lives in this bubble where you have an OS that's already available for you.
You install only the dependencies that you can use for your software. Put your software in the container.
And then you can take the container and maybe come somewhere else. You can execute it on environment A or B or C. It doesn't matter.
You're not dependent on the hardware underneath. In an ML workflow, we often want to orchestrate multiple steps.
So we have our first step, which is a component. Let's now write another one.
Again, we're going to take a Python function called how are you?
Which is also going to take a string as input. So we're going to take hello text.
We're going to use the output of the first component as input for the second component.
So that means the second component is dependent on the first component.
So we have to run the first component as the first step and then the second component.
Because it's dependent on the output. So the output goes downstream.
So let's say your first step generates files that contain training data like we've done in the previous lab.
The second step might be training a model that takes these files to train the model.
So it takes this data to train the model. And how you pass data from one component to another can be done in different ways.
So you can say the first component hands over the data files to the second one and then sell on so forth.
But typically what you do and the best practice that we see a lot is that you pass the path to the location of files into the next step.
So let's say you stored your training files as we discussed in the previous lab in a cloud bucket.
What you do is you tell the second component, hey, the training files are in this bucket instead of passing the training files into the component.
Why that is important is that if you're dealing with large data, you don't want to put all of the data in one step in one container.
Because then you're dependent on that memory and we talked about that. We want to build our pipelines in a scalable way.
So you pass the uniform resource identifier like the path example I just gave for this Python function.
We also need the DSL decorator, so the component decorator to tell the system that this is a Qflow pipeline component.
So now we have two components. Now it's time to draw a workflow or to sort of decide how we're going to execute these steps, these components.
We're going to build our pipeline. So for this, again, it's just a Python function and let's call it hello pipeline.
And this one is going to take a recipient again, a string and within this function, we're going to tell the system which step we want to execute first.
As discussed, our first Python function, our first component is the one we're going to execute first.
So this is the say hello component. And this is the one we're going to execute first. And this is going to take the recipient, which is a string.
So the name of a person as input and injected into our component when we're going to execute this.
The second step, and you probably guessed it already, is that we're going to execute our second component.
How are you? And this component is going to take the output of the first component as input.
So the hello text is going as input in our second step. So here we're telling the system, the output of hello task should go as a parameter in the how are you component.
And then we're going to return how task outputs.
So the output of the second step, now we have to tell the system that this function is our pipeline.
So our sequence of steps, we're going to use the decorator to tell this is a pipeline.
Next, we're going to implement our pipeline. A pipeline is a set of components that you orchestrate.
A pipeline also lets you define the order of execution and also lets you tell how data flows from one step to another.
So basically data can go downstream in a pipeline.
Okay, remember when I talked about importing the compiler. We now have specified one or components to our pipeline.
Now it's time to compile our pipeline in a file that you can execute. For this, we're going to use the compiler.
The compiler lets us generate a YAMO file. A YAMO files are human readable, used for storing your configuration and the sequence of steps of your pipeline.
The YAMO file also contains information on your components, their dependencies and the order of execution.
So last thing we need to do is that as you remember, we don't have a name yet.
So a name needs to go into our pipeline when we execute. We're saying hello name, how are you?
So we need to define our arguments. These arguments are injected into our pipeline during execution.
This is a pipeline argument and here we have the recipient and we're going to use world.
Once you've generated the pipeline of YAMO file, you can have a look at the YAMO file.
So let's run cat pipeline with YAMO and you can see for example here the components that we specified.
There's all sorts of other information that's related to the pipeline execution and dependencies as well in this YAMO file.
Let's continue for now. We now have a YAMO file. Once we have a YAMO file, you can execute it on different environments.
We've talked about Kubernetes. When you run a Kubernetes cluster, you have to manage it and operate it in production, which is a lot of work.
You can also use a managed environment like vertex AI pipelines that executes the YAMO file for you in a serverless environment.
For this, you can use the Cloud AI platform library and import pipeline job.
Next, you have to specify your pipeline job. There's a few things that we need to set.
First of all, we need to say to the system that we're going to use the pipeline YAMO for execution.
Secondly, give it a name. Then we have the pipeline arguments. These arguments go into the pipeline during execution and then we specify a region.
We talked about regions in the previous lab. The pipeline route is where temporary files are being stored by the execution engine.
Then it's as simple as job.submit and you can run it. For classroom restrictions, we're not going to execute the job now, but you can take the code and execute it in your own environment.
You can execute this on Google Cloud using vertex AI pipelines, which is a serverless layer on top of Kubernetes.
We don't need to worry about managing machines like VMs or a Kubernetes cluster.
We just sent the YAMO file and Google Cloud takes care of the execution. Vertex AI pipelines.
This is where I see the execution. We have the two steps, the two components that we just created.
Say hello and how are you? Here you can also see the parameters. You can see that type and the value.
If you go to the second component, we can also see the output. It says, Hello World, how are you?
This is where you typically see the execution of your pipeline, but also see the parameters and the output of the components.
We just built a Hello World example of the pipeline. You're now familiar with the concepts.
Let's now look at a real-life example of a machine learning workflow, a machine learning pipeline.
The advantage or one of the advantages of a pipeline is that you're able to reuse it.
Once you build a pipeline, you can reuse it so maybe you can share it with a colleague.
Let's say you've built a Q&A language model and you build a pipeline that processes the data, trains a Q&A language model and then outputs a trained model file.
Maybe your colleague also has a Q&A use case. Your colleague comes to you saying, Can I reuse your pipeline for my use case?
Reusability of pipelines is an important advantage.
In the next example, we're going to reuse an open source cube for pipelines that lets us do supervised fine tuning of a foundation model from Google called POM2.
The advantage of reusing a pipeline is that we don't need to build it from scratch. We only have to specify some of the parameters.
Remember that we generated two files in the previous lab. So we have our training data and we have our evaluation data.
The JSONL files we talked about. So we're going to use these two files to fine tune a POM model.
We're going to use these two files to do supervised fine tuning. So reusability of pipelines is a big advantage.
And as I mentioned, we're going to use an open source pipeline that's part of Qflow. Here you see the path through the pipeline file.
So remember the YAML file we generated. This is where we can find the YAML file for this pipeline.
In order to execute the pipeline, we need to set up some configurations. Basically we need to specify our arguments.
First of all, let's come up with a model name. In the previous lab, we talked about versioning and how important versioning is.
So here as well, we're going to use a date timestamp to keep track of the model that we generate.
We have to specify a model name and we're going to add the date to it. So at least we can always go back in time and see which model was trained with.
There's also some other parameters that we can tweak. I want to highlight two. So first of all, we have training steps.
Training steps, meaning the number of steps to use when tuning the model. As mentioned, this pipeline was parameter efficient fine tuning.
A single training step is the process of feeding a batch of data into the model when you do your training. It updates the internal parameters and it will calculate the loss or error.
Do not confuse it with e-box, which is a full pass of the data set. For extractive Q&A, you can set it between 100 and 500.
This is the best practice for this Palm model. For now, I just chose the number 200. And as you remember, when you run a pipeline, see it as one experiment.
So you can tweak a parameter and run it again and see how it impacts the performance of the model. Also, we're going to set an evaluation interval.
We generate an evaluation data set, the interval that specifies the frequency at which a trained model is evaluated against the evaluation set that we created.
Default will set to 20. You can play around with this as well. Just like we've done for our hello world pipeline, we have to specify arguments that the pipeline will use during execution.
Two of the arguments that we're going to use are a project ID and a region. Just as you remember from first lab, this is that we have to specify the project ID and the region where we're going to run this job.
Again, we're going with us central one. Next, we're going to specify the arguments. The arguments are being used by the system during runtime. So when the pipeline is executed, the arguments are ingested into the pipeline.
So the pipeline arguments for this open source pipelines are our model name that we just specified. The region where we're executing the workflow and, of course, the model that we're going to use for our parameter efficient fine tuning.
This is a palm family model. It's called text bison, which is a great model for question and answering. 001 is the version of this model project ID.
The training steps that we just specified that are set to 200, where our training data sits, our evaluation interval, and our evaluation data you write. So where are evaluation data sets?
These arguments go into our pipeline when it's being executed. You can imagine when you create a pipeline, you can include arguments.
Arguments can be changed. So that means is that when you have the pipeline, you can run it with a different set of arguments.
So let's say you have a new data set that you want to use. You just have to update the argument to the data set. Or maybe you want to use a different type of model for the tuning.
You just update that parameter. So it makes it very easy to take an existing pipeline and to make minor changes to the pipeline and rerun it.
Next we also have to specify our pipeline route. As you remember, this is where the artifacts are stored during execution.
And then we specify a pipeline job, which tells the system how the pipeline should be executed with which arguments.
The enabled caching to true means is that if we're going to rerun the pipeline, if we use caching, which means is that if we already executed a step previously, it will not execute it again.
Only if we updated the code or one of the arguments, it will rerun that step. Then it's time to do a job submit.
We're not going to execute this pipeline now because if we have to run it, it might take a whole day to execute. Also, it's quite expensive to run this pipeline because you need accelerators like GPUs or TPUs.
In the next lab, we're going to talk about deployment. So once we tuned a model, of course, we also have to deploy it in order to use it in our LLM system.
And see you in the next lesson.

## Prediction, Prompts, Safety
So now you will actually get used to choose model to make predictions.
The team has deployed one for you.
But there are still more to do in order to integrate this safely into a real-life application.
We will need to make sure the model will work the same in production,
but also that it can be safer and more responsible.
So you will first want to make sure production data has the same format as the data used in training.
You look into the response and how to get insights on safety attributes
and are there any sources that the response is based on?
Okay, welcome to the last lab of this course.
Today we're going to talk about predictions, prompts and safety scores.
Just like with the other labs, we have to run some of our setup code.
Again, we need to get credentials and our project ID.
We're going to use the same region as we did before.
We're going to use US Central 1.
And also in this lab, we're going to use the Vertix AI SDK.
We're going to use the text generation model that we just tuned.
Let's also initialize the Vertix AI SDK right away so that we can use it
with the project ID, region and credentials.
Let's talk about deployment.
In the previous lab, we talked about building an end-to-end machine link pipeline
that takes training data, chooses a model, does evaluation,
and then generates a trained model that you can deploy.
As we talked about, deployment can mean different things.
Take your model, doing batch predictions,
or it can mean creating a REST API that you integrate with your services.
When deploying your model, you have multiple options,
meaning the way you deploy your model and integrate it into your use case.
The two main ones are batch and REST.
I'll explain both of them.
Batch meaning is that you take your trained model
and you might have a batch use case.
So let's say you get customer reviews.
Maybe about products, and every week you want to score those reviews
if they're positive or negative.
So there's no need to do this in real time.
You maybe want to do it once a week and share it with the marketing people.
You can take your trained model and take all of the reviews,
the product reviews for that week,
and then send them to your trained model and run a dot predict on all of these reviews.
So what you're going to do is that for each of these reviews,
you're going to send it to the model and get a prediction if it's positive or negative.
So it's an offline process that doesn't need to be in real time.
And you batch all of the examples and you run the predictions in batch.
A REST API means is that we take our model and we deploy our model as an API.
This means that it's online and you can access it from a service.
So let's say you have a chat application that lets you stack overflow type of questions.
So when I come into the application, I ask a question.
The request goes to the API, the API.
So the model behind the API does a prediction and the prediction goes back into the user interface.
And I see the result.
So this is a more real time use case.
You need to get the response with a low latency.
There's different ways you can deploy a model as a REST API.
If you have, let's say, a TensorFlow model or a PyTorch model,
you might want to package it up using the Flask library or FAST APIs.
These libraries help you with packaging these models as an API.
Then you can take that API and put it in the container that we talked about in the previous lab.
If you're not familiar with Flask or FAST API, don't worry.
You don't need to know to finish this lab.
In this case, we deployed the model as an API.
So we've taken the trained model, package it up in a container as a REST API.
You can take the production data, send it to the API and get a prediction.
Let's go back to the code.
In order to call our API, we have to run a few lines of code.
We want to retrieve the endpoint of our model.
With endpoint, I mean the REST API.
So the model has been packaged up and deployed as a REST API on the cloud platform.
In order to get a response, we have to specify which model we want to use for predictions.
Remember that we tuned a text generation model, the text by some model.
This was one of the parameters used set in your pipeline.
Using the SDK, we can retrieve a list of models that are deployed.
Let's use the model list tuned model names to retrieve the endpoints of the tuned models that we have.
We ran the pipeline a couple of times.
So we have multiple endpoints.
Let's print out the endpoints that are available for us.
As you can see within our project, we have three endpoints that live in the US Central One region that you specified.
We deployed multiple models to make sure we can spread all of the traffic over the three models.
We're going to use a very basic form of load balancing to choose one of the endpoints to send our prompt to and get a prediction.
So we're going to use the random library.
And we're going to choose from our list randomly one endpoint that you're going to use to get a prediction from.
Once we randomly selected one of the deployed models, we can load this model so that we can use it to get predictions.
Getting predictions means we're going to send the prompt to the API and get a response from the model based on the prompt.
So we're going to load one of the models.
And next we can write a prompt.
So let's write a prompt.
And since we've done tuning on the stack overflow data set, I think we should ask it a Python question because we selected all of the Python questions from the stack overflow data set.
We want to stay as close as to what we trained on as possible.
And I'll talk about this later as well why this is important.
Let me ask a straightforward question.
How can I load a CSV file using pandas?
This is something we've done in the first lab.
We have our prompt.
We've loaded the API.
Now we can send our prompt to the API and get our response.
For this we can use the vertex AI SDK.
And we're just going to run a dot predict on the deployed model.
And we take the prompt into the dot predict.
So the prompt goes to the API.
Of course large language models are large.
And depending on the size of your model, depending on the size of your prompt, latency can differ for your model.
Meaning is it can differ how long it takes before you get a response.
Larger models typically take longer to get a response.
If you have larger prompts, it takes longer to do a prediction.
So depending on your use case, that can be different latency.
Let's first print our response and see how it looks like.
Well, looking at this, you can see that it is not very readable.
This is a lot of information we're getting from the model as a response.
So let's clear this for now and let's do some formatting.
For this we're going to use pretty print.
So from pretty print, we're going to import pretty print.
And let's unpack that response for a bit.
So when you have an API, you can send multiple prompts to get multiple predictions from the API.
We only send one prompt.
So we're only getting one response.
Let's have a look at that.
And let's use the pretty print to format this response.
Since we only send one prompt, we're only getting one response.
This response consists of a lot of information.
We have citations with scores.
I'll talk about citations and safety attributes later on.
Let's unpack the response a bit more and look at this exact object.
If we print this output, you can see that it only has one object.
In the output, you can see that there's multiple keys.
We have content, safety attributes.
Content is the answer, the answer on the questions that we asked.
So basically the response on our prompt.
Let's now extract the content.
So the response on our prompt basically answer to our question from the output.
We're going to use content as a key to extract response from the model.
Let's now print the final output, the answer to our question.
Here you can see the answer to our question.
So this is a response of our model.
Okay, so we talked about the different options of deploying a model.
Remember, batch and rest API.
When deploying models, there's also other things you have to take into account.
And other things that are important when running production machine learning workloads.
I've talked about packaging, deploying and versioning your model.
But there's also other things that we have to take into consideration by doing model monitoring.
There's different ways you can monitor your model.
First of all, there's the operational metrics.
Let's say how often do you send the prediction to the API?
Of course, you also want to evaluate the performance of your model in production.
So how is my model performing?
What's the evaluation metrics of this model in production?
Maybe you want to generate some scores.
Also, we want to consider safety.
So is there any bias or any harmful language in the output of the model?
And we'll talk about this later.
Also, you want to think about scalability.
You might want to do a load test on your API,
meaning sending a lot of predictions at the same time to see how your model behaves or how your API behaves.
And then when rolling out this in production, you want to do this slowly.
So let's say maybe you first want to have only 5% of your traffic going to your model instead of all of your users.
We call this a controlled rollout.
Also, discuss with all of your stakeholders what is a permissible latency?
So when you discuss with your stakeholders about what is permissible latency?
Have a discussion about what's the actual number?
For example, is two seconds of latency okay for a user experience?
So let's say going back to the Q&A system that you're building for stack-oval questions,
you have a UI, is it okay if the user waits one, two, three seconds,
or do you want it to be like a hundred milliseconds?
All of this is something you discuss with your stakeholders,
and you decide on what is permissible.
But what can you do if your latency is too high?
There are multiple options you can explore.
You might want to choose for a smaller language model.
Smaller models often have lower latency.
You can also explore deploying your language model on GPUs,
or maybe you'll deploy your model in a different region.
So let's say your application runs in the Singapore region,
you can deploy the model also in the Singapore region,
so that the model and the application are close together.
But hold on, didn't we in the previous lab did some data processing
and create a new feature where we combined an instruction with a question?
In the example we just ran, we only send a question to the API.
This means is that the model that we trained was trained on data
that has an instruction and a question.
But now in production we're only sending the question.
So there's a mismatch what we trained on,
versus what we had available in production.
There's a skew between the two.
It's very important that your production data
is very much the same as the data that you trained on.
If there's a difference between the data,
it can influence the performance of the model.
So we need to add an instruction to our question
before we send it to the model.
In other words, we want to add the same instruction
that we had in our training data.
Let's do that first.
Let's take the instruction that we created earlier.
And then take a question.
Let's go with a new question.
So we have the instruction.
We have the Stack Overflow question.
Now we want to combine it to and create a prompt.
So we have instruction, question, and taking it to
we're going to generate a prompt.
If we print the prompt,
we can see that now our instruction and question are combined.
So we have our instruction.
And we have our question.
We can now take this new prompt that is consistent with the data
that we trained on and send this to the API.
We'll do this in the same way.
We'll get a deploy model to a product
and take the prompt that we just created.
From this response,
let's use the content key again
to get the answer on our question.
Let's print this output
and have a look at the response of the model on our question.
We see that the output,
the response of the model has a very nice structure
and it answers our Stack Overflow question
like it's a developer.
So remember when we had the output of the model,
all of the output.
In this object,
we also have safety attributes.
The thing with large language models is that they can generate output
that you don't expect,
including text that can be offensive, insensitive,
or maybe factually incorrect.
What's more, the incredible firstility of large language models
is also what it makes difficult to predict exactly what kind of unintended
or unforeseen outputs they might produce.
So it's important for you as a developer practitioner
to understand and test these models
to make sure you deploy them safely and responsibly.
This is where we can leverage the safety attributes of this model.
The Palm API that we tuned has safety attributes scoring
to understand the output
and define confidence thresholds,
meaning that you can take these safety attributes
and define your own thresholds.
I'll talk about that later on how you can use that
and how you can think about these thresholds.
Let's unpack this a bit more and write some ago.
So from a response,
we're going to use a different key now.
We're going to use the key safety attributes.
And from the safety attributes,
we can first check if there was anything
that was blocked by the responsible AI service.
So in our response,
we have safety attributes.
We can also check if the response was blocked.
What you often see with model providers
is that before returning the response to you
as a developer or practitioner,
there should check if there's an issue with the safety attributes.
If there is, then the response might be blocked.
Let's check if it's blocked.
So we're going to print the blocked.
Well,
a second layer of checking
if there's any issues with safety attributes
can be done by you as a developer.
In the response,
you will find probabilities for each category.
And you can use these probabilities
to design your own threshold.
By using these thresholds,
you can take comprehensive measures
to detect possible harmful topics.
So the confidential scores are predictions only.
And you shouldn't depend on the scores only
for your reliability or accuracy.
It's one of the measures you can take.
So the second level
is where you as a practitioner
and the developer
can use safety attributes and safety scores
to set up your own thresholds.
Let's have a look at the safety attributes
to get a better understanding of them.
So from our response,
we're going to use the key safety attributes
to retrieve the safety attributes and the scores.
We're going to use pretty print again.
To print our safety attributes.
Okay.
What you see here
is a couple of things
that I would like to talk you through.
First of all, you can see
if the response was blocked
or not.
In this case, it wasn't.
Then we have some of these safety categories.
Like finance,
politics,
we also have the safety ratings
with different categories.
You can also see here
the probability scores.
So each safety attribute
is associated with a confident score,
like a probability score between 0 and 1,
and is rounded to one decimal place,
reflecting the likelihood
of the content being unsafe.
So it scores for each category.
You can see here
that the probability score
for most categories is 0.1.
I mentioned that the probability score
can be between 0 and 1.
So 0.1 is very low.
You see also here
a severity score.
Safety filter confidence scores
are based on the probability of content being unsafe
and not the severity.
This is important to consider
because some content can have
a low probability of being unsafe,
even though the severity of harm
could be high.
In the output you've seen
there's probability and severity.
Let's talk about
what is the difference
between the two using two sentences.
The first one, the panda pinned me down.
The second one, the panda beat me.
The first example can have
a higher probability of being unsafe
is because the panda has pinned me down.
Where the second sentence
is more ambiguous
and we don't know if it wasn't
unsafe situation.
I might have been racing the panda
for some bamboo
and the panda beat me in the race.
But the second one could be more severe
because it also can mean
that the panda has beaten me,
punched me multiple times.
So as a user,
you can leverage these scores.
The probability score,
the severity score,
to set up your own thresholds.
For example, if the probability score
is 0.5,
you can decide if that's
an acceptable score for you
or if you want to filter out that response.
Deciding what is a good threshold
is up to you.
And it depends on your requirements,
type of users you're working with,
and other requirements.
There's not one probability score,
one threshold that could work for everyone.
It's very much their first per use case.
So you have to look at your own data
and the responses with your data
and decide if that's an acceptable threshold for you
or not.
When dealing with generative AI,
you want to produce original content.
And preferably not replicate existing content at length.
So you want to design your systems
to limit the chance of this occurring.
In the response, you have something
that's called citation metadata.
Meaning you can check if there's any citations.
From the model response,
we can take the citation metadata
and from citation metadata,
we can take the citations
to check if the response is citated somewhere.
Let's use pretty print again
to see if there's any citation in the response.
That's good.
The response is not citated somewhere.
If there is a citation,
the model will return a page
from where it's cited from.
Like a website,
we don't have any citation.
In this example,
try out your own prompt to see
if you can trigger a citation.
Maybe something from Shakespeare
to be or not to be.
Try a few prompts yourself.
I'll see you in the next video
where we will wrap up the course.

## Conclusion
Congratulations on finishing the course.
The field of LLM ops is rapidly evolving and there's always more to learn.
In the notebook of the previous lesson, you will find a link to a notebook that shows
you how you can run the supervised tuning pipeline yourself.
I'm excited on what you will build next.
