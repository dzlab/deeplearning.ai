# Automated Testing for LLMOps

## 1. Introduction
Welcome to the short course Automate Testing for LLMOps,
built in partnership with CircleCI.
Software Testing helps you identify bugs
and security vulnerabilities in your applications.
And Automate Testing frees up your time and energy further
so you can focus on the creative parts of designing
and building your application.
In this course, you learn modern software engineering practices,
focus on testing for the practical development
and deployment of LLM-based applications.
Two kinds of LLM evaluations that you implement in this course
are Ruby's evaluations and modern graded evaluations.
Ruby's e-vails use string or pattern matching,
for example, regular expression matching
and a fast and cost effective to run.
I use these whenever I want to evaluate outputs
to have a clear right answer,
such as sentiment classification
and if say you have ground truth labels.
Ruby's e-vails are quick and cheap to run
so you can run these tests every time you commit a code change
to get fast feedback on the whole of your application.
Modern graded evaluations are relevant for applications
where there are many possible good or bad outputs.
For example, if you ask an LLM to write text content for you,
there can be more than one high-quality response.
Here, you might prompt an evaluation LLM
to have it assess the quality of the outputs
of your application LLM.
In other words, you use an LLM to evaluate the outputs
of another LLM.
Modern graded e-vails take more time and cost more
but it allow you to assess more complex outputs.
Delighted to introduce our instructor for this course,
Rob Zuber, Chief Technology Officer for CircleCI.
Rob has spent decades leading engineering teams
and also hoping customers scale up their software
to re-practice by making processes repeatable, scalable,
and reliable.
He'll show you how to do this
but your applications as well with an emphasis on testing.
Thanks, Andrew.
In your software development process,
you and your teammates may commit code updates
or bug fixes multiple times per day.
In this course, you'll learn to set triggers
that automatically run your evaluations
whenever you or your teammates commit code changes
to the repository.
Your team may also release updated versions of the app
on a broader cadence, perhaps once every two weeks.
Before deploying to users,
you can also automate more holistic,
comprehensive, pre-release evaluations.
For per commit e-vails, you can include rules-based
evaluations because they're fast and cheap to run.
And for those pre-release e-vails,
it may be very helpful to use model-graded e-vails
to do more thorough testing before deployment.
By the end of the course,
you will combine per commit and pre-release e-vails
into an automated testing suite.
And for this course,
you'll design tests to detect hallucinations
in LLM responses.
Many people have worked to make this course possible.
I'd like to thank on the Circle CI side,
Michael Webster, Jacob Schmidt, and Emma Web
from deep learning.ai, Ed Issue,
and Edgemail Gagari have also contributed to this course.
The first lesson will be a quick overview
of continuous integration terms and technologies
that will use as the foundation
for building our automated LLM testing pipeline.
And when you finish this course,
that will be a real test event to your dedication
to building good applications.
Or if you're not sure how much you'll use these ideas,
you can still test the waters.
So let's go on to the next video to get started.

## 2. Introduction to Continuous Integration (CI)
In this lesson, you'll get a quick overview of the technology we'll be using to automate our LLM
evals. Continuous integration. We'll discuss what continuous integration is,
why it's important, and that will give us a solid foundation for the automated workflows
we'll build in later lessons. Let's dive in. So what exactly is continuous integration?
In a nutshell, continuous integration or CI is a development practice that revolves around
making small frequent changes to your software and thoroughly testing those changes with everyone
else's. In other words, you're continuously validating that your new features and updates work
as intended when integrated with those from the rest of your team. This method allows you to detect
and fix issues early before they become bigger problems that are harder for you and your team to
solve. Let's see how this works in the context of an LLM-powered application. Imagine you're working
on a cutting-edge virtual assistant application. With continuous integration, every time you or a
team member makes a code change, be it refining the LLM prompts for updating data integrations,
you merge that change into a central repository. Now here's where the magic happens. Once the code is
merged, the CI platform kicks in. It automatically builds your application simulating how it would
behave in a real world environment. Automated tests are run to ensure that your LLM produces
accurate and reliable results. These tests cover everything from basic functionality to more complex
issues like hallucinations or biased outputs. Why is this important? Well, think of it as having a
supercharged feedback loop. The CI platform provides near instantaneous feedback on your changes.
If an issue is detected, perhaps the model is generating unexpected outputs, you're alerted
right away. This rapid feedback loop allows you to catch and address issues early in the development
cycle. That way, buggy code doesn't make its way into other features your team members are building
or even worse get deployed to users. Now let's talk about the perks for you as a developer.
Continuous integration significantly reduces the time and effort spent on debugging and troubleshooting.
It's not that your test will never fail, but when a test does fail, you'll have actionable
information you can use to get back on track quickly and continue innovating. CI empowers you to
iterate quickly, experiment with new features and make improvements with confidence. But the
advantages extend beyond individual developers. For teams, CI fosters collaboration by encouraging
smaller, more frequent contributions. Conflicting changes are minimized, making it easier to manage
and resolve issues. The shared repository becomes a reliable source of truth and the automated
testing ensures that everyone is building on a stable foundation. This creates a culture of trust,
collaboration, and faster delivery of high quality software. Moreover, continuous integration sets
the stage for additional automation in how you deploy, monitor, and retrain your language models,
enabling your team to move towards a more agile and responsive development practice. For AI
developers, where innovation is rapid and constant, CI provides the structure needed to excel.
If you're interested in more details on the conceptual integration of AI and continuous
integration, we have a mini-series on our podcast, the confident commit that goes into this in great
detail with many fantastic guests. Throughout the rest of this course, you'll use Circle CI as your
continuous integration platform. As you explore different techniques for testing and evaluating your
LLM applications, you'll be able to trigger your automated evaluations and view the results
in the Circle CI user interface. All of the setup work has been done for you so that you can
focus on learning and applying the strategies used in the course. In a team environment, you'll
often have a CI-CD expert who is responsible for setting up and maintaining your pipelines.
But if you are interested in learning more about what's happening behind the scenes,
feel free to explore the configuration files provided in the project repository.
We've also included an optional lab at the end of this course that will give you a chance to walk
through a step-by-step process of setting up a workflow in Circle CI. While this only scratches
the surface of what you can do with continuous integration, using it throughout this course
will give you the knowledge and tools you need to start making automated testing a central part
of your development practice. These are skills that will benefit you for your entire career
and set you up to follow modern software engineering practices no matter where your interest
take you. All right, let's get started writing your first evaluations and setting them up to run
in Circle CI. See you in the next lesson!

## 3. Overview of Automated Evals
In this lesson, you'll practice setting up rules-based e-vails, which are fast and cheap to run.
So rules-based e-vails are suitable for the early iterative stage of development,
in which you'll run these tests frequently. Any time you add a feature or fix a bug.
Once you have your first e-vails set up, you'll have a chance to see them run in an automated,
continuous integration pipeline. Let's get started.
So we're going to start with an overview of automated e-vails.
Ultimately, we want you to come out in this set of lessons with a good understanding of how to build
software effectively, how to build high-quality software, how to be able to move quickly and get
great feedback on everything that you're building. And this is a thing that's well understood in
more traditional forms of software, but as shifting as many of us start to build LLM-powered
applications. If we compare traditional software to LLM-based applications, we can see a few fundamental
differences. Overall, from a behavior perspective, traditional software tends to be focused around
things that are predefined. We know what the inputs are, and we know as a result of any particular
input, what the output will be. And that's fairly straightforward to test for.
By contrast, in LLM-based applications, we know what the inputs are, but we get a set of possible
outputs. So it's more probabilistic in nature. Many of these applications are based on natural
language. This can be highly subjective. And if your application does things like summarizing,
then it's quite possible that there will be many good outcomes. Outcomes that are good enough
and would be considered correct, there are also many incorrect possible outcomes or outputs.
And so the approach to testing can be quite different. For example, if you were to prompt an LLM
to answer the question, is Italy a good place to take a vacation? One answer could be, yes,
you can go to Rome, go to Florence, you can go to Venice, you can also explore museums, you could
explore beaches. Another answer could be a single word answer. Yes, which one is considered better
will depend on your use case. It's also important to note that LLMs can produce
harmful responses. They might be toxic, they might be offensive, and so LLMs bring new challenges
to application testing compared to traditional software. To deal with these new testing challenges,
AI researchers develop the concept of evaluations or evals. To assess how LLMs are doing
at specific tasks. There are many common data sets for different tasks. Examples include
M&LU, Heliswag and Human Eval. LLMs are often tested on different data sets, so researchers
have a point of comparison between models. However, what we're going to talk about is building your own
application and testing for your specific use case. Okay, so as I mentioned, there's some standard
benchmarks. These are some examples, but these are not necessarily going to give you great information
about what works the best in your specific application and for your use cases. So we're going to
take the time in this lesson to start building the tools to evaluate when you're building your own
application. What's great about all of these benchmarks and the existence of evals to perform
these benchmarks is that we have a lot of the tooling in place that we can use to start testing our
own applications. And in many cases, these evals are run manually, which is great for some quick
feedback as you're building, but once you get to scale, once you're working with other team members
and continuing to work and move forward, you want to have a tool that's going to keep checking
the quality of your work and keep checking the quality of your application again across larger
team, across time, and that's what we're going to build to over the course of this set of lessons.
So once we have the ability to perform automated evals, it's important for us to understand
what we're looking for and when we should be performing these automated evals. In terms of what
there's four main areas that we think about. First, context adherence or groundedness,
which is the question of whether the LLM response aligns with the provided context or guidelines.
Next is context relevance, and that's the question of whether the retrieved context is relevant to
the original query or prompt. Next, we have correctness or accuracy, which really does the LLM
output align with the provided ground truth and the expected results. How close is it to what we
would anticipate in the given scenario? And then finally, we're concerned with bias and toxicity,
which are really negative potential that exist in the world of LLM-powered apps. First, bias should
be favoritism or prejudice towards or away from certain groups and then toxicity, harmful words
or implicit wording that is offensive to certain groups. And then in terms of when, in a
traditional software model, we're typically testing after every change, whether that's bug fixes,
feature updates, or even data changes or changes to the model, which might be a different case
when you're evaluating from an LLM perspective. If that becomes slow or takes too long,
you can also start to pull out more comprehensive testing to specific points like pre-deployment,
so doing more comprehensive testing at the point that you're looking at pushing something into
a production environment or post-deployment because some of these changes might occur once your
software is actually executing in a production environment, which is again a bit of a shift from
our traditional software models. Okay, so let's get started putting all of this into practice.
In order to start understanding this framework, we're going to build an AI-powered quiz generator.
The app will have a data set comprised of facts categorized across art, science, and geography.
The facts are grouped into specific subjects. Some of those subjects apply to multiple categories,
for example, Paris is home to many great works of art and scientific inventions. The user will
ask our bot to write a quiz about a given topic and get back a set of questions. We'll write
evaluations to check that the bot is using the appropriate facts and only using facts from our data set.
So let's jump in. In order to get things going, we have to do a little bit of setup. This application
and executing the application for automated eBals requires us to use a couple of third-party services,
including CircleCI, GitHub, and OpenAI. So we're going to put some keys in place to give us
access to those things and setup our GitHub repo. This is all done for you. We just need to put
the coding place. So for the purposes of the lesson and this entire course, there are some utility
functions to make these things easier. And we'll use those, as you'll see here, from this utility
package to put our keys in place and get our environment setup in order to execute our application.
Now we have those three keys in place for the three external services that we're going to use to
make this work, CircleCI, GitHub, and OpenAI. The work to do that, as noted, is in the utility package
and these keys have been provided. If you want to take this and do something bigger over time,
then you would need to go sign up for those services, but they're provided for the purpose of this
lesson. And now, as you can see, we have our GitHub repo and branch in place. There's a generated
branch for each student so that it doesn't conflict with everyone else's work. And again,
that's done for you so that you can get through the lesson very easily. We're going to start by
using the eVals locally, but as the point of this is to get to automated eVals, we will use
GitHub and CircleCI to push the code into GitHub and then have that push code into CircleCI
for the purpose of executing the automated eVal. Which again, gets us to that place where as we
grow our team, as we work over time, we don't have to worry about whether everything has been tested
because it all gets tested on our behalf each time we make a change, helping us to move
confidently and quickly as we build. Okay, so now let's get into creating the actual application.
As described, we're going to build a quiz generator powered by AI. We have a few different subjects,
and first we're going to build the template. Now, note that we're going to store a lot of our data
in strings so that it's visible on the screen and in our templates. If this were larger or if
you were building this as a real application, you'd be much more likely to put this data into
files or into a database so that you could build it more dynamically. This is really built this
way to help you see it very clearly and make it easier to work with. So the first thing that we've
built here is the data set for the quiz and the goal is to have the quiz generator based on the
LLM specifically choose questions and answers from this data set and not to choose questions and
answers from anywhere else. So we're going to work with the validity of that as we start to build
out the application. So this is our underlying data set that we're going to use and now we're
going to build a prompt template that will allow us to ask for specific quizzes and validate that
those quizzes are actually generated based on this data that we've provided. So take a moment to
read the content of this prompt template because this is what's ultimately submitted to the LLM
based on information provided in the request in addition to the content that you saw in the
quiz bank which will be injected into here. So I think it's it's really valuable to understand
how this template is structured. This is how we're going to get the request constructed or the prompt
to the LLM and so you can see it's written here in fairly plain English with some specific
instructions for the LLM and if you look at where the quiz bank placeholder is that will include
the data that we just created as our data set and then combine it with these instructions which
basically ask for the LLM to generate a quiz customized in specific steps. So first based on the
category the user is asking about as discussed we have three options that are available within our
quiz for categories, geography, science and art. We've explicitly outlined those and then the
second step is to identify the subject that we're going to generate the questions about and those
are pulled from the quiz bank. Choose up to two of those and then in step three generate a quiz
based on those two choices the category and the subjects and then use a specific format to generate
the quiz which we'll see once we start running it. Now we're going to take advantage of a third
party toolkit chain in order to build a prompt template that we can use to submit all of the pieces
that we just outlined to an LLM. So if you print this out you can see the content or the generated
object which is the chat prompt that we're now going to use to submit to the LLM. So next let's
choose an LLM. Now we'll also use LLM to get ourselves access to an LLM for the rest of our
actions here. We're choosing to use OpenAI's GPT35 Turbo. You have the option to choose
many different LLMs both depending on your personal choices as well as the option to choose
different ones to try to get different results if you're not getting the right output for your
particular use case. And now we need a parser that's going to take the response from the LLM
and give us something useful in our case we just want a string so we're using lane chains
STR output parser. And now we're going to connect all these pieces together using the pipe operator
from the lane chain expression language. You can think of this as taking the output of one and
piping it into the input of the next the whole thing think of as a pipeline. So we're taking the
chat prompt piping it to the LLM and then piping the response through the output parser to get our
string. Now to take each of those components and make it reusable as one piece we're going to build
that into a function that we're calling assistant chain in order to be able to continue to use this.
Now that we've seen how each of those pieces works we're going to package that up into our quiz
assistant which is built on that chain so that we can use it repeatedly as we test and evaluate
different responses. So now that we have all the pieces we're going to start actually building the
evaluations for our assistant. In this first case we're going to be looking for expected words
meaning when we ask the assistant to generate a quiz for us there will be specific words that we
would expect to see in the response. So for our first example we're going to generate a fairly
straightforward quiz about science and then we have this list of expected words that we assume
if the quiz is generated correctly based on the data set that we provided then some of these words
will appear. This is a fairly straightforward rules based eVAL meaning we are using known
inputs for our testing. In a later lesson we will look at model graded eVALs where we actually
use the power of the LLM not just to generate the quiz but also to evaluate the quality of the quiz
after it's been generated. So now that we have all these pieces we've created our eVAL which is
looking for expected words we have the question we want to ask we have the expected words that
we're looking for we can execute all of this as an eVAL and see what happens. Okay as you can see
now we're talking to a real LLM and this takes a little bit of time as we make the request
and get back the response but here you can see what's generated for us which is a quiz about science
and it contains at least some of the words that we expected so it's going to pass our eVAL.
Because of the way that we created this particular eVAL the eVAL expected words function we use
an assert to throw an exception in the case where the eVAL fails but because our expected words were
found in this particular quiz it passed and printed out the answer without any extra output.
So now let's move on and create a failing eVAL. So in this case what we're going to do is ask
the application to answer a question that it doesn't have any information about and what we want
to happen is that the assistant will decline to answer rather than making up its own questions.
However we haven't actually created those restrictions in our prompt so we're going to run this
and what we should see is a failure of our eVAL. So as you can see in this case we're going to
ask for a quiz specifically about Rome and again we're hoping for the assistant to decline
in a polite and apologetic way by saying I'm sorry and so let's run that and see what happens.
Again you see some time as we connect to the LLM submit our prompt and get back the response
but what we got back from a response was an actual quiz which we did not expect and so in this
case you can see what happens with the assertion which is that an error is thrown indicating that
I'm sorry was not contained in any of this text again you can see an actual quiz that was returned
to us and as you can see right here we expected the bot to decline with I'm sorry and instead got
the full text of the quiz which does not contain in it anywhere the text I'm sorry this would be
a great place to stop the video for a minute and explore this eVAL to ensure that you really
understand what's happening because we passed expected words that we know will be in the response
nothing happened other than printing out the response it only something interesting will happen
if the eVAL fails which in this case would mean finding none of the expected words so take the
opportunity to play with the expected words and see what happens when they aren't found
in the response from the assistant and what you should see is that an exception is thrown
based on the assert later in the lesson we're going to use a similar prompt for the quiz assistant
but modify it so that it would pass this test in that it will refuse to create quizzes based on
data that it doesn't already have stored in its data set so from these examples you can see that
running evaluations on your LLM apps can be extremely helpful in assessing your LLM's performance
in discovering areas for improvement and enhancing the overall functionality and reliability
of your LLM based applications but you could also imagine that if you had to run those evaluations
manually for every change the process would become tedious and time consuming now multiply those
inefficiencies across a team 10, 20, 100 contributors it's just not scalable so instead now let's look
at how we would set up these eVALs to run automatically in a continuous integration process in this
case we'll be using CircleCI and ultimately this will allow your team to stay focused on developing
new features for our first round of eVALs we'll focus on adding basic checks similar to the ones
that we've used so far to ensure that our assistant is being set up properly and producing valid
results these are the kinds of checks that you'll probably run every time you make a change to
your application in later lessons we'll look at more advanced checks including model graded eVALs
that we might want to run on a different cadence your CICD pipeline will automatically run these
different types of eVALs depending on the situation that you're in and if any one of these automated
checks fails your pipeline will stop running and notify you what went wrong so that you can fix
the problem and get back to innovating for this notebook we're using the GitHub API to commit code
and your normal workflow it's more likely that you would use the Git command or command line tools
like GH as a reminder any code you push to GitHub will be publicly visible because we've set up
the course to let you practice these exercises without logging into your own GitHub account
for your own projects you'll want to use your own GitHub account and you can use a private repository
if that's what you need as mentioned previously we've updated the application prompt to decline
generating quizzes for topics for which there is no information in other words we want the LLM
to rely on the available context and not on information that it may have from pre-training
to limit the possibility of hallucination now we're taking the content of much of what we've
built already and putting it into a single app.py file on the file system
in order to see the contents you can use the cat command to dump that content back out
into your notebook environment however you'll note that the syntax highlighting is not visible
which makes it a little bit more difficult to read. As a reminder this is the code that we've
already stepped through we are just taking it and consolidating it into a single file so that we
can put it into Git and ultimately into our continuous integration platform. I'd like to draw
your attention to two specific points in this revised prompt the first is this line with an
explicit instruction to only reference facts in the included list of topics. The second further down
is specific instructions on what to do in the case that there is no information about the subject
that the user is asking about and providing specific text to say I'm sorry but I do not have
information on that topic. Now we've created a separate file test assistant.py which we're using
as the structure of our e-vows and the specific test cases. This is similar to what we did previously
in terms of evaluating for expected words and evaluating for refusal and we're reusing these in
a couple different test cases which I will show you. Again we can use cat to see the contents of this
file and this case we're looking at test assistant.py which includes the functions that we define
previously for evaluating expected words and evaluating refusal and then structuring those
into specific test cases which I'll walk you through now. The first test case is similar to what
we did previously around a science quiz so we ask the LM or the assistant to generate a quiz about
science and then we look for specifically the expected subjects that we know should come from
our data set. Next we take a similar approach and generate a quiz about geography. We use the same
function e-val expected words but this time we pass in a different set of subjects that we would
expect to be present in a quiz about geography and finally we redo the test refusal in this case
asking the assistant to generate a quiz about Rome and expecting the response to include the words
I'm sorry. But I'd like to draw your attention specifically to the system message which we've
modified in order to get the correct behavior as we saw previously when we asked for a quiz about
Rome despite it not being one of the explicitly identified categories we got a quiz and so we've
added these two additional rules at the bottom. First only use explicit matches for the category.
If the category is not an exact match answer that you did not have the information and second
if the user asks a question about a subject you do not have information about answer with
this specific text. I'm sorry I do not have information about that. So some very explicit
instructions given as part of the prompt to ensure that if the data is not available we don't get
made up quizzes about things that are not understood in our data set. Okay so we have all the pieces
in place in order to run these e-vals in our continuous integration environment which is in this
case a circle CI. We do have a circle CI configuration file contained in the lab. You don't
need to know anything about that right now. We'll talk about some of the details of that in a
later lesson. So now we're going to push the two files that we created to our repo on our branch
which means we're pushing them to GitHub. Again we're using a utility function that we've written
to make this easier for the purposes of the lab so we can see the outcome. And now we're going to
trigger a pipeline on circle CI and you can see the URL is passed back to us of where we can find
that pipeline. So here we see the execution of our e-vals in the automated environment on circle
CI. A number of steps here are based around setting up the environment creating the appropriate
Python or installing the appropriate Python version installing the dependencies and then
ultimately running our e-vals which happens right here. And you can see that we used
pi test to run the test assistant.py that we looked at earlier and all of our tests passed
in just under 12 seconds. And this is something that will happen every time we make a change
to ensure that we haven't broken anything in perhaps edits that we make to our prompt
or edits that we make to our application. Excellent. So in this lesson you wrote some very simple
string matching e-vals and learned how to run them in a CI pipeline. In the next lesson you'll
learn how to use LLMs to do model graded e-vals and introduce those into your pipeline. See you in
the next lesson.

## 4. Automating Model-Graded Evals
So far, you've used fast rules-based evaluations that are suitable for running frequently
on every commit.
For stages of development leading up to deployment, more robust and comprehensive evaluation
methods can help you better ensure overall quality before you deploy the app to users.
One such method is model-graded evaluation where you'll use an LLM to evaluate the LLM
app.
Let's take a look at this kind of Eval and how to automate this as part of your testing
pipeline.
OK, until now, we've been using rules-based Evales to make sure our models follow the
guidelines we set up in our prompt and stick to the facts we provided in our data set.
But in order to have full confidence in our application, we also need to be sure our
model is generating high-quality, contextually appropriate responses.
Evaluating LLM output can be tricky because a good response to a query is subjective.
We could try and write custom rules like we did for our initial Evales to make sure
that expected data is in the output, but this gets more and more complicated and more and
more fragile as the application grows.
One approach to checking the output of an LLM is to use another LLM as a grader.
This is referred to as model-graded evaluation.
We'll show a quick example to make sure our model is actually producing output as a quiz.
We aren't concerned with the content just yet, only that the LLM is giving back responses
that look like a set of quiz questions.
We'll look more closely at the quality of output in our next lesson.
So in this lesson, we're going to focus on judging whether the responses in the desired
format, and then in our next lesson, we'll look at things like hallucination and adherence.
So let's jump right in and write a passing and a failing test case to see how this works.
First off, we need to establish our keys again as we did in the previous lesson.
So we'll do that now.
For the purposes of adding model-graded Evales, we're going to continue to use the application
that we generated in the last lesson.
So let's take another look at that just to remember what we're working on.
Again, you can use the Cat command if you want to see the contents of one of these files
on your local file system.
They're all included with the lab that you have.
Now let's look at what it looks like to build a model-graded Eval.
This is going to look similar to the work that we did to build the quiz assistant, except
in this case, we are building a prompt that tells the LLM to evaluate the output of the
quiz assistant.
So you can see here that we are giving specific instructions to the LLM to tell it the role
that it plays in evaluating the work of the quiz assistant.
Before evaluating an actual quiz assistant, we are going to simulate this by writing an
LLM response as if it came from the quiz assistant and using our Eval to determine if that response
would pass the test that we're looking to make.
As you can see here, the full message or prompt for the Eval is telling the LLM to evaluate
a generated quiz based on the context and determine whether or not it looks like a quiz
or test.
It is not meant to evaluate whether the information is correct.
Then it is told specifically to output a Y if the response is a quiz and N if the response
does not look like a quiz.
Now we are going to use lane chain to build the familiar chain that we saw in the previous
lesson, except this time we're building a chain specifically to do the evaluation.
So first we have our chat prompt template as we built previously.
Next we select our LLM, which also we did in the previous lesson and again we're using
GPT 3.5 Turbo.
And finally we select an output parser, which again is the STR output parser to take the
response from the LLM and generate a basic string.
We chain these together as we did previously, but to recap we take the Eval prompt, we pipe
it to the LLM, we take that response and pipe it to the output parser to get the string
that we're looking for.
So now we've built a basic Eval chain using a known good LLM response, which we showed
earlier, and we can prove that we get a positive result by invoking that Eval chain against
that known text as if it were the response from an LLM.
And as you can see in this case, we get a Y, which means the known good response that
we put in is believed by the LLM to look like a quiz in the format that's expected.
However, we also want to ensure that it will fail if it doesn't look like a quiz.
In order to do that, we'll first store all of the Eval chain creation code as a utility
function so we can use it repeatedly.
Next we store a known bad result so that we can pass it into a new Eval chain.
And now we invoke this new Eval chain and see that we get the correct response, which
is an N to indicate that the text does not look like a quiz.
Now we're going to take our newly created, model graded Eval capability and incorporate
it into the tests that we're running inside of our continuous integration pipeline.
So we have two files for testing on the file system in the lab.
The first is testassistant.py, which you can see here and is what we were running previously.
The second is testreleaseevales.py, which is a rollup of all of the work we just did
to show how to create a model graded Eval and execute it against OpenAI.
OK, so now that we have the release Evales, which are the model graded Evales stored
in the testreleaseevales.py file, along with the previous files we had the test assistant
and the app where we stored our original code, we're going to push all of that to GitHub
and then to circle CI so that we can evaluate our application including the model graded
Evales.
Now in this particular case, we are using the full evaluation for the positive case,
but intentionally passing the known bad result to the negative case to show what it looks
like when there's a failure in the continuous integration pipeline.
We are also only going to trigger the releaseevales job on circle CI so that we don't do as
much testing for this particular example.
We do this through this function trigger releaseevales, which uses a parameter passed to the
workflow.
We hear that the only job that is running is the run pre-releaseevales because that
was the one we specifically selected through that parameter.
So as you can see, we ran the model graded Evales in this run and we did get a failure
because we intentionally passed the known bad content.
Great.
So you can see that your pre-releaseevales run when you merge your application changes
to the main branch in a real-world development scenario.
This would happen after you've made changes on your dev branch and run your per commit
evales.
You're setting up a system to progressively increase our confidence in the application
as we get closer to releasing it to users.
In our next lesson, we'll look at ways that you can make the pre-release checks even
more robust, including running evaluations on multiple data points, writing evaluations
to detect hallucinations in your application, and storing evaluation results for human
review in circle CI.
I'll see you in the next lesson.

## 5. Comprehensive Testing Framework
Now we'll look at techniques for expanding and enhancing your evaluation pipeline to give you a comprehensive testing framework for your LLM apps.
You'll learn how to write evaluations to detect hallucinations in your application, run evaluations on multiple data points, and store evaluation results for review in CircleCI.
Let's get started.
One common problem with LLMs is hallucinations, which happen when the model provides an answer that is false output.
hallucinations are a side effect of current LLMs being next token predictors.
In other words, the model will always provide some output that is statistically likely, but there is no built-in way to ensure that the model is producing correct output.
In our application, this might look like the agent creating a quiz with facts that are not in our database.
For example, we could get an inaccurate response, like a user asking, what is the capital of Brazil and getting the answer cell Paulo, when the correct answer is Brasilia.
We could get an irrelevant answer, again asking, what is the capital of Brazil, and getting the answer, the capital of Canada is Ottawa.
In this case, the statement that the LLM produced is actually factually correct, but has nothing to do with the question that the user is asking.
Or we could get a contradictory or nonsensical answer, such as asking, what are the major cities of the USA from largest to smallest by population, and getting New York, Los Angeles, Chicago, and New York?
One way to detect hallucinations is to create a model-graded e-val that accepts some ground-truth data that the model should produce and compare it to the actual output.
Writing an e-val to detect hallucinations does not guarantee that the model never hallucinates, but it is a useful tool to detect if a prompt does not have guardrails to prevent the model from guessing outside of the provided context.
In our application, an example of a guardrail might be modifying the prompt to ask the model to tell the user it can't create quizzes for subjects, not in the quiz bank.
Let's take a look at putting this into practice.
Before we get started, as always, we are going to reload our keys so we can use all of our third-party services. Let's do that now.
As noted in a previous lesson, a much more sustainable way to manage the quiz bank is to put it in a text file or maybe even store that data in a database.
In this case, we're using a text file and loading that into memory when we want to use it.
If you want to see the contents, you can print that out in the notebook.
In order to demonstrate the hallucination detection, first we will quickly rebuild the quiz generator that we used previously.
Now we have the prompt to build the quiz and the assistant chain that puts those pieces together, now we'll move on to creating a modulated e-val that explicitly looks for hallucination.
As you can see, for the purposes of the modulated e-val, we've put a lot of energy into expressing how important it is that quizzes only contain facts from the question bank.
For example, here it is stated that the primary concern is making sure that only facts available are used and quizzes that contain facts outside the question bank are bad quizzes and harmful to the student.
Down below, this is highlighted again. Remember, the quizzes need to only include facts the assistant is aware of.
It is dangerous to allow made up facts.
And then, similar to our previous e-vals, we will output why if the quiz is correct, meaning it only contains facts from the question bank and N if it contains facts that are not in the question bank.
Now, we'll define a function to test our modulated e-val to check what happens when we have hallucinations.
Using the previously stored quiz bank, we'll pass it to our new modulated e-val and check for hallucination.
In the modulated e-val test function, we ask for a quiz about books, which is not included in our quiz bank.
However, in its attempts to be very helpful, the LLM has returned a quiz about books. Our modulated e-val, however, knows that its job is to detect quizzes that contain information that's not in the quiz bank.
And so, in this case, we get an N stating that this is not an acceptable quiz.
What might be causing these hallucinations? If you review our prompt, we're telling the assistant to make the quiz interesting.
This might be a good attribute of a quiz. Education should be engaging, but it is causing our model to hallucinate responses.
Fortunately, our evaluation is detecting these hallucinations so we can go back and correct the prompt.
We're going to move on to the next section, but in our future prompts, we will remove the piece that asks for interesting facts.
As your application grows and changes over time, you will want to add new functionality. For our quiz, this might mean supporting new subjects or adding facts about existing subjects.
To do this, we can create data sets of questions where we know how the model should behave and run tests on each example in the data set.
Let's walk through an example of testing with the data set with our application code. We'll update our application code a bit to prevent the hallucinations.
So far, we've used evaluations as an automated test suite.
This is a good way to catch obvious errors and regressions and to rapidly iterate on your application.
But when working with AI models, it is important to get comfortable manually inspecting and curating data.
Being willing to dig into the data is something the AI and ML engineers that we've worked with say is critical for working effectively in the field.
This is sometimes referred to as error analysis or performance auditing.
In this example, we'll show a way to store evaluation results in CircleCI as an artifact that you can review and share with your team to make sure your application and test suite are behaving exactly as you expect.
First, we're going to rebuild our evaluator to provide not just a response or a decision, but an explanation of why that decision was made.
As you can see in this version of the prompt, we're asking for the decision and the explanation to be separated in a particular format.
Here's an example provided as part of the prompt to ensure that we get the useful information that we need back for later human evaluation.
So now we've rebuilt the chat prompt template with the new prompt and we're going to build a data set so that we can run multiple tests against our new prompt.
As you can see, this test data set includes multiple different prompts from the user or inputs from the user along with some expected responses.
Next, we create a function that will loop through our data set to invoke our quiz generator and evaluate the response for each entry in the data set.
Next, we'll make sure that we have access to all of the functions that we need in order to generate the report that we want based on our data set.
Now we write the wrapper that you're familiar with to create the eval chain that we're going to use through all of our evaluations.
And finally, we take advantage of some tools from pandas to create a data frame across all of our evaluations and the results which will allow us to easily generate a report.
Great, so now you can see the formatted table with multiple results for the different quizzes that we've generated and the responses from the greater.
The first was a quiz about science and the decision is over here. The decision is yes, it's an appropriate quiz and the quiz only references information from the question bank.
There's more details included here, including what specifically is in the quiz and where it's found in the question bank.
The second one is about geography and similarly, the decision is yes, the quiz only references information from the question bank and then goes on to explain what information was taken from the question bank to get these questions.
Now for item three, the quiz about Italy, the decision is also yes, and this is interesting because the quiz does reference information from the question bank and the facts are from the question bank, but the model of a quiz about Italy doesn't specifically map to the categories.
And this is why you want to have human evaluation so you can make the determination of whether this is the type of quiz that you want generated or whether you want a different result, which is maybe to say I don't know how to generate quizzes about Italy, which is the actual topic or subject that's outlined to begin with.
So from here, you could maybe change your prompt or decide to do something differently based on the information that you collected and again involving a human to evaluate at a high level whether all of the pieces of the system are working and where you expect is a great outcome for us here.
Now we're going to take that structure and put it back into CI so that it runs in an automated fashion, but creates the artifact of this report so that you can review it on a regular basis.
In order to run it in continuous integration, we have one additional file save eval artifacts, which generates and stores the output report that you just saw, except as part of the workflow process.
Here's the content of that file which you can look at yourself.
So now we're going to run our evals again against that same data set, but in the continuous integration pipeline to show what it would be like to work with this on a regular basis as we continue to work and grow.
And so we're going to trigger the eval report using the save eval artifacts file along with our original application and the quiz bank TXT file where we stored the full content of the quiz bank.
So now you can see that our evals passed in circle CI and we were able to store a formatted version of the output as you saw it previously in the notebook.
But now it's in an HTML file that is easily retrieved so that you can look at it later, the outcomes and descriptions decisions and explanations are similar to what they were before, but stored in a manner that anyone on the team can come and look at them instead of being local in a notebook.
So in a production application, you might share those results with a colleague to review, help debug unexpected outcomes.
This feedback loop enables you to not only address immediate concerns, but also gives you tools you can use to implement strategic updates to your model and evaluation process.
With this enhanced visibility, you and your development team can make data driven decisions, streamline the debugging process and proactively address potential issues.
Having a holistic understanding of user interactions, model responses and evaluation outcomes is an important part of building robust and reliable LLM apps that meet and exceed your user's expectations.
So that's the end of this lesson.
Again, in this lesson, you learned about hallucinations and how to detect hallucinations using model graded evaluation.
You also learned about data sets and human evaluation on top of model graded emails to ensure that the quality of the output you're getting is exactly what you want for your users.
If you want more details on how to configure all these things in CircleCI and your continuous integration pipeline, there's an additional notebook included that shows the core pieces of that and how to set it up for your ongoing projects.

## 6. Conclusion
Congratulations.
In this course, you've learned some amazing things.
You've looked at eVals, how to use them effectively,
and then how to automate them in a comprehensive pipeline.
So you can work together with a large group
and build amazing technology quickly and confidently.
This is an emerging space.
Combining eVals with CI pipelines brings together
a lot of core learnings and is changing rapidly.
So I'm excited that you're at the front of that
and learning everything that you've learned here in this course.
I'm also really excited to see what you'll build next.
