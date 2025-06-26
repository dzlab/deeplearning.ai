# ACP: Agent Communication Protocol

## 1. Introduction
Welcome to ACP, Agent Communication Protocol, built in partnership with IBM Researcher's BAI Project.
In this course, you build and run agents that communicate through ACP or the Agent Communication Protocol.
You configure sequential and hierarchical workflows of multi-agents that collaborate through a unified rest interface provided by ACP.
I'm delighted that instructors of this course are Sandy Besson, whose AI Research Engineer and Ecosystem Lead,
as well as Nicholas Renoint, who is Head of AI Developer Advocacy at IBM.
Things are true. We're excited to work with you on this course.
To build a team of agents that communicate and collaborate through ACP,
you can host an agent on an ACP server which will receive rest requests from an ACP client and forward those requests to the agent to execute.
The ACP client, which can itself be an agent or any other process,
discovers the agents using the endpoints of the ACP servers and initiates requests.
This unified client server interface helps standardize the communication between agents across teams.
Take, for example, a multi-agent system for customer support.
One team might build a logistics agent on the questions related to all the status.
A different team might build a different rag agent on such general questions related to products.
Each team can make the agents available through an ACP server,
while yet a third agent, hosted inside the ACP client, can process customer inquiries and route them to the specialized agent.
Each team can implement their agents using a totally different framework,
or even switch between frameworks without needing other teams to make any changes to their own code,
because the communication happens through a standardized protocol.
This standardization also makes it easy to integrate any existing agent into the multi-agent workflow.
You can wrap an agent in an ACP server which enables an ACP client to discover it and integrate it into the workflow.
These existing agents can also be made visible in a registry or agent catalog.
The catalog can provide centralized agent listings and simplify searching for agents,
especially in large-scale deployments or enterprise environments.
ACP is an open-source protocol that is also openly governed, meaning no single company controls it.
Many open discussions with the community have shaped the development of this protocol and continue to shape its evolution.
It's also worth noting that ACP and MCP can be seen as complementary protocols.
An ACP agent can use MCP to access tools and then use ACP to interact with other agents.
In this course, you'll first wrap a rag agent built using the QAI framework in an ACP server,
and then interact with it through an ACP client.
To create the sequential workflow, you'll wrap a second agent built using the small agents framework in another ACP server.
Inside the ACP client, you'll learn how to discover the two agents and chain them sequentially.
After that, you'll build another hierarchical workflow where instead of sequentially chaining the agents,
you'll use a third routing agent coded using the small agents framework in the ACP client,
which will wrap the input queries to one of the specialized agents.
You'll then extend the small agents ACP server to use MCP to get access to tools.
And finally, you'll learn how you can discover and run and import ACP agents to the BAI platform,
which is an open-source registry for agent discovery.
Many people have worked to create this course. I'd like to thank from IBM, Kate Blair, and Anna Fuchs.
From deepline.ai, Haral Salami also contributed to this course.
ACP standardizes communication between sensitive agents that might have been developed by the same developer or team,
or developed by different users within the same organization.
But it's not only intended for local development environments,
it can also be used to connect distributed agents even across different organizations.
And that specifically is what you'll learn more about in the first lesson with Sandy.
Let's go to the next video to get started.

## 2. Why Agent Communication protocol
In this lesson, you'll learn about the importance of open standards,
how ACP is helping to fill a gap in the evolving agent landscape,
and about the types of use cases that it enables.
Let's jump in.
In 1989, Tim Berners-Lee proposed the World Wide Web while working at CERN.
To enable transfer of raw data over the World Wide Web,
he developed the first version of HTTP, a transportation protocol for raw data.
Before HTTP existed, there was a patchwork of protocols,
like FTP, Telnut, and Gofer,
that each had specific use cases.
But HTTP soon took over on popularity
because it was simple and openly governed.
It was one of the catalysts that led to the explosive growth of the web.
In the same way that HTTP connected disparate hosted web pages,
ACP, or agent communication protocol,
uses HTTP to connect the diverse ecosystem of standalone agents.
It provides a way for agents to communicate and collaborate with one another easily,
no matter what framework they're built on or technologies they leverage.
Building complex multi-agent systems,
like agents performing search, writing, editing, or tackling workflows,
requires collaboration with other agents to accomplish a shared goal.
There are many frameworks that target this agent development,
but the ability to plug and play with agents from different frameworks is an unmet need.
Today, most agents lack a shared protocol for communicating with one another.
Connecting agents built using different technologies requires developers
to build brittle and non-repeatable integrations,
which are also susceptible to the constant changes pushed in the frameworks.
ACP provides a unified interface for agents to communicate,
regardless of their framework.
One of the key values that ACP holds is open governments and community collaboration.
This means, instead of just being open source,
where developers can look at the code and request changes,
but decision-making is made by a single or group of companies.
ACP's open governance invites the community to shape the roadmap.
So ACP becomes a protocol shaped by the community for the community,
much like HTTP.
ACP benefits not just the developer who doesn't need to constantly rewrite integration logic,
but also the end user who will receive better results
because there is more choice for the developer to choose the best agent for the task.
Now you're going to learn some of the exciting patterns that ACP makes possible.
First, dynamic updating.
With how fast innovation is advancing in this field,
the agent that was best last month is often not what works best this month.
And as we just discussed, making an agent ACP compatible
means that you can swap agents in your systems,
even if the new agent was built using a different technology.
This makes your system truly interoperable
and can be very useful for testing what agents give you the best performance in your system.
Specialized agents can work as a team.
So instead of building one giant agent to handle everything,
ACP allows teams of specialized agents to collaborate dynamically.
If you had a research agent collaborating with an agent that creates a beautiful visualization
and an agent that is especially configured for financial modeling,
then when you ask for a report, the agents could hand off tasks to one another
just as teams do when they collaborate.
You can also create a cross-company workflows.
Companies rely on many platforms like a customer relationship management platform
enterprise resource planning, human resources systems,
project management tools, and many more.
For example, if a retail company uses a customer support platform
and an ERP system to manage their inventory and logistics,
each system could have a dedicated agent to manage tasks.
So if a customer submits a complaint about a delayed shipment,
the customer support agent can recognize that it needs the assistance of an inventory agent
and sends a request using ACP.
Each agent stays focused on its domain but collaborates using ACP.
Now ACP also opens up the opportunity for inter-organizational collaboration.
You'll build out an example of agents communicating from different organizations with Nick.
But in a nutshell, ACP can enable agents from different companies
to securely collaborate even when they're hosted by completely separate companies.
This opens up a whole new world of possibilities
as we move from just using agents internally
to making our agents available to external providers so companies can work together.
ACP also supports human and the loop collaboration
so that you can ensure that your agentic system is operating responsibly into your expectation.
In the next lesson, we'll learn more about the architectural components
and core principles of ACP.

## 3. ACP Core Principles
Now that you have a good understanding of the value that ACP brings
and the use cases that it enables,
let's understand where it fits in the tech stack.
It's overall architecture, the lifecycle of an ACP agent,
and how it compares to some of the other protocols out there
like MCP and A to A.
Let's go.
To really grasp how ACP works,
it's helpful to understand where it fits within an agentic system.
Alongside components like the foundation model, storage,
agent orchestration, application layer, and more.
This diagram shows that ACP sits below the application layer
at what's sometimes called the operational layer.
It uses HTTP and the rest architectural style
to facilitate communication between independent systems.
ACP is based on a client server architecture
that uses a simple rest interface.
This means that the client is responsible for initiating communication
and the server responds to that request.
In ACP, both the client and the server can be an AI agent,
a human, or a microservice.
The client will always initiate the request
and the server will always respond to the request.
In a basic example of ACP interaction
where the server host is an AI agent,
the ACP client discovers available servers
and then initiates a request.
It sends a rest request to the ACP server.
The server, which wraps an AI agent, manages the request
and returns a response back to the client over rest.
As things get a little bit more complex,
we can observe how multiple layers of protocols
each serving different purposes can work together.
In this diagram, we see a client makes a request
to an ACP server over rest, which wraps an agent.
The agent determines it needs to invoke a tool,
so it sends a request to an available MCP server
to execute the tool call and return the result.
Once the ACP agent completes its run,
it returns the output to the ACP client.
Before you dive into building out your first ACP compliant agent
in the next lesson, let's go through some core concepts
that will help you understand how the protocol works
on a deeper level.
Agent Detail is where you define your agent's
basic identity and capabilities.
You can specify the agent name, provide a description,
and add optional metadata for additional information.
This basic information makes your agent discoverable
and usable within the ACP ecosystem.
The agent Detail enables both online and offline discovery.
Online discovery occurs when ACP servers are already running
and can be accessed through their API endpoints.
All-flying discovery happens at a higher level,
like at the Agent Catalog or Registry,
where Agent Detail is embedded in the agent package,
allowing a user or system to discover agents
without requiring them to be running first.
This enables the creation of an agent catalog or directories
where agents can be browse, selected, and then spawned when needed.
To activate your agent and enable its online discovery,
you need to deploy it.
You can either use the SDK built-in server,
which is the easier way to get started,
or use an external server.
Starting the ACP server makes the agent available
to the ACP client to run.
The built-in server is what you'll be using throughout this course.
Once the agent is activated,
it's ready for execution,
meaning it can actively process requests and generate a response.
ACP offers three execution modes,
synchronous, asynchronous, and streaming.
In synchronous mode, the client waits until the agent completes its run
and then returns a final result.
In asynchronous mode, the client doesn't wait for the agent to respond.
It can carry on with other tasks in the background.
When streaming, the server establishes an SSE or a server sent event connection,
which provides real-time updates as the agent generates results.
Regardless of the execution mode,
each run progresses through states such as in progress,
awaiting, and eventually reaches a termination state,
such as completed or failed.
The agent communication protocol is designed
with a production-grade environment in mind.
So it prioritizes security, scalability,
and observability for reliable performance at scale.
One frequent question is what is the difference between using ACP,
MCP, or A to A?
And put simply, MCP is designed for enriching a single model's context
with tools, resources, and prompts.
But MCP and ACP are compatible
and even complementary.
An agent might use MCP when making the tool call
that returns more context to the agent,
and then that agent might decide that it needs to communicate with another agent,
which it does over ACP.
However, MCP is decoupled from ACP
because it's not primarily designed for agent-agent communication.
For instance, as of now, the MCP protocol is not designed
to easily enable an agent to carry over a task from another agent
or peer-to-peer collaboration.
When it comes to shared memory,
MCP has the support for session management,
meaning servers can be stateful
and maintain information about a client's sessions between requests.
However, MCP doesn't handle the state itself.
The ACP SDK supports centralized storage for runs and sessions,
meaning that multiple ACP servers can persist information across these runs or sessions.
For message structure, MCP doesn't form any opinion on the message.
Since agents need to support natural language as well as other modalities,
ACP messages follow multimodal structures for exchanging content.
Google's A to A protocol,
which was introduced shortly after ACP,
also aims to standardize communication between agents.
Both protocols share the same goal of enabling multi-agent systems,
but they diverge in philosophy and governance.
Some of the notable differences are that both A to A and ACP are open source,
but ACP is openly governed with the Linux Foundation,
ensuring its open, neutral, and centered around community participation.
An ACP you can run multiple agents with the same server,
reducing overall setup and management effort.
Whereas with A to A, each agent must be run with a separate server.
ACP also follows the rest-based architecture style,
supporting standard web infrastructure and patterns
that enhance scalability and interoperability.
A to A uses JSON RPC style, which can introduce a bit more complexity.
In A to A, the agent output and message history are separated.
This can make it very difficult to determine the order of events in a multi-agent turn
without implementing additional ways of persisting this information,
which is really important for transparency and observability.
ACP supports a wide range of agent types,
and allows developers to explicitly choose between synchronous,
A synchronous, and streaming modes.
This makes response handling predictable
and simplifies client logic that guarantee streaming via SSC.
A to A also supports both stateless and stateful agents,
but the interaction mode is determined dynamically by the agent.
As a result, clients must be built to be flexible in handling sync,
async, or streamed responses based on the agent capabilities.
While ACP and A to A have some differences,
it's important to highlight that they have a lot in common.
What you learn in this course with ACP
should largely carry over to A to A.
You now have a handle on one you might use ACP, MCP, or A to A
based on your goals and system configuration.
It's important to remember that these protocols are all
living and breathing projects that change rapidly,
and what might be true when filming this course
could easily be out of date in the near future.
So always make sure to do your own research
and confirm any assumptions with your own experimentation.
Congratulations! You've made it through the theory,
and now you're ready for the hands-on part of the course.

## 4. Building a RAG Agent with CrewAI
Welcome to lesson three. Now, before we get into ACP, we're going to need an agent to create
an ACP compliant agent. So in this lesson, we're going to be building our very own retrieval
augmented generation agent with a particular focus on insurance coverage. This is going to form
the basis for the agent that we eventually wrap inside of ACP. Before we get into any ACP,
we actually need an agent to convert to an ACP compliant agent. So we're going to start out by
building our very own rag agent with crew AI. This is pretty straightforward, but we're going to
take it step by step nonetheless. So first up, what we need to do is bring in a couple of dependencies.
So we're from crew AI, we are going to import a crew and this forms a basis for the architecture
of a multi agent system. We're going to have a single agent, but we kick things off using the crew
itself. We're then going to be able to define a task and this is where we're going to pass through
our prompt as well as our desired outputs of effectively our task contains everything that we
actually want to do. We're also going to bring in an agent and this is going to be our LLM
plus our tools and that brings us to our LLM. We're going to bring that in as well. So we've got
our LLM provider to be able to create an agent. Now, the nice thing about building a retrieval
augmented generation system is that a number of providers have made it a lot easier to go and
chunk up and vectorize your data. So we are going to do exactly that. The nice thing about crew AI
is that it comes with a rag tool. So let's bring that in. So we're going to from crew AI tools,
we are going to import the rag tool. So this rag tool is going to facilitate us building up a rag
system. So if I go and run that, that is now all run successfully. Now, there are a couple of
warnings when you go on ahead and run this. So we're just going to ignore them for the time being
and to do that, we are going to import warnings and we are going to filter warnings and ignore those.
This means that we're not going to get any unnecessary warnings that we don't necessarily need.
The next thing that we're going to need is an LLM, right? So retrieval augmented generation is all
about having a vector database and querying that vector database, bringing back the context,
passing it through to our LLM and generating the output to generate that output, we're going to
need an LLM. So let's go ahead and bring that in. So I'm going to define a new variable called LLM
and that is going to be equal to our LLM class that we brought in from crew AI. So to define that,
I'm going to create a new LLM and we're going to specify the model. I'm going to set that to open
AI GPT-4, but you could use a range of other providers. You could use Obama, you could use
what's next. You could use a ton of other providers. This is just the one that we're going to be using
for this example. So if you want plug in a different one, see what actually happens. So the next
thing that you're going to need to do is define how many maximum tokens you want to allow LLM to
generate. So to do that, we're going to specify max tokens and set that to 1024. So if you'd like
your LLM to be able to generate more or agent to be able to generate more, you could bump that up.
Likewise, if you want to be a little bit more conservative with how many tokens you're generating
as part of your agent system, you might choose to bump that down. Okay, so that is our LLM now created.
Now to build up our rag tool, we first up need to define a bit of config. So let's go ahead and do
this. So I'm going to create a new dictionary called config. And this is going to contain our LLM
as well as the embedding model that we want to go and use. And this specific config is all for
our rag tool that we imported from crew AI. So we need to define what LLM we're going to use to
generate output. And we need to define what embedding model we're going to use to embed our chunks
as well. So let's go ahead and define this. So we're going to define a new key called LLM.
And within that dictionary, we are going to have two values provider and config. Our provider is
going to be equal to open AI and keeping with the theme and the specific model that we're going to
be using is going to be contained within the config key. And we're going to set that equal to model
and you guessed it, GPT4. Now we've talked about our LLM, but we really need an embedding model as
well. This is what is going to take the chunks from our document and convert them into a method
that we can search inside of our vector database. So we are going to create a new key called embedding
model. And then we're going to specify another provider. And again, we're going to set it to open
AI. And then we need our config. So we're going to specify config. And this time we're going to
specify the embedding model that we want to use here. So we're going to set that equal to model.
And then the value that we set to model is going to be text embedding ADA 002. So this is the
config that you're going to use for your RAD tool over here. But again, feel free to update this
if you need to. So I'm going to run this. That's our config now defined. Now so far we've gone and
set up our config. We've set up our base LLM. We need to go and define our RAD tool now. So again,
crew AI's got our RAD tool that you're able to use to be able to go and define this and build up
your agent system. So we're going to create a new variable called RAD tool and we're going to set
it equal to RAD tool or an instance of RAD tool. And then through that, we're going to pass through
three different values. So we're going to pass through the config and we're going to pass through
the config dictionary that we just defined up here. We're also going to specify the chunk size and
we're going to set that into chunks of 1200 tokens. So this means that we're going to be chunking
up our document into chunks of 1200. And we probably want a little bit of overlap just to make
sure that we're not cutting off chunks in the incorrect places or places which aren't so ideal.
Now that's our RAD tool sort of defined, but we also need to add a document to that. So how do we
actually go about doing that? Well, I've got a document over here and this is going to be stored
inside of the same environment that you're running. So you'll be able to access this. This is a
PDF document about potential insurance inclusions. Now this agent that we're going to be defining is
going to be all to do with finding out what's included potential waiting periods on insurance. So
we need to provide a document which contains that specific information. So it's almost like a product
disclosure statement for an insurance policy. Now we want to get this into our RAD tool. So
how do we go about doing that? So the first thing that you're going to go ahead and do is define
this RAD tool obviously, but the next thing that you want to go ahead and do is actually add that
document to the vector database. So we are going to go and use the add method from our RAD tool
and specifically add in that document. This is going to be contained inside of the environment
that you're working in. But again, if you want to do this locally, you'd go and update this path
to the document that you want to embed and load into your vector database aka your RAD tool. So
to the add method, we're going to pass through the file path for our document. This is currently
stored inside of the root folder and then inside of data. And then we are going to define the
specific name of the file, which is gold, hospital and premium extras. PDF. So if you've got a different
file name, you would go in and update the file name for the document that you're working with
right over here. We also need to define the data type. So we're going to specify the data type
and we're going to set that equal to PDF file over there. So if we go and run this, this should
start chunking up our document and push it into our vector database. And that looks like it's run
successfully. So we've got to define our LLAM, you've got to define your config, you've got a RAD tool.
Now, when are we going to define the agent? Well, it just so happens, we're going to do that now.
So in order for you to define your agent, we're first up going to define a variable,
which is going to contain the agent. And then you need to go and use the agent class, which we
had right up there. So we're going to define a new agent. And this is going to contain a few
things. But this is kind of why I like crew AI because it's very verbose in how you go and build
one of these agents. The first thing that we need to define inside of our agent is the role. And
this is the role that that agent is going to take. Now keep in mind, we're going unloading a
product disclosure statement for an insurance policy. So we are going to have a senior insurance
coverage assistant here. So that's going to be the role for our agent. Now, we also need to define
the goal. So we're going to specify our goal as determine whether or not something is covered or
not. And this is all to do with whether or not something is covered within your insurance. And
might also talk about waiting periods, so on and so forth. Really, it's all to do with the
information contained within that PDF. And then we want to go on ahead and define a backstory.
This is probably my favorite key. And so the backstory in this particular case is going to be you
are an expert insurance agent designed to assist with coverage queries. So you can see that we've
now gone and to find our backstory. Now, we also need to define a few more things. So you're going
to go and set verbose equal to true. This means that as our agent is running, we're able to see
active progress. You're going to set a loud delegation to false. This means that we're not going
to be passing off the task to other agents. You're also going to pass through your LLM over here.
So this is the LLM that we defined up there. It's the open AI GPT for instance,
but we also need to pass through our rag tools because so far we've gone and defined it,
but we haven't actually given it to our agent over here. So under tools, we're going to create
a new list and we are going to pass through our rag tool that we've now gone and defined.
And the last thing that we need to define in order to create our agent is the maximum retry
limit. So we're going to define that and set that equal to five. This basically states that our
agent is going to try to get the answer at least five times. And if it doesn't, then it's going
to error out. So this means that it's again, a bit of an insurance policy that you don't go and
blow out your number of API calls. So if we go and run that, that looks like our agent has now
successfully run and we've all not necessarily run, but we've now gone and defined it.
The next thing that we need to do is define a task and this is going to contain our prompt.
So inside of our prompt or prompt is going to be encapsulated inside of our task that we've
got over here. So let's create a new task. So we're going to create a variable called task one
and this is going to be an instance of the task class from crew AR. Now, task has a couple of key
things inside of it. A description, the expected output and the agent that we want to delegate
this particular task to or give it to. So the description, the first prompt that we're going to be
passing through is what is the waiting period for rehabilitation? Now again, you might go and try
a different prompt there. You might choose to try something which you see in this document.
If you go and use the different document, this is where you'd go and pass through your specific
prompt. We then want to define the expected output. So inside of your expected output,
we are going to set that as a comprehensive response to the user's question. You might choose
to change this a little bit. It might be a summary. It might be bullet points. You're effectively
describing the style here. Now, remember the last thing that you need to define inside of your
task is the agent that you want to give this prompt to. So we're going to define that right
over here. So we're going to specify the agent as the insurance agent. Remember, we define the
insurance agent over there. It's now coming into our task down here. Now, if we go and run this
task, that cell has now completed successfully. So we now define our agent. We've now defined our
task. We now need to encapsulate it all inside of a crew and then kick it off. So how do we do that?
Remember, when we went and imported our dependencies, we imported a crew. So we're going to define
a new instance of our crew. And we are going to pass through a few key things here. We need to pass
through the agents themselves, the tasks. If we had multiple tasks, we'd pass through multiple tasks.
We also want to specify that it's going to be verbose. So inside of our crew, we're going to
define our agents. And this is going to be a list. And right now, we've only got one agent. But if
you had multiple agents, you'd pass multiple through. I think for now, we're just going to have one
single agent over here. We're then also going to have the tasks that we want our agent to complete.
For now, we're just going to have task one. If we had multiple tasks, we'd pass them in through
here as well. And we're going to set verbose equal to true that way. Again, we get output progress
as our agent is running. Then all that's really left to do is kick off our task. So to do that,
we are going to create a new variable called task output. And we are going to take our crew from
over here and run the kickoff method. This is akin to like running the crew or actually starting
things or when you hit enter in chat GPT or hit enter inside of an LEM system like Olamam.
So that should get everything up and running. The last thing that we want to do is print out
our final task output. So if we go and run this, this should hopefully kick off your
our agent and get your agent running. So if we go and run it, take a look. You can see crew
execution has started. We've gone and passed through the prompt to our agent. You can see that we've
got our chunks from our RAG system. If you scroll a little further down, take a look. We've got our
final answer displaying over here. So the waiting period for rehabilitation coverage under this
insurance policy is two months. However, if the requirement for rehabilitation is due to an
pre-existing condition, the waiting period extends to 12 months. It's important to consider
this information can vary on the specifics or situation, so on and so forth. You have the idea.
The answer is now here. So again, we've got a bit of metadata. And you can see this is the
output from actually printing through the actual task output from over there. So if we wanted to
go and push this to a different system, if we wanted to output to a JSON document or something
of the sort, we've got the ability to work with that final output as well. But again, keep in mind,
you can go and change this prompt. So you might choose to change this query over here. You might
choose to change your insurance agent or even change the data that you passed through to your RAG
agent. But for now, we've gone and defined our initial RAG agent.

## 5. Wrapping the RAG Agent into an ACP Server
Now that you've built the RAAG agent, it's time to bring in ACP into play,
because eventually what you're going to do is take your first insurance ACP agent
and connect it up to a hospital agent. That way, imagine you've got agents communicating
across different organizations. That's the beauty of ACP. We can define an agent once and have them
called from a universal client. In this particular case, we're going to get two organizations to
talk to each other using their agents. But before we get to that, we first up need to wrap our
existing agent, our RAAG agent, inside of ACP. So you're now going to convert your existing
RAAG agent into an ACP server. So over here, I've got the existing code that we broke in lesson
three. This is the exact same insurance RAAG agent that we've already built out.
But how do we convert this into an ACP compliant agent and specifically an ACP server?
Well, first up, what we're going to need to do is run this as a standalone server.
So we need all of this code to be output into a file that we can run using Python or UV.
So how do we go about doing that? Well, we typically use an IDE. But for this specific learning
environment, we're going to use a magic function that will copy the contents of this cell into a
Python script. And we're going to export this into a file called crew underscore agent underscore
server.py. So when we're going to run this cell, it's actually going to export all of this code
to that specific file there. Now, the next thing that we're going to need is a couple of dependencies.
To actually convert this into an ACP server. So we're going to bring in the ACP generator from
collections.abc. This is going to form the output type for the server. We're also going to import
message and message part from the ACP SDK. This is going to form the format that we use to output
the results of our agent. While we're at it, we're also going to import run yield run yield
resume and server from the ACP SDK dot server class run yield and run yield resume are going to
form part of the asynchronous generator. So this effectively showcases the type of output that our
ACP server is going to be exporting. The server itself is going to form the foundation of our ACP
server. Okay, so those are the main dependencies now imported. Now, the next thing that we need to do
is create an ACP server. So to do that, we can create a new variable called server and assign it
to ACP server that we've imported from over there. Then what we're going to need to do is begin
wrapping our existing agent. So I'm going to scroll on down to where we had our agent and we're
going to use the server dot agent decorator. This is going to tell the ACP SDK that what comes under here
is an agent that we want to make available on the ACP server. Now, the name of the agent that actually
goes into the server is based on what we actually name this function and we're also able to provide
metadata via the doc string. So let's go ahead and start wrapping this. We're going to make this
an asynchronous function and we're going to call it policy agent. Then we need to define what
types of input we're going to take and this is effectively where our prompt is going to be captured.
The first variable that we're going to capture is our input and this is going to be a list of
messages. Then we're going to return a generator. That means we're able to iteratively return
output from our ACP server and that is going to be an asynchronous generator and that's going to
take in run yield and run yield resume from over here. Now, one of the most critical things is also
explaining what these agents are useful for. So ACP makes this available through the doc string. So
all we need to do is provide a doc string underneath our function. So our policy agent, if you
remember correctly, is specifically an insurance related agent and provides us with information
around our coverage periods, whether or not something is included or not. Now, we're able to
provide this back to our users via the ACP server using this specific doc string here. So we're
going to write it out. So this is an agent for questions around policy coverage. It uses a
rag pattern to find answers based on policy documentation. You can use it to help answer questions
on coverage and waiting periods. So that's pretty much our doc string now defined. Now, when somebody
looks for metadata around this agent via the ACP server, they'll be able to return that particular
bit of information. Now, we need a tab all of this in and there's one key thing that we need to
change here. Right now, we've got a fixed prompt, which has been passed through inside of our
task. But we really want to be able to capture a dynamic prompt when a user goes and calls out to
our ACP server. And this is where our input variable from up here comes into play. Now, we do need
to unpack the values from that input because right now we're saying that we're going to be passing
through a list. But for now, we just want to take that one prompt. So to do that, I'm going to
convert this static prompt to an input variable. So we're going to grab the first value from our input
and we're going to grab the first part and we're going to specifically return the content.
Now, if you actually go and print this out, you'll actually see that when we return that message
variable, we're able to actually extract all of those different components. So if you add multiple
prompts, you've got the ability to maybe loop through and create multiple tasks. Maybe you try that
after this. Okay, so that is our task now amended. Now, because we are running this asynchronously,
we want to kick off our agent asynchronously as well. So we're actually going to await our
crude or kickoff method and we're going to convert that to asynchronous. Now, right now,
we're not actually returning anything from the server. So we need to return something as well.
Now, rather than returning, because we are producing a generator, we're actually going to yield.
So we're going to use the yield method and we're specifically going to return a message,
and we're going to return the different message parts from our agent. Now, this is going to
take in our task output. Now, let's say you had another agent and it just returned back a specific
text prompt. You'd be passing through the text prompt inside of this value here. Keep in mind,
we're formatting it as a string. That way, we're keeping consistent when it comes to generating
our outputs. But when we go and run this now, after our agent gets kicked off, it'll take in the
prompt for our user over here, we're then going to unpack it and push it into our crew AI task.
And then eventually, we're going to return it down here. Now, we do need to run this server,
because keep in mind, we're going to be outputting this as the crew underscore agent underscore server.py
file. So how do we go about running this? Well, we're going to use a pretty common Python structure.
So if name is equal to main, then what we're going to do is we're actually going to run this server.
And we're going to run it on a specific port because a little bit later on, you're going to spin
up another server so we can build sequential ACP server calls. So we are going to take our server,
and we're going to use the run method to kick it off, and we're going to run it on port 80001.
Now, if I go and run this cell, this is going to output it into that specific folder.
So my underscore ACP underscore project, and we're going to output it to the crew underscore agent
underscore server.py file. So if you go and open up that file and specifically inside of that
repository, you'll see that we've exported this code that we've just written over here. Now that
we've got that code exported, though, we can actually begin to run that ACP server. So to do that,
we're actually going to create an iframe inside of our Jupyter notebook. That way you can start
up the server as well and give it a go. So I'm going to grab this code, which will open terminal one.
Now, we're eventually going to run multiple ACP servers. So we're going to run this server in terminal
one. And then our later on, we're going to define a hospital server. And that's going to be running
in terminal two. Now, if we go and run this, we should get our terminal back and take a look.
That's our terminal. And right now, you can see that we're inside of the work folder and inside
of the my ACP project folder. Cast your mind back when we went and wrote this out. We were writing out
our crew agent server file to the my ACP project file. Now, if you're going to run this locally on
your own machine, just keep in mind that you need to be in the same folder where the crew agent
server is currently running. So we're going to be using UV to run this. Now, if you've never used UV
before, you can install a by just running pip install UV. And if you're creating this for the
first time, again, on your local machine, you're going to need to initialize a new project by
running UV and knit and then creating a virtual environment by running UV, V and V. We've already
set this up inside of your environment. So you've got the ability to just go and kick off the crew
AI server. So to do that, we can run UV run crew agent server.py. And if we run this, hopefully,
we'll get an endpoint back that shows us that our server is running. And take a look. We've got
our endpoint returns. So we can now see that our server is currently running at a local host
environment on port 80001. And remember, we specified our port up here. Now, the thing about this
is that it's going to stay up for 120 minutes. So if we start this server this first time,
it'll run for 120 minutes. But if you go away and come back to this lesson, you're going to need
to start it up again. This only applies to this environment. If you're running it locally,
it's obviously going to stay up for as long as you're running the Python script. But just keep in
mind that when you're running it inside of the deep learning dot AI environment, you've got 120 minutes.
So now we've got the ability to call out to it via an ACP client in the next lesson.

## 6. Calling an ACP Agent using the Client
So you've got the ACP server up and running, but how do you actually make a call to it or use it?
Well, this is where the ACP client comes in.
Using the client, we're able to create a standardized set of calls out to that ACP server.
If we add multiple agents running on that server, we can hit all of them.
And if we add different servers, well, we can use a similar style structure to call out to that ACP server,
which we'll do a little bit later. But for now, let's go and define our ACP client and make a call
out to the ACP server that we've created. So we've come to the time actually making a call out to
our ACP server. Well, first up, we probably want to double check that our server is still running.
So we're going to run through the exact same method that we did to get our server up and running.
We're going to open up the terminal. And then what we're going to do is we're going to render terminal
one exactly as we did previously. So that way, we can just double check to see whether or not
our server is up and running. Because remember, if you're running this inside of the deep learning.ai
environment, it's going to stay up for 120 minutes. If you're doing it locally, it runs for as long
as you've got the server up and running or the Python script running. So if we go and run this,
we can take a look to see what's happening with our server.
And take a look. It looks like our server's still up and running. You can see it right over there.
Now what we want to do is begin making calls to our ACP server. And the nice thing about this is
that you can have a standardized client or eventually you're going to be able to integrate
into different systems using a similar style pattern. For now, what we're going to do is we're
going to make sure that we can nest async.io calls or async.io calls. So we're going to import
and nest async.io and apply that to our environment. This allows us to run the ACP client
from the Jupyter Notebook environment. And then what we're going to do is we're going to begin
building up our client. Now within the ACP SDK, you've got a client available. And think of this
like a way to just communicate with our ACP endpoint. We're also going to bring async.io to be able to
make asynchronous calls. And we're also going to import color armor and specifically the four
library. This just makes it so that you can output stuff using colored terminal formatting.
It just looks a little bit nicer. You're able to see things a little bit more clearly.
Then what we're going to do is we're going to create an asynchronous function. So we're going to
create a function called example. And that's going to return nothing. And then we're going to connect
to our client. And our client is going to point to the URL where ACP server is running. So right now,
we're going to set our base URL and set it equal to local host 80001. Because if we scroll up,
you can see that it's running there. Later on, if you've got different servers running and when
we begin making sequential calls, we'll have to connect to two different clients because we're
connecting to different servers. And even though we've only put one agent on this particular server,
you could have multiple. And I'm going to show you how you might call out to different agents on
a single server. So we're going to define that as a client. And then what we're going to do is we
are going to run a synchronous call to that particular server. And we're going to target the agent
that we define. Because remember the agent that we define was our policy agent. The name of the
function when you define the ACP server is going to be the name of the agent when it comes to calling
it with the client. And I'll show you what I mean in a second. So we're going to create a variable
called run. And this is going to await a synchronous call. And then we're going to target our policy
agent. Remember when we defined our agent in the ACP server, we name the function. So we typed in
DEF policy agent. So policy agent was the name of the function. That then becomes the name of
the agent when we use ACP. There's one last thing that we need to pass through to our client in order
to run a prompt or trigger a prompt. And the cool thing is that when you go and run this,
you'll actually see the call kick off in a ACP server appear. So inside of the input argument,
we're able to pass through our prompt. So I'm going to pass through what is the waiting period
for rehabilitation. Remember we've asked this previously, but we had it hard coded inside of our
agent. So all things holding equal, this input should then be passed through to us agent running
on the ACP server. So we should be able to see that pass through and run up there. Now we're also
going to print it out back into our Jupyter notebooks. So that way we can see it a little bit
more easily. And to do that, we're going to print and we're going to use the color armor yellow color.
And then we're going to unpack the output from our client. So we're going to get the first value
from our output. We're going to grab the first part. This is a similar way to how we actually unpack
our input prompt when we went and defined our server. And then we're going to grab the content.
And then we're also going to reset the terminal coloring using color armor over there.
So this should really just be our client now done. Now if you wanted to, you could go and change
the prompt here to something different. Try running a different prompt, see how the ACP server
performs. Remember, it's going to be limited to what we've actually put inside of our vector
data days, but you might try uploading different documents and try different inputs and different
prompts as well. And eventually we're going to link these up to make sequential calls to multiple
servers and connect multiple agents together. Okay, so if we run that cell, that looks like we
don't have any issues there. And then we're going to use async.io and the dot run method.
And then run our function over here, which is called example. So if we go and run this now,
as soon as we trigger this, we'll be able to see whether or not it's running up here in our ACP
server. So if I go and run this now, take a look. Execution has started. You can see that we've
got our prompt pass through. What is the waiting period for rehabilitation? Looks like it's all
running. We're getting all of our context and our prompt and all things holding equal. We should
get a completion. Take a look. The waiting period for rehabilitation in the insurance policy is
two months. We should get this printed out out here as well. So take a look. We've now made a call
to our ACP server using the ACP client.

## 7. Wrapping a Smolagents Agent into an ACP Server
So you've gone and created an ACP server
and you've defined an agent running on it
and you've been able to call out to it via a client.
But this doesn't really demonstrate interagent
operability, does it?
We've only called out to a single agent.
Well, we're going to define another ACP server
and begin the process of sequentially
training these agent calls.
The first thing that we need to do is create our second server.
And in this case, we're going to create a hospital ACP server
to work with the insurance server.
So we've gone and defined our first ACP server,
the insurance server.
Well, we're going to define another ACP server,
but this one's going to be more focused on a hospital.
If you imagine this might be a common pattern
when it comes to different organizations,
you might use different ACP servers to communicate
across those different organizations.
Maybe you might even define different ACP servers
to communicate across different teams.
In this case, we're going to have a hospital
to ensure a type of relationship
when it comes to working with our different ACP servers.
So we're going to start in a similar manner
to what we did for the insurance server.
The first thing that we're going to go on ahead and do
is we're going to create a server
and we're going to output this into our MyACP project repository
or directory.
To do that, we're going to use the right file magic function
inside of our Jupyter Notebook
and then we're going to export this file
into the MyACP project directory.
And the file that we're going to be outputting
is smallagents underscore server.py.
Why smallagents?
I'll come back to this in a set.
The next thing that we need to do
is bring in a couple of dependencies.
Down to these dependencies are going to be pretty much the same
as what we brought into our insurance server
from an ACP standpoint.
The agent stuff is going to be a little bit different.
So first up, what we're going to do
is we're going to bring in the Async generator class
from collections.abc.
This is going to be used as the type
when it comes to defining our agent function.
We're then going to import the message
and message part class from ACP SDK.models.
We're going to use this to structure the output
that we send back from our ACP server
and specifically our agent on that ACP server.
Then we're going to bring in a couple more dependencies
from our ACP SDK.server class.
We're going to bring in run yield, run yield resume
and the server itself.
Run yield and run yield resume are going to form part
of our asynchronous generator
and the server is going to form the context
of our ACP server altogether.
Now, why did we call it smallagents server?
Well, rather than using crew AI,
like we did for our insurance server,
we're going to use the smallagents framework this time.
This sort of shows that you've got the ability
to use different agent frameworks
when it comes to working with ACP.
So what do we need to bring in from a smallagents perspective?
Well, we're going to go from smallagents,
we're going to import the code agent,
we're going to import the duck.go search tool.
So this is going to form our search tool.
We're also going to bring in the light LLM model class
and the visit web page tool.
So the code agent is going to form the crux of our agent
and this means that it's going to be writing up Python functions
to be able to call out to different tools.
The duck.go search tool and the visit web page
tool are going to allow us to access the internet
and the light LLM model is going to form the basis
for the LLM that we're going to use inside of our agent.
Those are the main dependencies now imported.
Now, we're going to follow pretty similar structure
when it comes to defining our ACP server.
So what we're going to do is we're going to create a new server
and this is an instance of our ACP server
that we define over there.
We're then going to define our LLM model.
So to do that, we're going to use our light LLM model class
and we're going to pass through our model ID,
which in this case is going to be OpenAI GPT-4.
But again, you could use your own LLM provider,
you could use Obama, you could use local LLMs.
You've got the ability to use a number of different choices.
If you want to use what's next or Obama,
those are the other ones that I mainly use.
Then we're going to define the maximum number of tokens
for our agent.
We're going to set that to 2048.
We need a little few extra when it comes to using small agents.
So that is our server now defined
and our LLM now defined.
Now we're going to define our agent.
So to do that, we're going to use the server.agent decorator
and then we're going to define another asynchronous function.
And this time we're going to call this agent health agent.
Remember when we defined our last agent,
it was called policy agent.
Then when we use the client, we called the policy agent
to be able to send our prompt to it.
And again, this is going to have one argument
and it's going to be the input
and this is going to be a list of messages.
We're again going to return our asynchronous generator
and that's going to take in run yield and run the yield resume.
Then what we're going to do is begin building out our agent.
And remember, one of the most important things
is defining our dox string
because this is going to provide metadata
about the agent that we've got on our ACP server.
So we are going to define dox string.
So we're going to say this is a code agent
which supports a hospital
or to handle health-based questions for patients.
Current perspective patients can use it
to find answers about their health and hospital treatments.
Well, stop.
Cool, so that's our dox string now defined.
Now when it comes to actually building up our agent,
we're going to create a new agent
and define that as our code agent
from some more agents over here.
We then need to pass through a couple of tools.
So to the tools argument, we're going to pass through two things.
The duck duck go search tool and the visit web page tool.
Then we need to pass through our model.
So we've now got our tools defined.
We also need our LLM.
We're going to set our model equal to model
which in this case is set to our open AIGPT4 class.
Then what we want to do is we want to get our prompt.
And if you remember, when we define our insurance agent,
we're able to unpack our input,
grab the first prompt from the list
and then grab the first part from that prompt
and then grab the content value.
So if we go and do that again,
we should be able to grab our prompt.
So we're going to create a new variable
and we're going to set that equal to prompt.
And then we're going to get the input,
grab the first prompt,
grab the first part of that prompt
and then grab the content.
Then we can take that prompt and pass it through to our agent
using agent.run,
and then we're going to pass our prompt over there.
So now what we should be doing is will be getting
our standardized prompt from the ACP client will be unpacking that so that we get it into
a format that we can pass it through to our small agents code agent and then we're going
to be running it using agent.run over here.
Then we're going to return our response and again, we're going to use the message and
message part format because this is the way that we structure our outputs using ACP.
So we're going to yield message and then we're going to specify the parts and the parts
are going to be an instance of our message part.
We're going to set the content value to a string and that string is going to take in our
response that we had over here.
So now we've got a similar input structure where I'm packing it over here to get our prompt
and then we've got a similar output structure which is ACP compliant to output this when
a user goes and makes a call via a client.
All that's left to do is go and run it.
So again, we're going to use the similar Python structure.
So we're going to go if name is equal to main, then what we're going to do is we're going
to run that server and we're going to run it on port 8000.
Remember, we ran the insurance server on port 8000 on one.
We don't want them to clash.
So we're going to run this particular server on a separate server.
Now if we go and run this, that's going to overwrite whatever we had inside of the
my ACP project and specifically it's going to overwrite the small agents underscore server.py
file.
But this is really our hospital server now defined.
So how do we go about running this?
Well, similar to what we did previously for the insurance server, we're going to run
this inside of a terminal.
OK, so to go on ahead and start up our hospital server, we're going to create a new instance
of an iFrame, but this time we're going to get terminal two.
So previously we got terminal one to run our insurance server.
We're going to run the hospital server inside of a terminal two.
So when you're doing this locally, you just create another terminal.
If you're running on Mac, you might create a bash terminal.
If you're running on Windows, you might create another PowerShell instance to run this
second server.
We're going to specify the height and width of this particular terminal.
So we're going to set the width to 800 and the height to 600.
And if we go and run this, we should get our terminal back.
Take a look.
That's our terminal running.
And again, we're running inside of the my ACP project file.
So if we were to go inside of that folder, we should see our insurance server, but also
our hospital server now.
Now if we want to go and run our hospital server, we can run UV run.
And we are running what was our file called smallagentserver.py.
So if we scroll on over, you can see that we're going to be running UV run smallagents
underscore server.py and all things holding equal, that should kick off our hospital server.
And take a look.
We've now got our hospital ACP server running on port 8000 so we've now gone and defined
a second ACP server.
This one's defined using small agents and we've now started it up running on a separate
port.

## 8. Sequentially Chaining the Agent Calls
You've now gone and defined a insurance ACP server. You've also built out a hospital ACP server and
keep in mind that hospital ACP server is running on a different framework as opposed to the
insurance server. So it sort of shows that ACP can handle a number of different frameworks.
But how do we get them to talk to each other? Can we pass context from one to the other and build
sequential calls? That's exactly what we're about to cover. So we're now going to chain our agent
calls. The first thing that we need to do is once again just make sure that our ACP servers are
up and running. So to do that, we're first up going to connect to terminal one. If we go on
ahead and run this, let's double check to see if our insurance server is up and running
and take a look. Looks like it's still up and running and we can see the last call that we made
to it from a previous lesson. Let's do the same for our hospital server. Remember, the insurance
server is running using crew AI and it's a retrieval augmented generation agent. Our hospital
server is using small agents and it's using a code agent. Okay, so for our hospital server, we're
again getting import iFrame, we're going to import OS, grab the local URL and then run the terminal.
But remember, the hospital server is running on terminal two, as you can see here. So if we go
on ahead and run that, let's double check if our hospital server is still up and running and take
look. It does look like it is up and running over there.
Okay, so the next thing that we want to go on ahead and do is chain our LLM calls. We're going to
import nest async IO to make sure that we can nest our calls and apply it to our environment.
Okay, now comes the crux of it. So to go and make asynchronous calls, what we're going to do is
first make a call to our hospital server and we're going to ask, do we need rehabilitation
after shoulder surgery? And then what we're going to do is we're going to unpack and get the
response and pass that through as context into the next call to our insurance server and we're
specifically going to be calling out to our policy agent. So again, we're going to bring in our
client from the ACPS DK and we're going to bring in four from colorimer as well. We're then going
to define an asynchronous function and this is going to be called hospital workflow. So we're
going to run this when it comes to running our chain calls. This is going to return nothing.
Now, we want to go on ahead and create two clients to connect to our two servers because remember,
we've got our first server, which is the insurance server, which is running on port 80001,
and then we've got our second server, which is our hospital server running on port 8000. So we
now need to connect to both of those servers. So we're going to create an asynchronous connection
and we're going to create the first client to connect to the first server and label the client
as the insurer and then we're going to create the second client to connect to our second server
and label that as the hospital. Cool. So those are our two clients. So this is our first client,
which is going to be our insurer and then this is going to be our second client, which is going
to be connecting to our hospital. But it shows you that you can connect to different ACPS servers,
which is ultra useful if you have agents in different organizations, if you have different
specialized agents and they're running in different places, this gives you a way to connect them
more. But how do we pass context between them so that we can make sequential calls?
So first, what we're going to do is we're going to connect to our hospital client
and we're going to run a synchronous function and we're going to call out to our health agent
and then we're going to pass through our prompt. So our prompt is going to be passed through
into the input variable and we're going to set that to do our need rehabilitation after a shoulder
reconstruction. Now, once we've gone and run this particular prompt, we'll be able to unpack that
or remember, we're able to unpack using the message and message parts format.
So to get our output or to get the specific text from our output, we're going to go to our run,
we're going to get the output and we're going to get the first example because remember,
if you have multiple prompts, you're able to get multiple responses back and we're going to grab
the first part because we're sending it all as one output and then get the content from that.
Now, to make sure that we can see what we're outputting, I'm going to print this out. So we're
going to print it out in light magenta using color armor. So over here, we've got our terminal
formatting. We're then going to append the content and then we're going to reset to make sure
that we go back to white. Then we're going to run our second call. So this is effectively our
first call to our first ACP server. Now, what we want to do is make our second call and we want
to take this, which is effectively our output from our first call and append that as context to
the next one. So we're pretty much going to replicate what you can see here. But this time,
what we're going to do is we're going to connect to our insurance client and we are going to
connect to our policy agent and we're going to take this content, which we've already captured from
our first run and we're going to append it to our prompt. So prior to asking our question,
we're going to append our context and you can see here I'm just doing a little bit of text formatting
and then I'm going to pass through my second question. So what is the waiting period for rehabilitation?
So now we should know do we need rehabilitation and then we're passing that same context to our next
LLM call and we're going to print this output out as well. So we're printing out the output from
our hospital run in light magenta and we're going to print out the output from our insurance agent
in let's print it out in yellow. So again, we're going to unpack the results of that. So it's going
to be run to dot output. We're going to get the result of the first prompt and we're outputting
everything in a single part. So we're going to grab that part and we're going to grab the content
from that and then we're going to reform that terminal. So this should give us a sequential
ACP call. We're going to run the first call here. We're then going to get that context,
pass it over here and then we're going to run that second call. So we're taking the output
from this, passing it to this. If we go and run that call using ACNGIO, so we're going to import ACNGIO
and then we're going to use the run method and we're going to run the hospital workflow from over here.
Now, if this all works correctly, we should see, so we are hitting the hospital server first.
So we'll see this server running first and then we'll see this server running. So let's kick this
off and see how we go. So if we go and run this cell, so take a look, we've already kicked off
that first call. So do I need rehabilitation after shoulder reconstruction? We've got some output
from the net so you can see that we're actually calling out. We've got some results from a number
of different websites and take a look, we've got our final answer. Yes, you do need rehabilitation,
after shoulder reconstruction. You can see that's printed out in light magenta.
If we go and take a look at our crew AI agent, take a look, that's running as well over here.
And we should get our final, looks like we've already completed as well. So we do have our final
answer. We do. The waiting period for shoulder reconstruction is typically two months under the
gold cover plan. So take a look, we've now gone and sequentially run ACP calls. So this is the
result of our first call and this is the result of our second call with the context appended.
But it sort of shows you the possibility. If you need it to change different processes together,
this is the way that you'd be able to do it.

## 9. Hierarchically Chaining the Agent Calls using a Router Agent
You've now built up ACP servers and connected them together using sequential agent calls.
But what if we wanted to use a router agent to be able to navigate the different ACP servers by
themselves? How might we go about doing that? Well, this brings us to this lesson,
making hierarchical calls. So we've gone and run through sequential
chaining, but now we want to be able to go and hierarchically chain these agents together.
IE use a router agent to be able to automatically navigate how to best answer a question.
And the way that we're going to do this is we're actually going to use a prompt,
which is a combination of what we've done before. So rather than asking, do I need rehab and then
what is the waiting period? We're actually going to combine it. So the final prompt that we ask
is, do I need rehabilitation after a shoulder reconstruction? And what is the waiting period
for my insurance? So that way our router agent sort of needs to navigate between both of our ACP
servers to determine how best to answer this question. So how do we go about doing this? Well,
first up, we've got to make sure that both of our ACP servers are up and running. And again,
you've probably done this a few times now, but we want to make sure that they're up and running,
because there's a limit of 120 minutes when we're running it via this lesson. If you're running
it locally, it's as and when you are running that Python file. But for now, let's go on ahead
and double check that both of our ACP servers are still up and running. We're going to first up,
bring up terminal one, which should have our insurance server. Let's just make sure that's
still up and running. And that looks all well and good. Looks like we've still got the output
from our sequential call up and running there. Okay. Now, what we want to go in ahead and do is
make sure that our hospital server is up and running as well. So again, we'll grab our second
iframe and we're going to grab terminal two now. Let's run that. That still looks like it's
up and running. So that's looking pretty good. Now we're going to get into the ACP calling
agent. So this is a mock up router agent that I've gone and built to demonstrate how we might go
about running hierarchical workflows. So to begin with, we first up need to bring in a couple of
dependencies. So we're going to bring an async IO to be able to run our server. We're going to
bring a nest async IO so that we can nest them. We're also going to bring the ACP SDK client.
This is going to be used for a really specific reason. When we go and run this hierarchical workflow,
we're first up going to discover what agents we've got available. Then pass it to our router agent
so it can automatically navigate which agents it should be calling on which ACP service.
Then we're going to bring in small agents and I'm mainly using this so that I can access
their light LLM model class. You could also use the completions capability direct from light LLM,
but I really like the small agents implementation. Then we're going to bring in my special class,
which is fast ACP, but really it's just an example of how you might go about building your own
hierarchical workflow. So to do this, we're going to bring in from fast ACP. We're going to bring in
the agent collection and the ACP calling agent. So first up, the agent collection is going to structure
our ACP agents in a format that we're able to use them. Then we're going to take the agents from
here and eventually pass them to our ACP calling agent, which is really just a router agent. So
it automatically navigates which ACP agent is best to call to answer a particular question,
but it'll break up our prompt eventually. You should see this when we, because we've got a concatenated
prompt, it's going to break it up and work out which agent is best to answer which part of that
question. We're also going to bring in colorama and specifically the four capability. So that way,
we've got a little bit of terminal formatting. If we go and run this, it looks like we've got a
bit of warning, but that's perfectly fine. Now, let me give you a quick deep dive into the ACP
calling agent. So if I go and print out the doc string, you can see that it's a, the agent uses
JSON like ACP agent calls, similar to the tool calling agent from small agent. So I've sort of
mimicked it off that. But rather than calling tools, it's going to call different agents. So
whereas the tool calling agent is able to navigate between different tools, this ACP calling agent
is able to navigate between different agents. And it takes in the ACP agents, a model,
prompt templates, planning intervals, so on and so forth. But we're mainly going to use the
agents and the model for now. But those are our ACP calling agent dependencies now imported. Now,
what we want to go ahead and do is run our hierarchical workflow. First up, we're just going to make
sure that we're able to nest our async IO calls. So that should be perfectly fine. And then we're
going to begin building our hierarchical workflow. So again, this is going to look pretty similar to
the sequential call. The main difference is that rather than using the client directly and
sending a prompt direct to that client, we're going to use the ACP calling agent and let it
automatically navigate which ACP server and which agent on which ACP server should be calling out.
So the first thing that I'm going to define is the model that I want to go and use for our
router agent. So we're going to use the light LLM capability from small agents for this. And the
model that we're going to use is open AI and specifically GPT-4. As per usual, if you want to
use a different model, you've got the ability to do that. When I run on a Lama, I usually use
Quen 2.514b. When I use what's the next AI, I typically use one of the Lama 4 family of models.
Okay, that is our light LLM model now defined. Now, we're going to define our hospital workflow.
So we've defined this previously when we did our sequential API calls, but we're going to do it
again. This time, we're going to focus on our ACP calling agent or our hierarchical router type
agent. So let's go ahead and do this. So we're going to define an asynchronous function.
And this is going to be called run hospital workflow. It's going to output none. And then we're
going to create two clients to connect to our servers. So our first client is going to connect to
the local host at a 2001, which I believe is the insurance server. We're going to label it as
the insurer. And then we're going to create another client to connect to our other server, which
is our local host port 8000, which should be our hospital server. And you can see we're connecting
to that over there. Perfect. Okay, then what we want to do first is we want to get all of our
different agents. So to do that, we're going to use the agent collection over here. So let's go
and define an instance of this. So we're going to define a new variable called agent collection.
And we're going to await an agent collection dot from ACP call. So to this method,
we're going to pass all of our different ACP clients. So we're going to pass through the insurer
and the hospital. So these are our two different clients. Now via the agent collection, we're able
to discover all of the different ACP servers that we've got available and all of the agents that
we've got available on those ACP servers. So if we now go and reformat this into a method that we
can use to pass to our ACP calling agent, before we pass it to our agent, let's print it out.
And so let's go and loop through each one of these. So we're going to four client
an agent in our collection dot agents. We're then going to create another dictionary and that
dictionary is going to have the keys of the agent name. And it's going to have two parts to it.
It's going to have the agent itself. And then it's going to have the client. So once we know
which agent we want to call, we want to make sure that we use the right client call out to it.
So we're going to append a client key to that as well. So this means that if our router agent is
like, Oh, I should call the policy agent. Well, it's going to be able to grab the insurer client
and call out to that policy agent. Now let's go ahead and also print out these ACP agents. So you
can sort of see what they look like. And what we can do before we actually go and call to
out to our ACP calling agent, let's just make sure that we get these agents back. So if we go
and use async.io.run and run our hospital workflow, we should get back our ACP agents. So let's
run this. So we've now gone and discovered all of our agents. So you can see that we've got our
policy agent over here and we've also got our health agent over here. And you can see that we've
also gone and appended our clients. We've got a bunch of information. Remember how I said that
when you go and define that function, it's going to take on the name of that agent. So you can see
here that this is directly from ACP. So it's grabbing the name of the agent, but it's also
grabbing the doc string. Remember how we defined our doc string that is coming through when we go
and discover our different agents. So down here inside of this description, you can see that
our policy agent is described as this is an agent for questions around policy coverage,
uses a rag pattern to find answers based on policy documentation, uses to help answer questions
on coverage or waiting periods. So we're able to not only call out to these different agents, but
also discover them. So how do we go about making this hierarchical call now? Well, let's go on
ahead and do that. So we're going to define a new agent now. And this is going to be our ACP
calling agent, AKA a router agent. And that is going to be an instance of the ACP calling agent,
which will take in our ACP agents, which we've defined over here. So it's all of them, right,
across both servers. And then we need a pass through which LLM that we're going to use. So we
are going to be using GPT4. So we're just going to define that as model is equal to model. So our
model argument is equal to this model over here. Then what we need to do is pass through our prompt.
So we are going to create a new variable called result. And we're going to wait a ACP agent call
to that. We're going to pass through our prompt, which is going to be do I need rehabilitation
after a shoulder reconstruction. And what is the waiting period from my insurance? So this means that
so there's a key difference here, right? So when we previously asked our prompts, we separated
the amount and targeted them to the specific agents on a specific ACP server. Now we're just
combining them all and allowing our router agent to determine how it should best answer that.
Because there's really two questions there. Do I need rehab after a shoulder reconstruction? And
then what is the waiting period for that rehab if I need to get it? Now what we're going to do is we're
also going to print out the outputs of the final result. So we're going to print it out using color
armor. So we'll print it out in yellow. And we're just going to construct it as a formatted string.
So we'll get our final result. We'll append the result that we've got from here. And we'll just
make sure that we reset our terminal. So now we've got our hierarchical workflow defined. So if
we go and run this cell again. So we've now not only just doing discovery, we're also actually going
and triggering that call. So if we go and run this cell again, we should see these calls go back
out to our agent. So we're now going to be calling out to our insurance client, out to our hospital
client to be able to go and answer our concatenated prompt. So let's go and run this. So we've discovered
our agents again and take a look. You can see that we're calling out to our different agents.
So we've got our input being sent through to our health agent, which is our hospital server.
Looks like we've got some responses back. And if we go and scroll up to our server as well,
you can see that we're actually triggering this. So our first response, yes, rehab is typically
required. And if we go and take a look at our previous one, looks like our crew AI is running.
Let's see. There we go. The waiting period for rehabilitation after a shoulder reconstruction
is two months. So we have our final output. Let's scroll on down.
There's our final result. So you can see that it's automatically navigated between the two
different ACP servers to generator response to effectively a prompt that's being concatenated
together. So our final response is yes, rehab is typically needed after shoulder reconstruction.
It includes physical therapy and regular exercises to restore motion and flexibility
to the shoulder and to return it to everyday activities. In terms of insurance coverage,
the waiting period for rehab after a shoulder reconstruction is two months as per our clinical
categories coverage. Remember, that's from the RAG agent. This includes a provision for medically
necessary treatments. However, if the condition is a pre-existing one, then you would have to wait
a period of 12 months. So we've now gone and automatically navigated through our different ACP
servers and the agents on those servers using a hierarchical calling agent.

## 10. Adding MCP to the Hospital Server
So you've covered sequential calls and hierarchical calls,
but there's one thing we haven't quite covered, MCP.
How can we get ACP and MCP to work together?
That's exactly what we're going to do now.
We're going to convert our hospital ACP
server to use MCP as well.
So there's a lot of talk about ACP and MCP.
Well, remember, ACP allows agents to communicate to each other
or provides an agent communication protocol.
MCP provides a communication protocol for mainly tools.
So we can actually use them together.
The first thing that we're going to need is a MCP server.
So what we're going to do is we're once again
going to use a Jupyter Notebook Magic function
to write out our MCP server.
Now, this MCP server is going to allow users
to find doctors near them.
And it's going to use this particular capability
to overwrite our hospital server.
So how do we go about doing this?
Well, first up, you're going to use the right file function
to go and write out MCP server to the MyACP project directory
because we're eventually going to need our ACP server
to communicate with our MCP server
file via standard input output.
So then what we need to do is define the MCP server.
So first up, what we're going to do is import colorama
and we're going to use for for some terminal formatting.
While we're at it, we're also going to import
the fast MCP capability from the MCP SDK.
That's going to last to define our MCP server.
We're also going to import JSON to handle the data
that we're going to be using.
And I'll come back to this in a second.
And we're also going to need requests
because the data that we're going to be returning
using this particular MCP server
is currently served via GitHub.
Now, what exactly is this MCP server going to do?
Well, it's going to return a list of doctors
and we're going to use it to return the most relevant doctor
to a particular person based on their state,
assuming they live in the United States.
So let's go and define this tool.
So first up, we're going to create our MCP server.
So we're going to create a new variable called MCP
and set that equal to fast MCP,
which we've just imported over here.
And we're going to name the server doctor server
or being a little bit verbose here.
Then what we're going to do is we're going to build up our tool.
So we are going to use the MCP tool decorator
and then create a new function called listdoctors.
This is going to take in one argument,
which is going to be a state as a string
and it's going to return a string back.
Then you're going to provide your doc string.
So in this case, this particular tool returns doctors
that may be near you.
It's going to take in state, as we've mentioned,
as a true letter state code based on where you live
and it's going to return a list of doctors
that may be near you.
So how are we going to do this?
Well, I've currently got the data stored on GitHub.
And you can see that we've got a big JSON document
of a number of fictional doctors
that we're going to be able to return back.
So in order to bring that back via our MCP server,
we're going to make a request out to that URL.
So that's the URL there.
So it's raw.githubusercontent.com
Fordsashmygithub, which is Nicknocknack.
And then we're looking at the ACP walkthrough repository.
We're looking at the refs, the heads main,
and doctors.json, that's where that file resides.
Then to bring that back via our MCP server,
we're going to make a get request to that URL.
We're then going to load that response back via JSON.loads
and we're specifically extracting the text from the response.
And that should give us a list of doctors.
But remember, what we want to do
is we want to filter through based on the user's state.
So we're going to loop through each one of those doctors,
probably not the most efficient way to do this,
but this gives you an example of what's possible.
We're going to loop through each one of the doctors
inside of that list.
So four doctor in doctors.values.
If the doctors address and specifically their state
matches the state that we've passed through to our MCP server,
we're going to return that doctor back in a list.
And that's going to be stored inside of a variable called matches.
And then what we want to do is we want to return that back
as a string.
So that's pretty much our MCP server done
in terms of the core functionality.
Now, if our ACP server calls this via standard input output,
we want to make sure that we run it using if name,
equal main, then we're going to use mcp.run,
and we're going to set the transport type to STDIO.
So that's pretty much our MCP server now created.
Now, if we run this cell,
it's going to override it inside of our MyACP project repository.
So that that MCP server is available to our ACP server,
which we're going to update next.
Now, that brings us to our ACP server.
We're going to take our existing hospital ACP server
and add in another agent.
But this second agent is going to be leveraging MCP
as well.
So remember, we define our initial small agents agent,
which was a health agent,
and this was sort of there to help handle
general hospital-based queries.
You can see that that was our doc string.
Current or prospective patients can use it
to find answers about their health
and hospital treatments, so on and so forth.
Now, this second agent is going to be using the MCP server
that we've just gone and defined.
So we're probably going to define a doctor-based agent.
So how might we go about doing this?
Because so far, we've only really had one agent
per ACP server, but you definitely can have more.
So we're going to add a second one.
So the first thing that we need to do in order to do that
is bring in a couple more dependencies.
So from small agents so far, we've had the code agent,
the duck duck go search tool, light LLM,
and the visit web page tool.
Well, we're also going to bring in the tool calling agent
and the tool collection.
These two are going to help facilitate working with MCP.
Now, whilst we're at it,
we're also going to bring in an MCP dependency
because we're going to need to connect to our MCP server
using STDIO.
So we're going to bring that in.
So we're going to go from MCP,
we're going to import STDIO server brands.
This is just going to make it a little bit easier
to help define how to connect and run it against our MCP server
over here.
So how exactly do we do this?
Well, remember, when we went and export our MCP server,
we defined it as MCP server.py.
And it's running inside of the My underscore ACP,
underscore project directory.
So we need to be able to connect and run our MCP server,
which the command for that would be UV run MCP server.py.
So we can construct that command inside of the STDIO server
parameters class.
So let's do exactly that.
So I'm going to define a new variable called server parameters.
And I'm going to set that equal to STDIO server brands.
And what you're going to do is create a new argument,
which is going to be command.
And that is going to be equal to UV.
So that's the first part of the command
that you need to be able to access the MCP server.
So what are the other two?
Or remember, the full argument or the full command
was UV run MCP server.py.
So we've got our command.
We now need to pass through the arguments.
So two our args keyword argument.
We're going to set that as a list.
And we're going to pass through the last two arguments,
which are going to be run, and then MCP server.py.
So this is effectively going to allow us to run
and access our list doctors tool inside of the MCP server.
Now, we also want to pass through one last keyword argument.
And that is going to be EMV.
And we're going to set that equal to none
because we don't have any environment parameters.
Now, what we're going to do is we're
going to define our next agent.
So so far, we've had our first health agent,
which you can see over there.
We're now going to create another agent.
But this one is going to use MCP.
So this first one is using our code agent.
The next one is going to use the tool calling agent
and the tool collection.
So in terms of the structure, it's pretty similar.
We're just going to sub out the agent
and get it to use MCP.
So let's begin doing that.
Well, first I'm going to define a decorator.
And we're going to set that equal to the server.agent decorator.
We're then going to create a new function.
And this is going to be an asynchronous function.
It's going to be defined as the doctor agent.
Remember when we went and discovered our agents
were able to see that they pick up the name of the agent
or the name of the function as the name of the agent.
So in this particular case, if we went
and rerun that hierarchical workflow,
we'd see the doctor agent popping up
with our new hospital server.
And then as usual, we're going to pass through our same input.
This is going to be a list of messages.
And that is going to return an asynchronous generator,
which is going to have parameters of year run yield
and run yield resume.
Then we're going to define a doc string.
This one's going to be pretty simple.
So this is a doctor agent, which helps users find doctors near them.
Now, what we need to do is start defining our tool collection.
Now, our tool collection is going to last
to discover tools on the MCP server.
So to do that, we're going to create a new with statement.
And this is going to connect to the tool collection
using the from the MCP method.
We're going to pass through our server parameters,
which we've just defined up here.
And then we're going to set the trust remote code parameter
equal to true.
So that's going to allow us to access those tools on that server.
And we're going to run this as the tool collection.
So we'll be able to access it using this shortened statement.
Now, what we need to do is we need to pass these tools
to our tool calling agent.
Because so far, we haven't actually defined an agent.
We've defined the tool collection, but not the agent itself.
So let's do that.
So we're going to create a new agent.
And this is going to be equal to tool calling agent.
To that, we're going to pass our first keyword argument,
which is going to be tools.
And this is going to be a list, which
is going to be an unpacked set of values
from our tool collection.tools value.
So effectively grabbing all the tools
from the tool collection over here,
and we're passing them to our tool calling agent over here.
Now, we also need to give our tool calling agent an LLM.
So we're going to set that parameter as model
is equal to model.
That's just going to take in our GPT-4 LLM over here.
But again, you can go and update this to what you want.
You can use Obama.
You can use what's next.
You can use a number of different providers.
ACP is framework and provider agnostic.
OK, so that is pretty much our agent now done.
Now, we also need to go and extract our prompts.
Now, remember, because we had our input,
we can go and grab the first prompt,
and then grab the first part of that prompt,
and then the content value, and that should be our prompt.
So let's do that.
So we're going to grab our prompt variable,
and we're going to set that equal to input 0,
and then part 0, and then we're going to grab the content,
and then we're going to send that to our agent.
So to do that, we can create a new response variable.
And we're going to set that equal to agent.run,
and we're going to pass our prompt to it.
So that's pretty much the crux of our agent now using MCP.
The last thing that we need to do is yield this back to our user
when they go and run the server.
So we're going to go yield message,
and that message is going to be constructed
of a number of parts.
The main message part is going to take in all of our content,
which is going to be the response from our MCP,
or our tool calling agent.
So that is that there.
Perfect.
It should be our MCP agent now done.
Now, if we're going to run this,
that is our small agent server now updated.
So now, if we go and take a look at that small agents server.py file,
maybe go and take a look inside of that repository now.
So it's going to be available for you to take note and play around with.
So if you go to my ACP project,
and then go and take a look at this small agents.server.py file,
you'll see that we now have a doctor agent.
So you can see that we're now running two agents on this one ACP server.
Now, we need to make sure that we go and rerun the server
now that we've gone and updated it.
So if we go and open up our terminal again,
we're going to make a quick update.
Remember, the environment's going to stay up for 120 minutes.
If you go away from this lesson and come back,
just make sure that you double check that the server is still up and running.
And then we're going to render that I frame.
So remember, we're going to be running on terminal two
because that's where we've run the hospital server so far.
If we go and run this, we're going to bring up our terminal.
Take a look, that's available there.
What we'll do is we're just going to stop that server.
We're just going to run clear to clear that terminal.
And then we're going to make sure that we rerun our server.
So remember, it's been export to small agents
underscore server.py.
So let's just make sure we run that particular server.
So we're going to a UV run small agent server.py.
And if we go and run that, let's see if we get any errors.
That's looking pretty good so far.
So we can see that our server is now up and running
and it's running at port 8,000.
All that's really left to do is actually make a call.
Now remember, when we've gone and connected to our MCP servers
before ACP servers before, we've gone and used the ACP SDK client.
That's exactly what we're going to do here.
But remember, now that we've gone and defined another agent,
we're just going to make sure that we target this agent
rather than the health agent.
So let's go and do that.
To begin with, we need a couple of dependencies.
So let's bring those in.
So we're going to bring AC and IO to make the call.
We're going to bring Nest AC and IO so we can nest them.
We're going to bring in the client from the ACP SDK.
And we're also going to bring in Colorama.
So we can print it out using a colored output.
We're then going to apply Nest AC and IO.
And that should be pretty much it when it comes to our dependencies.
Now, when it comes to actually making our call,
we're going to create a new client.
And this is going to be a new function
which allows us to call out the main differences.
We're going to change the agent that we use.
And we're going to change the prompt that we use.
So let's go and define this.
We're going to create an asynchronous function
called run.doctor workflow.
This is going to return nothing.
And that is going to connect to our client using a with statement.
And remember, we're still running a port
8000 so we can connect to that there.
But also take note that we're getting this warning.
Cannot reach server.
Check if running on HTTP127.0.0.1.
port 8333.
This is occurring because when we start up our HTTP server,
it's automatically going to try to register our agents
onto the BAI platform.
This provides a register for your agent
so that you can call them via UI but also via the command line
and orchestrate them all together.
We're going to start this up in the next lesson.
But for now, just be mindful of this warning.
It won't affect the running of this code.
So let's keep going and connect to our hospital server.
So we're going to create a base URL.
We're going to set that equal to HTTP, colon,
forward slash, forward slash local host, 8000.
And we're going to run that as the hospital client.
Then what we're going to do is we're going to make a synchronous call
and call out to our hospital client.
So this is going to be stored inside of a variable called run1.
We're going to set that equal to await hospital.runsync.
We're then going to pass through the agent that we want to call out to,
which is going to be the doctor agent.
Because remember, we've gone and defined a second agent now.
We're going to call out to the doctor agent rather than the health agent.
So we're going to hit that agent and now we pass through our prompt.
So our prompt is going to be, I'm based in Atlanta, Georgia.
Are there any cardiologists near me?
Question mark.
So that's going to be our prompt.
And keep in mind, this should ideally take the state, Georgia,
and go and find whether or not we've got any doctors or cardiologists
that are available in that particular state.
We haven't actually gone and done a filter for the type of doctor.
You might go and choose to update that and see how that performs.
For now, we're just very much focused on state.
So what we can go and do is then we just need to print out our content.
And remember, because we're getting the message parts and the messages back,
we want to make sure that we get the result from our first prompt,
because we're only passing one through and the first part,
we're only outputting everything as a single part,
so we're just going to grab everything.
So if you're doing streaming, you'd be able to grab each one of those parts
as they generated.
OK, so let's create a new variable called content.
And we're going to set that equal to run1.output.
We're going to grab the first message.
We're then going to grab the first part and then grab the content.
And then we're going to print it out.
So we're going to use light magenta to print it out to our terminal.
And we're going to print out the content and reset the terminal
so that we go back to our regular coloring.
OK, cool.
Now if we go and run that, that is our workflow now defined.
All that's really left to do is actually go and run this workflow.
So to do that, we can use async.io.run and to that,
we're going to pass through our run doctor workflow function.
And if we go and run that, take a look.
Looks like the run started.
We're searching.
Looks like we've found a doctor.
Do we get a final response?
There we go.
So we've now got a final response.
Yes, there are cardiologists near you in Atlanta, Georgia.
For example, Dr. Sarah Mitchell, who specializes in cardiology,
is based at 1, 2, 4, 7 Medical Center Drive at Atlanta, Georgia.
She has 15 years of experience and is board certified.
Now I wonder if we go back to the data set
if we can find that particular doctor.
And take a look, that's actually our first result.
So you can see that we've gone and found that first doctor.
And they just so happen to be a cardiologist.
But we haven't actually focused on that.
We've filtered purely based on the state.
So what you might go and do is try updating the prompt.
See if you can find a doctor in Arizona or in California
or in Colorado.
You should be able to use your doctor agent to now do that.
But we've now demonstrated how we can use MCP and ACP together.

## 11. Managing ACP Compliant Agents
Okay, so you've gone all the way. You've built an ACP server, you've called out to it via a client,
you've created a sequential call and even created a hierarchical call. We're also updated
that server to be able to use MCP, but what if we could call those agents via UI and have them
centrally managed inside of a repository? Well, this is where the B.A.I platform comes into play.
In the last lesson, you were able to discover agents on an ACP server using the Python client,
but what if there were other ways to do this? Well, one of these ways is via registry.
Think of a registry as a centralized store where you can access a range of agents, which may
have different use cases, tools and capabilities. The registry also allows you to centrally manage,
deploy and search for agents, and in the case of the registry I'm about to show you,
you can also perform offline discovery. This means you're able to search for agents without
network connectivity. The registry that you'll be learning about is provided via the B.A.I platform.
It has its own built-in registry, but also provides a user interface to run and manage agents,
as well as providing the ability via ACP to have them chain sequentially and hierarchically.
There's a few different ways to install it, depending on what operating system you're running on.
If you're running on Mac OS or Linux, you can follow these instructions here or these here.
If you're using Windows, you can follow this set of instructions. Now, to start the B.A.I platform,
you can run B.A.I platform start and walk through the setup instructions. I'm going to use what's
the next.ai and specifically Lama4 on the platform, but you can use whatever you'd like.
Now, build right in. You've got a few popular agents right out of the box,
like GPT Researcher for Research, Ada for Programming, and the podcast creator agent for
you guessed it, creating a podcast. If we choose one of them, let's say Ada, we can pass through
a prompt and run it, and you've got your agent run.
The thing is, you've gone to all that effort to build your own ACP compliant agents,
so can you add them to the registry inside of the B.A.I platform? Sure, Ken, let's go do it.
So, we're going to bring in our ACP agents into the B.A.I platform to do this. First up, I've
gone and replicated the deep learning.ai environment on my local machine, and I've updated this to
use some different LLMs. So right now, I'm using the Watson X Lama4 instance in my small
agent server, and I'm doing the same inside of my crew agent server. You can see that here,
and you can also see that here. Now, what I want to do is I want to sort of show you what's
possible with B.A.I. So the first thing that we're going to do, or the first command that I'm
going to run you through is how to go about installing it if you want to run it locally.
So to do that, you can run Brue, install B.A.I, and go on ahead and run that. It should run on
your machine, and then you can go and kick it off by running B.A.I platform start, and that will
kick the B.A.I server off. Then you'll be able to run the command B.A.I, and all of it's derivative.
So if we go and run B.A.I to begin with, you can see I've got a number of different options.
I've got a number of different commands. I've also got a number of different agent commands.
We're going to focus on the agent commands for this particular lesson. So over here,
if I go and run B.A.I list, I'm able to list all of the different agents that I've got available.
So let's go and try that. So if I hit clear, and then run B.A.I list, you can see that I've got
a number of agents to the complete package inside of B.A.I. You can see I've got an agent documentation
creator. I've got Ada chat, GPT researcher, and a bunch more. Now we can actually go and run these
by running B.A.I run and then the agent that you want to run here. But I want to focus on using
the agents that you've already gone and built throughout these lessons rather than using one of these.
So before we go and run that, let's actually go and make the ACP agents that we've already gone
and created compliant with the B.A.I. platform. So let's go and do that. So inside of our small
agent server, I'm going to import the metadata capability, and this is just going to allow us
to define what the documentation is associated to that particular agent and how to go about running it.
So if I bring in the metadata capability there, and then we're going to scroll on down to our
health agent. Now inside of the agent decorator, I'm going to set the metadata keyword argument,
and that's going to be equal to our metadata class, and then we're going to set what the UI
value is. And I'll come back to this in a second. What that UI is going to dictate. But for now,
stick with me. The type is going to be a hands-off agent. There's other types of agent. There's
also chat agents. And the user greeting, this is going to be almost like the prompt greeting,
is going to be ask your health question. Now I'm also going to set the same or a slightly similar
one for our doctor agent that we created in our small agent server. So I'm going to set the metadata
and set that equal to metadata. The UI is again going to be a hand-off agent. And then what we're
going to do is we're going to set the user greeting for this one. Find a doctor, pass your query,
and state here. So we've now got our type, and we've also got our user greeting set for our doctor
agent. Now let's quickly go and do the same for our crew AI agent. So we're going to, again,
import metadata over here. And then inside of our agent decorator, we're going to set those same
parameters. So we're going to set the metadata keyword argument, provide the metadata class,
we're then going to set the UI type to hands-off. And the user greeting, we're going to have a
slightly different one. Go to question about your policy, ask here. Okay, so that's pretty much it
when it comes to going and setting up our agents. Now the beautiful thing about this, and you might
have seen it when we went and built our last agent and updated it for MCP, that we weren't actually
registering the agents. That's because BAI wasn't running on the deeplearning.ai platform,
but I've got it running locally. So what does this actually mean? Well, if I go and start up
our two agent servers, so let's go and kick off our small agent server. If we go UV run small
agents server.py, we should get a server up and running and take a look. We are running over,
yeah, you can see we're running on port 8,000, but we've got all this extra information. We're
going to come back to that in a second. Let's go and start up our crew agent server as well. So if
we go UV run crew agent server. Got a bit of a warning, but that's okay. Looks like we're now
running. Let's maybe zoom out so we can see that a little bit more clearly. So we've got our server
running right over here. So we've got our server running on port 8,000 and one we've got our other
server running on port 8,000. Now the cool thing about this is that if we go and run BAI,
let's actually create a move BAI into its own separate terminal. So if we now go and run BAI
list, take a look. We've now got our doctor agent, our health agent, and our policy agent now
described or now available inside of BAI. So if we wanted to go and run one of these agents,
for example, let's go and run our doctor agent. We can go and run BAI run doctor agent and take a
look. We've got a greeting find a doctor, pass your crew in states. So let's try the same prompt
that we had within when we actually want to build that server. So I'm based in Atlanta.
Georgia, are there any cardiologists? Yeah, me. Okay, so if we go and take a look at our server,
take a look. It's actually kicked it off. It looks like we've got a response. And if we go and
jump back, it looks like we've got a response. Yes, there are several cardiologists near you and
Atlanta. Georgia, one of them is Dr. Sarah Mitchell, who's a board certified and has 15 years of
experience. So you can see that we're now able to add our agent to the registry, but we're also
able to run them. If we went and ran our policy agent, for example, we could go and do that. So
if we go and run BAI run policy agent. So you can see that we've got our greeting there. So I've
got a question about your policy. Ask here. So we might go and say, what is the waiting period
on physiotherapy? And if we go and jump back to our servers, so you can see that that is now
running. It's looking like it's going and searching through our knowledge base. I'm not actually
sure whether or not we've got that inside of our vector database. But let's see if we get a response
back. So take a look. We've got our agent running. Do we have a final response? Take a look.
The waiting period on physiotherapy is two months group physiotherapy is covered with a limit of
$35 per visit. So we've now gone and rendered a response using the BAI platform. But could we
also use this inside of a UI? Well, if we actually go and run BAI UI, this is going to open up a
separate UI where you've got the ability to go and use the server here. And if we go and take a
look, all of our agents are going to render over here. So we've got our doctor agent. We've got
our health agent. And we should have our policy agent. So if we went and ran it over here, we can
go and say, what is the waiting period on what did we want to say? What is the waiting period on,
I don't know, dental. Actually, let's just stick with the free rehabilitation. And if we go and
run this, so this is going to go and kick it off. If we go and take a look at our servers, take a
look, they're running, but we're now running it inside of the UI. And we've got our final result
back. The waiting period for rehabilitation, which falls under the hospital substitution programs,
is generally two months of continuous cover. Unless it's pre-existing condition, in which case,
it is 12 months. So we've now gone and built it out. And we've actually run it inside of the BAI
platform and inside of the UI. So you can see they've now got the ability to assign it into the
registry and run it inside of a nicely designed user interface. On another note, definitely go check out
the resource section. At the end of the course, I'll make sure to link to the documentation,
the ACP protocol, the BAI platform, and my ACP GitHub Reaper, as well as a few other helpers.

## 12. Conclusion
We hope that getting hands-on and building ACP agents has sparked your creativity
and gotten you excited about what's possible when agents can collaborate effectively.
Throughout this course, you've explored why agent communication protocol is valuable,
learn some of its core principles, and actually build and ran your own ACP agents.
If you're ready to take the next step and join the growing community shaping the future of ACP,
here's how you can get involved.
Stay in the loop by checking the GitHub repo, join the Discord channel,
and jump into the discussions on GitHub.
If you've built something cool, share it with us in the show and tell.
Contribute directly by checking out open issues labeled help wanted,
or if you've got your own ideas you'd like to explore, start a discussion.
Lastly, give feedback. We actively encourage it.
If you run into bugs or problems, open an issue on GitHub,
and we'll do our best to resolve it quickly.
ACP is open, governed, and community-driven.
It's being built by the community for the community.
So whether you're building, contributing, or just curious,
there's a place for you in this ecosystem.
Thanks for learning with us, and we can't wait to see what you built next.
