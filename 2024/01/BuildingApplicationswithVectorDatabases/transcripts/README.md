# Building Applications with Vector Databases

## 01. Introduction
Welcome to this short course, Building Applications with Vector Databases built in partnership with
Planko. Vector databases have become a key part of the infrastructure stack for building
applications with large language models, specifically RAG or Retrieval Augmented Generation.
But it turns out that they're useful for even more applications than is widely appreciated.
In this course, you learn how to use Vector DBs to quickly build applications in text similarity
search, RAG, image similarity search, anomaly detection, recommended systems, and hybrid
search, which refers to using simultaneously sparse and dense vectors.
I know that's a lot of stuff, but that's why I hope you find this course useful. We'll
cover a huge space of applications that you'll be able to build using Vector DBs.
I'm thrilled that the instructor for this course is Tim Tully, a board member at Pinecold.
For over a decade, Tim had led teams of thousands of engineers as Spunk's CTO and then Yahoo's
Vice President of Engineering. Tim had also developed a Pinecold Vector Databases
command line interface, and I look forward to him sharing of us what he's been seeing
from working with many people building on top of Pinecold's Vector DBs.
Thanks, Andrew. I'm excited to be here.
I know it gave a long list of applications just now, but let me unpack briefly some of
the key topics. Vector DBs lets you input a collection of vectors, say embedding the
documents if you're implementing RAG, and then same queries to it, such as an embedding
of a new text query to retrieve documents relevant to your query. That's what made
Vector DBs a key component of RAG where you fetch additional context for an OM to generate
this response. But it turns out that disability to fetch similar vectors makes it very useful
for many other applications as well. For example, in image similarity search, you can compute
embeddings of images, and then use a vector DB to quickly find similar images. Or you
can carry out anomaly detection by checking if a new item has anything similar to it, and
if it doesn't, then maybe as an anomaly. There are a lot of details on how to implement
these effectively, and Tim will dive into the specific mental frameworks and tips for
building these applications. Yes, as Andrew said, there are a lot of potential applications
for vector databases. You learn about six of them in this course. First, you'll start
by building a basic semantic search for text documents. Then you'll move into building
a RAG app. In the next step, you'll build a recommender system. Next, you will implement
a hybrid search app for product recommendation. Then, you will build a child parent image
similarity app. And finally, you will work on an anomaly detection app based on a data
set of server logs. Through these examples, you'll learn how to use a vector database
to store and help you manipulate lots of different data sets. We'll draw up some examples
from Wikipedia text, pictures of faces, core text, structured data in the form of network
logs, and a fashion data set with paired images and text. One of the cool techniques that
you'll learn about as well as well is hybrid search in which you can use a vector database
to manipulate vectors with both a sparse and a dense component. For example, in the product
recommendation setting, you'll see how to use a dense embedding of images of clothing
alongside a sparse embedding of a text description and how to dial a knob controlling the relative
weighting between the sparse and dense parts of the vector. By learning the foundations
of vector databases and working on the applications you'll learn in this course, I'm sure you'll
find even more possibilities to explore. Many people have worked to create this course.
I'd like to thank on the Pineconside James Break, Raan Shihashar, and Baron Douglas. From
deeplearning.ai, Diala Ezzedin, and Ashma Gugari have also contributed to this course.
In the first lesson, you will explore the essentials of semantic search, which will be used throughout
the other lessons in this course. That sounds great. And a lot of materials in this one
short course. So let's go add a new dimension to what we can do using vector DBs. Let's go
on to the next video and get started.


## 02. Semantic Search
Semantic search is a type of search that focuses on the meaning of the content being searched
as opposed to lexical search, which looks for literal or pattern matches to strings.
This is an extremely powerful construct that is underlying most of what we see from text
oriented to generated the AI applications.
In this lesson, you will be introduced to the essentials of vector databases and semantic
search, which will be used throughout the next lessons in this course.
Let's dive in.
So we're going to build a very simple semantic search application using Pankham.
What we're going to do is start here on the left hand side with the user and download
a core data set and turn it into a series of vector embeddings and store it in Pankham.
Then we're going to build a simple question and answer response system and we'll ask
which country has the highest population and this will be a generic enough application
that you can ask multiple questions against it as well and so you'll see how that works
in the example that we showed in just a moment.
So let's start off building the application.
So what we'll first do is pull in the import warnings so that we don't get warnings
when we run the application and then pull in the rest of what we need and what we'll rely
on here.
So we're going to use sentence transformers.
There's a small helper utility that we created called deep learning AI eTils that handles
some Pankham and open the AI key management for us.
That works great.
And then we'll use tqdm to see the progress of a lot of this work that we're doing.
So let's download the core data set and you'll see that we're taking here a subset of
the data between rows 240,290,000 because it's so large and we want to make it manageable.
And what we can do now is take a look at the data set and see what we have.
So you can see here we looked at five rows of the data and we can see there's questions
or there's identifiers for the questions.
All looks really great and unusable.
So let's try to prepare the data to ask questions.
So what we'll do is get all of our questions.
So we'll iterate over all of the questions and add it to essentialize to read that has
all of them in it and we'll use extend here because it has multiple elements in it.
And then we'll build a list out of it and then join it with the control, the new line
character so that we can look at it and see where we have.
This is just a small delimiter here.
This can be really as long as you like, but this makes it easier to see.
This is going to put a hyphen across 50 times and then let's look at the data.
Oh, have a small listing of column there.
All right, so those are all the questions that we have and then we'll prepare.
We'll see that we have 100,000 questions prepared.
So now let's get ready to turn this into embeddings before we send it into pine cone.
So depending on the device you have at home, you may or may not have CUDA available for
you.
And if you do, some of this will go a lot faster for you if not, it's not a big deal because
the data set's not that large.
You'll remember we took a subset of it.
Oops.
And now we will use sentence transformer to transform this data into embeddings.
All right, no CUDA, that's okay.
We'll move forward.
That is not that large, but we have our model ready to go.
And so now what we'll do is actually create a sample question and turn that into embedding.
And then we'll look at the embedding and we'll see hopefully that it's 384 elements
wide.
So now that we have an embedding, we have our data, we have our questions, we're going
to start to use pine cone a bit.
And so we have a small utility that we prepared for you called the L-A-I-U-Tills and we'll
instantiate it here.
And then we'll get our API key for pine cone to use.
And that's all you'll really need to use to use pine cone.
Okay.
Okay.
So now we're going to connect the pine cone.
And what you'll see here is the syntax is slightly different now.
There's what we call a singleton object in in pine cone and it just takes the API key.
This is because it's connecting to what's called pine cone version three, which is a so-called
serverless version of pine cone.
So what we'll do now here is use that utils package that we talked about earlier, the
deep learning AI tools and create an index name out of that.
And we'll get our index name and then we will start to connect to pine cone.
But before we do that, we're going to do a little bit of house cleaning and delete
old indexes in case you're doing lessons out of order.
And so what the so-do is look for the index name that we just declared and see if it
already exists in the pine cone index will actually just delete it.
And then we will create our index and let's just keep it clean here.
This is saying we're going to use I happen to know the answer to this dimension parameter
is 384 when we saw that earlier.
We're going to use cosine distance as the metrics to find similarity.
And then there's a new object in pine cone called the serverless spec.
And the serverless spec simply just specifies what public cloud that your index runs on.
In this case, it's AWS and us west too.
And now we will finally get a pointer to our pine cone index.
So say index equals pine cone index and we used, we'll use the index name that we created
just a moment ago.
And if we say index, okay, great.
If I say print index, we should get back a pine cone index object and it's going to
take a little bit to run.
All right, there's our index object.
Finally, after we've created our pine cone index, we can upload some data.
So we will upload data using that size of 200 and a maximum of 10K factors.
And so now we will get a subset.
We said we're just going to use 10K, right?
And then what we're going to do is iterate through our questions, okay?
Find the end of the batch that we're looking at.
Generate IDs for each vector.
Each vector inside a pine cone needs a unique identifier.
In this case, we'll just use the index number.
We'll store inside the metadata feature of a pine cone, the actual text from the question
next to the vector embedding.
And then use the model that we created above to generate the vector embedding and store
it all together.
So think of it as a tuple of an ID vector embedding in the metadata that goes with it.
And we'll actually zip up that tuple using zip and then absurd it into pine cone.
And so if we run this, you'll see TQDM kick in and this will upload all of our embeddings
into pine cone.
This will take a minute to upload.
There's a lot of data here that we're uploading as well as the model has to run.
So there's some encoding for the model that has to happen.
There's a round chip that has to happen to pine cone.
So we'll just wait here for a minute for that to happen.
All right, so that finished.
And let's check to see what we've stored.
So there's a function called describing the X stats.
And we have 10,000 vectors just as we match is what we have here above inside a vector
element.
So that worked great.
So now we can finally get to the to the point where we can ask questions.
And so I've defined something very small that we'll use here that executes queries for
us called run query.
And all this does is takes the text version of the question that we want to ask.
That's query builds a vector embedding out of it that we'll need.
And then runs the query against pine cone.
So the question is becoming embedding.
We pass that embedding into pine cone.
We ask for the top K. In this case, we're asking for 10.
You can feel free to change that to any number you like.
You can get more results back.
And then we ask for the metadata, which actually had the question in it, right?
It's useless for us to just look at the vector embedding.
We actually want to see the text as well.
And so that's what's inside of the metadata.
And we want to say include values equals false because what that does is it makes it so
that we don't get the embedding back because there's nothing to do with them.
They're just being used to find similarity.
And then what we'll do is iterate through the results and then print them out.
OK, so now let's actually run a query.
So which city has the highest population in the world?
All right, we're great.
So we got back a bunch of similar questions with the most beautiful city in the world,
which country has the fastest growing population.
All of that is relevant and works great.
We can come up with other sort of silly examples to see that our system works well
and it's extensible.
How do I make a chocolate cake?
Just sort of a random question that came up with an image is called running query again.
All right, how do I make a delicious cake?
How do you make a 10 inch cake?
These are all relevant questions as well.
So that's it for this lesson.
We built a semantic search system top the bottom using Panko and we embedded the titles
from the questions from Kora.
And we built something that will be extensible and that you can use over and over again.
In the next lesson, we will build a rag system or a retrieval augmented generation
using Panko and OpenAI.

## 03. Retrieval Augmented Generation (RAG)
In this lesson, you will build a RAG system or a retrieval augmented generation system
using Pinecon.
You will work with the sample dataset of Wikipedia articles and build vector embeddings
out of the articles.
Then you will look at what search results look like by doing a simple document retrieval
from Pinecon.
Then while working with OpenAI, you will build a nicely summarized article out of these
results.
Alright, let's dive into the code.
Now we're going to build a classic retrieval augmented generation or RAG system using Pinecon
with OpenAI.
It's going to look something like this.
We have our happy face user that we had in previous lessons here on the left.
What he's going to do is ask a question against the dataset that we will have prepared and stored
in Pinecon.
In this case, we'll ask what was the Berlin Wall.
Similar to other lessons, we got back from our responses or documents from Pinecon.
To be really nice if we could take some of those long responses and get a summarized version
of them.
What we'll do is show how to build and do a little bit of prompt engineering out of the
responses from Pinecon, send those back into the prompt of OpenAI and get back a beautifully
written response from OpenAI.
This is really just RAG in a nutshell.
I definitely encourage you to try different versions of doing prompt engineering or it's
just a fairly simple example to show you what it might look like.
Let's get started.
What we're going to do is first clear out warnings.
You've seen this previously before where we import warnings and filter out the warnings.
This is just to keep things clean for us.
Then we're going to pull in the packages that we rely on.
Again, we have our deep learning AI tools package that manages keys from OpenAI and Pinecon
for us.
Then we're going to set up Pinecon similar to how we have before.
We're going to get the API key.
Now that we have our Pinecon API key, we're going to, as usual, connect to Pinecon.
First connect to Pinecon, then get the index name using our Utils package and then delete
our index if it already exists and then create the index and then finally get a pointer
to the index that we just created.
Let's look at our index name just for fun.
Something long and unintelligible, but that's okay.
It works great nonetheless.
Now we're going to download our data and we've prepared that ahead of time so that you
don't have to watch us download data, like a cooking show in that sense.
It's just that you get a nice key for quiet in an output file would be called lesson2wiki.csv.zip.
It's a file sitting on Dropbox somewhere with a long, long URL.
Then after it gets downloaded, we unzip that file and you'll have a CSV file locally in
your notebook.
Once we do that, then we're going to create a data frame out of it.
This is a data frame using DFS, short hand for data frame, and then we're using pandas
to read the CSV file.
That reads underscore CSV and we're reading a file that we just unzipped.
Let's just take a quick look at the data that we have.
We use data frame.head to look at the data.
There's a few columns of interest.
One is the unique identifier for the data.
There's the metadata that has the source of the articles and the content.
Then there's the vector embedding themselves under values.
These are good examples of what a vector embedding looks like.
It's really just a list of floating point numbers.
Now we'll prepare the embeddings and upload them into pine cones.
We'll create an array called prepped.
That's going to save what we store into pine cone.
Then we will iterate through the data frame.
We'll iterate through the data frame.
We'll use 2QDM again to get a progress bar.
Then we will get the metadata from it.
That's a string we're going to use literally val from the AST package to get a dictionary
back.
Then we're going to append it to what we just created.
This is how you create a vector embedding in pine cone.
This is this long dictionary of values that I cut and paste here because it's a long
value I'll explain it to you.
What we have here is really a tuple again.
You have an ID that's unique identifier for the embedding.
You have the values which is really the vector embedding itself.
Then you have the metadata.
We just took the metadata that we had from the line above.
This is the article information.
Now what we're going to do is upload that data into pine cone.
What we want to do is do it in batches.
If the length of that prepared array is greater than 200, we're going to do it in batches
with 200, then we're going to insert it into pine cone.
Once it's inserted, we're going to clear that out so that we can do it all over again.
Again, a pine cone embedding is really three things.
There's the ID that's unique identifier for it.
There's the values that's the vector embedding.
Then there's metadata that goes with the vector embedding.
In this case, it's the article metadata.
Now what we're going to do is upload data in batches of 200 vectors at a time in the pine
cone.
We'll check to see if we've stored 200 elements into this prep array.
If so, then we insert the data and then we clear it out again.
We'll run that.
We'll see TQGM do its thing while we upload data into pine cone.
We finished uploading our data into pine cone and let's verify what was uploaded.
We will use index.describe index.stats and look at it.
We have 10,000 elements just as we had hoped.
Now we're going to use OpenAI.
What we're going to do is get our OpenAI API key.
That is utilizing this package that we created that we keep mentioning called DLAIUTIL.
We're going to get our API key and we're going to get a pointer to the OpenAI client.
I have a small helper routine that I've defined to help you get back a vector embedding
from OpenAI.
This is called get embeddings and what this does is it's basically taking array of text
and it's going to use the ADA embedding model and return to you vector embeddings for
the text in the array of articles and it just returns it sort of blindly.
By the way, you'll notice that we had that batch of 200 that I talked about earlier
before.
Feel free to try and play with that number and see what you can get.
I found that for me personally, it worked great at 200.
Pine cone recommends that you do something between 100 and 500 but I personally found
that 200 was the fastest for me but I'd love to hear from all of you out there taking
the lesson what was fastest for you.
Now we're all set up.
We have our data sitting in pine cone.
We've connected to OpenAI.
We've prepared OpenAI and we're ready to execute some queries.
Let's talk about the Berlin Wall for just a moment and we'll ask what is the Berlin Wall?
We will build an embedding out of that query and then we will run a query against pine
cone.
This is how we query a pine cone again.
It takes a pointer to the embedding that we just got above.
We're going to return the top three results again.
Feel free to play with that.
Change it to 5 or 10 or whatever you want to look at.
In this case, I only want to see three for demo purposes.
Then we want the article data itself and that's going to be sitting in the metadata section
and we want to have that sitting set the choose that we get the metadata back.
What we'll do is parse the response out.
We get back an object from pine cone and we're going to extract out from the matches object
the text inside the metadata.
That's what this comprehension does.
What we're going to do is look at it by joining it to the new line character.
Now we're done with this setup and we're going to query pine cone and see the results
come back.
Here we go.
We've got three articles.
We asked for three because we set top K to three that have information about the Berlin
Wall.
This is a lot of text and it's disconnected because we have three discrete articles but
it'd be really nice to get that summarized into something that's readable for folks in
case we want to use it for our homework assignment or preparing a memo for work.
That's definitely what I would do and that's what we're going to do next.
We're going to prepare through what's called prompt engineering.
We're going to build a small prompt out of those articles that we just saw in pine cone
and it's going to take the following format.
We're going to say write an article, title, colon, what is the Berlin Wall?
We're basically instructing the OpenAI to write an article for us based on the Berlin Wall.
What we're going to do is again get our embedding, using the get embedding helper function
that we just asked for.
We'll run this all over again.
We just did this but that's okay.
It's fast.
We're going to run the query and now we're actually going to build our prompt and this
is going to be fairly sophisticated compared to just asking it a simple question.
What we're going to do is iterate through the matches object that we just got back from
pine cone and get back that text and that's what we're going to store in the context
array and then we're going to have a prompt start in a prompt end and prompt start is going
to say answer the question based on the context below note that that has a slash in slash
in at the end.
That's fairly important and then context and what's going to be inside the context is
those three articles that we just got back but before we do that we're going to define
the end and then what we'll do is join the start to our context to our end and our end
is basically the question right that we have above here and then we want to ask opening
how to give us back in answer and so what we're going to do is define a a tuple called
prompt and prompt is going to have prompt start plus some key new line characters and dashes
that's just helping open AI recognize that we're doing some prompt engineering here and
then we have prompt end that we just declared above so we have prompt start and so the question
based on context below we have context which is the results we got back from pine cone
and then we have the end which is question colon you know was the question we're asking
in this case it's what's the Berlin wall and then we're asking it for an answer and
that would be stored in the tuple named prompt and let's take a look at what we have
all right answer the question based on context below the context which is the summary of each
of our responses from pine cone and then write an article titled what is the Berlin wall answer all
right now we're ready to send that to open AI and so what we're going to do is talk to open AI
in a very typical fashion right we're going to use the completions API we're going to use GPT 3.5
turbo and we're going to ask for 1500 tokens max and again this is you know all of this is
things that you can tune and play with yourself feel free to change max tokens or temperature
and then we're going to do one of my favorite things which is this little toy function
to print out a dilemma to keep everything clean and then we're going to print out a response from
open AI all right and there we have it we have a nicely written article about the Berlin
wall right the Berlin wall known as iron curtain was a physical and ideological barrier okay that's
not so great but the articles well written and it's using the information that we got back from
pine cone so in this lesson we looked at data that came back from pine cone we got three discrete
documents back from it we took that built a a prompt engineered query into open AI and got
back a nicely summarized response from open AI that's well written and probably usable for your
homework assignment even in the next lesson we're going to build a basic recommendation system

## 04. Recommender Systems
In this lesson, you will build upon what you have learned in previous lessons.
You will work with the sample dataset of news articles and build vector embeddings out
of the article titles.
You will then build a recommender system that will search across all article titles and
retrieve the titles that are most relevant.
You will then build upon that even further by building a recommender system based on the
article content rather than the topic.
All right, let's go.
So we're going to build a simple recommender system and it's going to look like this.
And we're going to do it in two parts.
First we're going to take a series of news articles and build vector embeddings out of them
and put them into pine cone.
And then we'll ask it about Obama, a very simple query, and we'll get back some results
with three different articles.
And this will be a title-based search.
In other words, we're going to look for Obama and the title of the article.
Then we're going to check to see if we can do it slightly differently.
And so what we'll do is we'll take the data that we uploaded in the pine cone, we'll
query Obama, but this time we'll do it against the article content itself rather than the
title.
So let's get started.
So what we'll do, we'll start off by cutting off the warnings as we have in other
lessons and then importing the packages that we depend on.
Note we have our deep learning AI utils package once again.
So let's go ahead and run that and run our imports.
Okay, great.
And then we will set up pine cone as we have in other lessons.
So we'll get a pointer to our deep learning AI utils object, get our pine cone and open
AI API keys and then start to prepare the data.
So once again, the data is not insignificant in size.
We have a file called all the news-3.zip that is being downloaded here using WGET.
And just again, as a cooking show, we won't watch you do that or I won't force you to
watch me download a series of data and I'll comment that out.
And then the same thing to unzip the data as well.
So now let's look at the data.
So this is rather straightforward.
We'll say with open and then the name of the file that we want to look at and we want
to read it.
Okay.
And then we will get the first line and as in any CSV file, the first line is the headers
of the file.
And so we'll just look at the header.
Okay.
So we have date year, month, day, author, it's sort of, we're interested in the title
who we're interested in article.
So to make it a little bit more convenient to work with, let's use a data frame.
So we'll say data frame equals the pandas package and we're going to read the CSV, call
all the news dashedv.csv.
And let's just read 99 rows, okay.
And then look at the data.
So we'll call head, all right.
So we have the data.
You can see the date, the year, the month, the day, et cetera.
Again, we're interested in title and article.
So let's prepare a pine cone in the same way that we have previously.
But to do that, first we will get a pointer to open AI.
We'll use that later on and we'll prepare our utils package.
Okay.
And then get our index name from the utils package.
And then get a pointer to our pine cone object.
And then this is boilerplate at this point.
So we'll just copy and paste this in.
But again, we check to see if the index name already exists.
If it does, we delete it and then we create it because we just deleted it.
And then finally, we'll get a pointer to our pine cone index.
And then run that.
You saw this function in an earlier lesson, but we have something called get embeddings.
And this is just a wrapper function for open AI to return embeddings from a list of
text.
In this case, that text array is called articles.
And so we'll declare that it's called get embeddings.
And then what we're going to do is prepare and insert the data.
So we'll iterate through the data frame, we'll read it, number of rows at a time, build
an embedding out of it, and then upload it into pine cone.
So this is a lot of code.
It's about, say, 12 to 15 lines of code.
So I'll copy and paste it in, but walk you through it.
So we're going to read the CSV file called all the news dash three.
This is the one that we've been working with earlier.
But we're going to read it, chunk size rows at a time to make it manageable.
OK?
Let me just make this a little bit easier to read by putting it there.
And we're going to read at most 20,000 rows.
OK?
Then what we're going to do is iterate through the chunks one at a time.
And remember, the chunk size is 400.
So what we're going to do is get the titles out of that data frame chunk, right?
The chunk is just a subset of the larger data frame.
OK?
Then we're going to get the embeddings for all of the titles that was declared here.
And to prepare a pine cone embedding, we're in array of embeddings as we have before.
And this is just simple Python syntax, right?
For I and range through the number of titles, create an embedding for each one of them.
All right?
Then we're going to increment chunk name because that's what helps us get a unique identifier
for the embedding.
And then we're going to upload it 300 at a time.
So if we get to 300 or XC300 number of vector embeddings inside the prepped array, then
we're just going to absurd it into pine cone and then to the or clear out that prepped list
and then update the progress bar for TQDM, which we declared up here.
So let's go ahead and run that.
This is going to take around three minutes on our machine, as you can see here, but we'll
speed this up in post for you so that you don't have to sit here for three minutes and
watch it.
OK?
So that has completed.
We absurded 20,000 vector embeddings in the pine cone and now we will look at what we
uploaded.
So let's say index dot describe index stats.
And yes, we had 20,000.
So that's perfect.
So let's declare our function again that gets embeddings from open AI.
We've seen that function before.
And let's actually get some recommendations.
So I will go ahead and paste in this helper function that we've created here called
get recommendations.
And what it does is takes a pointer to pine cone, what you want to search for, the number
of elements that you want to return inside of top K in this case will default to 10,
gets an embedding out of the search term that was passed in from that helper function
declared above, executes and returns the results from a pine cone query.
And what we'll do here is the core data about Obama.
OK?
So we declared that function.
And so let's say recommendation equals get recommendations are pine cone index and something
about Obama and for each result in the recommendations, which is returned inside of the matches element.
We'll say print the score, which is the similarity score coming back from from pine cone.
And the metadata that we stored inside of pine cone next to that embedding.
OK?
All right, so we've got back a bunch of information based on the titles of the articles about
Obama.
Obama has been a much more effective communicator than he gives himself credit for.
That sounds pretty interesting.
You know, parting message is warning for Donald Trump.
That sounds like something you want to read as well.
And so what we're going to do now is see if we can do it differently.
So you saw, we just looked at, we searched for titles as the vector embedding.
Now we're going to look for elements that are vector embeddings where they're representing
the actual article itself.
And so what we'll do before that is delete our index first.
So you saw this piece of code.
You're familiar with it at this point, right?
If the index is there, we're going to delete it, otherwise create it, nothing different
there.
And so what we're going to do is somewhat involved, and I'm going to walk you through it.
So let's first declare a function called embed that takes an array of embeddings, a title,
an array that's going to store results into, and then a counter called embed number.
And I'll walk you through how this works in just a moment.
But before I do that, let me paste in the code that actually uses it.
So what we're going to do here, I'll jump down here to the bottom, we're going to iterate
through all of the articles list from our data frame, let's declare here, then we'll
grab an article and we'll grab the title that goes with the article.
If the article is not none or null, what we're going to do is use the recursive character
text splitter from chain, and we're going to use that to effectively chunk up the article
of interest into 400 characters at a time, and an overlap of 20.
And then what we're going to do is call this embed function that we declared up here.
So let's store that.
So as we split it up into smaller chunks, we're going to get the embeddings for each one
of those chunks by calling get embeddings, which is declared above, and then we're going
to call embed.
OK, and so what embed does is it walks through this array and stores the vector embedding
for each one of the chunks into the prep element.
And if it hits, and you're used to this pattern at this point, if it hits 300 or more elements
inside of the peptor array, then we're going to go ahead and upstart that data into
pine cone and then clear it out so we can do it all over again.
OK, so we ran this, I believe, and let's go ahead and run this code.
So this is going to run and take quite a bit of time, it's going to take around five
minutes or so, and we will fast forward that for you, so you don't have to watch upload,
and we'll be back in just a moment.
All right, so that finished uploading, and now we're going to make sure that after we
previously cleared out the index that our upload worked.
So all right, 10,500 vectors were uploaded, that's great.
So what we'll do is run the same exact code almost that we already ran, except the main
differences that we're going to keep the small dictionary called scene, and the reason
we want that we're going to do that is because if you recall, we actually split up the
article in the chunk-seizing length chain, and what we want to do here is keep track of
what we've already seen so that we don't get the same article being returned twice.
So we're going to say, if title, not in scene, then we will actually print out the score
of the search as well as what we got back from our search, and then we'll say, like,
hey, we've already seen this article.
It's very simple.
There's a bit of dot in there to keep track.
And what we get back is, all right, and we go back to different set of articles.
This was based on the actual content inside the article, not the title of the article this
time, and you'll actually see the results are different this time, which is great.
All right, that's it for this lesson.
In the next lesson, we're going to build out something called hybrid search.

## 05. Hybrid Search
This lesson is all about hybrid search.
You will leverage a feature in pine cone that allows indexed entries to have both dense
and sparse embeddings at the same time.
Of course, this means we can also search across sparse and dense at the same time as
well.
To do this, we will leverage BM25 and Clip to generate embeddings from fashion products
to search across both text and image product descriptions at the same time.
Let's get coding.
We're going to build a hybrid search system using pine cone.
So, pine cone supports vectors with sparse and dense values, which allows you to perform
hybrid search.
In this lesson, we're going to teach you how to use hybrid search, which is the ability
to combine vector semantic search with traditional keyword text search against different modalities
for the same entity, and this example will show you exactly what that means.
So we have our user here on the left-hand side, and we're going to take a data from a fashion
product database.
And we're going to encode it into two different types of vectors.
The vectors that we've been using previously are what's called dense vectors, and to do
that, we'll use something called Clip, and we'll get into what that is later on.
But this time, we're going to generate what's called sparse vectors, and to create that,
we'll use BM25, which is a common encoding technique in information retrieval, but we'll
talk about what that is later.
The key point here is that for any given row in the index of pine cone, you can store dense
and sparse vectors at the same time with the metadata that goes with it into one row.
Then what we're going to do is query for dark blue French connection genes, and we'll
get back a data set from pine cone, which we'll have some genes, obviously.
And then we'll tune what's called the alpha parameter inside of that query to change whether
we weigh the results from the BM25 encoding or the Clip encoding more or less, and we'll
see how the results change.
So let's get started.
Let's get going.
We will go ahead and import in our warnings, cut off the mechanism here, so import warnings
and filter warnings.
And then we will pull in the packages that we've been depending on previously, nothing really
different here except for the BM25 encoder from pine cone underscore text dot sparse.
Okay?
So we'll go ahead and run that.
Then we're going to continue doing what we've been doing previously.
We'll use the deep learning AI utils package to get our pine cone API key, okay?
And then we're going to do a little bit of setup.
So we've seen this as well.
If you have kudo on your machine, that's great.
If not, no big deal.
We'll just use your CPU.
And then we'll get a pointer again to our package.
We've already done it, but why not do it one more time?
And this time create our pine cone index.
Well, at least get the name of the index here, okay?
And then what we'll do is create an instance of a pine cone object.
And then do what we've done historically, but with a little bit of a of a change this
time.
So if the index name is in the list of indexes that pine cone has, then we'll delete it
just to keep things clean, but this time what we're going to do differently is change
the metric that's used to calculate similarity inside of pine cone.
We're going to have it be dot product instead of cosine, like it has been previously.
And then we're going to get a pointer to our pine cone object, and we'll go ahead and
run that.
Everything's great.
We have a CPU here because we're inside of Jupiter on a Mac, no GPU for us to use here.
So let's set up our data.
First we're going to set up our fashion data set from hugging face.
And that comes from here, Ashrock, has fashion product images small.
And we're going to use the training split, okay?
And then I'll put what we have, okay, all right, there's our data set has ID, gender,
master category.
Again, we've already downloaded it for you.
So that's why it came back so quickly.
So let's look at the data.
We're going to go ahead and look at some of the images inside of the data set.
So images equals fashion image, and then we will do a little bit of column removal, just
to keep things simple.
And then why not look at one of the pieces of data in the data set.
All right, so that is the 900th element in that data set.
And now we're going to convert into pandas data format, just a bit, and we'll get to
the metadata, and data dot to pandas, and then look at that data using head.
All right, so you have ID, you have gender, category, subcategory, article type, pretty
extensive fashion data set, this is going to be fascinating to work with.
So now what we're going to do is create our sparse vectors.
So for creating and training sparse vectors, we're going to use the BM25 function from
the Panko and Text library.
BM25 is a popular technique for retrieving text.
It uses term frequencies to determine the relative importance of the term to the query.
It's simple but effective, and only requires knowing the number of documents in the data
corpus and the frequency of terms across documents.
We're going to show you how to use BM25 with Panko and sparse dense vectors for use in
hybrid search.
Clip on the other hand is a neural network devised and created by OpenAI built on millions
of images and respective descriptions that can return the best caption for an image,
right?
So given an image, we're going to get back a caption.
So let's get started with that.
So we're going to create a BM25 encoder, and we're going to fit our data to the metadata
that we just looked at above.
OK, and then we'll look at the product display name, turtle check, min, navy, blue shirt.
All right, that's the product display name, it sounds.
And now what we're going to do is do two different things.
First encode queries of a product display name, and look at what we get back.
All right, so we've encoded queries, we've encoded documents, and now we're ready to
get going because we've completed our work with BM25, and we're going to cut back over
to dense vectors next.
So we're going to come back to using sentence transformer as we have in previous lessons,
sentence transformer, and I'll put what the model looks like, and then do some dense
vector encoding, so see what we get here.
All right, so 512 dimensions in dense vector, which is exactly what we wanted to see.
So now let's go back to our fashion data set, OK, we're going to see how many elements
that we have.
And so this is going to be the first time inside a pine cone that you see us uploading
sparse and dense vectors at the same time.
The way that you do that's actually not too different from how you upload normal vectors,
it's just that you're going to add sparse and dense at the same time as you can imagine.
So a bit of code to do that, I'll walk you through a webinar, go ahead, and paste that
in for you so you can see it.
And so we're going to do it in batches of 200, like we have before, for each one of the
elements in the batch, in the fashion data set, in batches of 200, where you get pointers
to the beginning end of the batch that we're going to use, extract the metadata, convert
the metadata, and then build metadata fields that are ready for upload.
We're going to take the batches of the images, we're going to encode our sparse vectors
using the encode documents function, we're going to get our dense vectors using the model
encode function, convert that back into a list, and then get our unique identifier to each
one of the vectors.
So if you remember, each pine cone vector embedding is a combination of an ID, the vector is
that it represents, and the metadata that goes with it.
So this is the metadata up here, this is the vector embedding here, and this is the
unique identifier for the embedding.
The only difference here is that we have dense and sparse at the same time, and you're
going to see that we can upload them down here in the append function below.
So what we're going to do is zip up all of that data that we just created above, and
we're going to append it inside of a dictionary object here, and upsurge that append.
Okay, we're going to keep on adding these three as we iterate through the for loop, and
then do an upsur in to pine cone.
And then we'll see what we got back after uploading all of that data.
So let's go ahead and run that.
All right, so that upsurge in the pine cone has completed, and now we're going to get
into the essence of actually doing hybrid search.
So let's go down a bit and start to query.
So we'll say, we're going to look for dark blue French connection genes for men, okay.
But to do that, we're going to create sparse and dense encodings out of that query string,
and then we're going to run our pine cone query, okay.
So in the same way that we did upsurge slightly differently, we're going to do query slightly
differently by specifying dense and sparse vectors at the same time, and you can see that
here in these two lines, and we're going to return that most 14 items.
And we're going to take the result of that query and look at it.
And the result of that query is going to be the images inside the data set, okay.
We're going to look at the IDs, right.
So let's go ahead and run that.
All right, so we get back basically pointers to images, but we'd like to view that result.
So we've created a little bit of a helper function so that you can look at these images here.
We don't want to see the, see the pill images, we actually want to view the image itself proper.
So let's go ahead and put in this, this small helper function.
There's a little bit, neither here nor there in terms of what it does,
but it basically creates a HTML so that you can view the image and, and
pulls it out of, of the metadata.
So let's go ahead and run that.
And we'll actually just call this display result with the result from our query.
Oh, there we go, perfect.
BlueJane's exactly as we had hoped, okay.
But now let's balance the search a bit, right.
We talked about this alpha parameter before.
Let's declare another helper function to get to that, right.
So it's possible to prioritize our search based on sparse and
dense vector results and to do that.
To do so, we scale the vectors.
And so for this we're going to create a function called hybrid scale.
So let's go ahead and just paste that in because it's, it's pretty large, right.
So what this does is take in a parameter called alpha.
And what that's going to do is control whether we rely on sparse or dense,
dense more.
I encourage you to look at this code when you're offline at home and
look at it, what you want to do is know that controlling alpha relies on sparse or
dense more.
Let's go ahead and run that, okay.
And run the query one more time, but with different amount of alpha.
So what we'll say is that's the question.
Okay, I'm going to put in dense and sparse and then we'll run our query one more time.
And then view the images.
Before we're in the code, we're going to set alpha to one.
And we're going to run that code and we get back our images.
It's perfect.
And what we're going to do now is check to see whether we got back more men's genes.
So what we'll do is go ahead and copy and paste this in and execute that query.
And yep, we got exactly back what we had opened.
Okay, so let's actually execute the hybrid scale function here and look at the results
that we got.
Now remember our alpha parameters, it's actually a continuous variable on the range of zero
to one.
And I challenge you or encourage you actually to play with what different results you get
from changing it.
It's, you know, you could change it from zero point two, you can make it zero point five.
It's really up to you.
But for illustrative purposes, we're going to make it zero and one just to make the example
fairly obvious.
So let's go ahead and run this and make it more sparse this time.
So we'll run that, you know, seeing our results, we got back women's genes, which is perhaps
not exactly what we want.
We're looking for men's genes up here.
And let's confirm that by looking at the titles, yeah, we got back women's genes.
And just again, for illustrative purposes, let's set the alpha parameter way on the other
end of scale towards more dense and make it one and look at how that changes it.
Yeah, we got back all men's genes, which is a little bit better.
And just to confirm, yep, it's men's genes indeed.
So that's perfect.
So that's hybrid search, the ability to merge sparse and dense vectors into the same row
inside of pine cone.
And you saw how we did it with fashion data.
In the next lesson, we're going to run through facial similarity search.

## 06. Facial Similarity Search
In this lesson, you will have fun with the age-old question of to which parent the child
looks the most similar.
To do this, you'll answer the question by using vector embeddings.
You'll use a freely available image dataset, in particular one of the British Royal family
in your science to determine if Prince William looks more like King Charles or Princess Diana.
It's simple and extensible enough for you to try on your own family pictures.
Let's have some fun.
We're going to build a facial similarity search system to compute which parent looks
most like their child.
So we'll take a father, we'll take a mother, and we'll take one child and see whether
the child looks more like their father or more like their mother.
The person that looks the closest will be based upon the highest average score between
each parent and child.
We're going to use the deep-face open-source library with the face net model inside of
it.
That's 128 dimensions wide, and the dataset that we're going to use came from the families
in the wild dataset, which I'll have a link to in the lesson here.
So the way that it's going to work is the user is going to run images of King Charles,
Princess Diana, and Prince William through the deep-face model that we're going to use,
and we're going to store those vector embeddings in bank home.
The way that we're going to calculate similarity is we're going to run images of King Charles
and Princess Diana and Prince William through the deep-face model that we're going to use
and store those values in bank home.
Then we'll calculate similarity scores for the mother and the child and the father
and the child and see based on an average who looks more like each other.
Okay, let's get started.
So to begin with, we'll do our standard of importing our warnings to stop the warnings
from showing up, as well as a slightly longer list of imports into the notebook.
And then we will get our pine cone API key.
Now I've prepared a dataset for you ahead of time, and as usual, I will not make you sit
there and watch me download the data, but similar parameters to WGET make it quiet, show
the progress.
I'll put the file name as family underscore photos.zip and pull it off of the location that
I've specified here, and you run that, and then we will unzip the photos as well.
Okay, perfect.
I'm sure that you run the query like this, it'd be uncommented, right?
But for this lesson, we'll keep it common to doubt so that I don't accidentally run
that for you.
The data has been downloaded for you in your notebook.
This is how I ran it when I was at home before we prepared it, but you already have it
local on your disk.
In the family photos, zip file has a structure of family as a top-level directory with
sub-directories of dad, mom, and child each.
So let's actually look at some of the pictures.
I've created a small helper function to help you see the photos, called show underscore
image.
And we're going to resize the photos slightly here and actually show the images locally.
So we're going to look at one random photo I thought that was particularly interesting.
Family slash dad slash p060 to 60 underscore face 5.jpg and see what we get.
So that's an image of King Charles, let's do the same thing for Princess Diana, right?
The same thing for Prince William, all right.
So those are the folks that we're going to run the queries for.
And now we're going to set up pine cone.
So same as before, we're going to create our index name using the deep learning AI Utilities
package, get a pointer to our pine cone instance and run that.
And now we're going to create our vector embeddings inside of pine cone.
I'm going to go ahead and paste this in for you, since it's a bit of code, but I'll walk
you through it.
So we're going to actually store a file called vectors dot back and we're going to use
that and keep it around because we're going to iterate through the data multiple times.
And we're going to suppress warnings in case, you know, the file doesn't exist and we're
going to remove the file just to keep things clean.
So what we're going to do is open up the file.
We're going to glob all of the images of each of the people separately.
And then we're going to create embeddings out of each person's face and write it to
disk.
And then with that, we're going to write it to disk inside of that vectors dot veck file
is person, colon the image name, and then the embedding value itself.
And then we'll go ahead and call generate vectors and store that.
All right, that happened pretty quickly, thankfully.
And now let's actually look at that file and see what it looks like.
So we'll just use head to look at the data and you can see it has the file format that
we talked about earlier.
The person's name, in this case, it's the mom, one of the images, P11, et cetera, it's
sort of a .jpeg.
And then the vector embedding for that specific image, okay, so we'll scroll down since we
have five lines of that file, okay.
And now we're ready to calculate some scores and upload data into Panko.
Before we upload the data into Panko, we're going to produce a tSneak scatterplot based
on PCA reduction that's going to give us a really pretty scatterplot of the data so
we can visualize it.
Principle component analysis, or PCA, reduces the dimensionality of data, in this case embeddings
and is often used to reduce the dimensionality of large data sets by transforming a large
set of variables into a smaller one that still contains most of the information in the
large set.
A tSneak plot, where t distributes stochastic neighbor embedding plot, there's a type
of data visualization used primarily to represent high dimensional data in a two or three
dimensional space.
This technique is particularly useful for understanding complex data sets by revealing
patterns, clusters, and relationships between data points that might not be apparent in
higher dimensions.
PCA is often used as a preliminary step to reduce the dimensionality of the data.
This can be particularly important when working with very high dimensional data as it helps
to mitigate the curse of dimensionality before applying tSneak.
TSneak can be computationally expensive and slow, especially with large data sets.
And so by using PCA to reduce the data to a lower dimensional space first, you get a
significant speed up in the tSneak process.
For the tSne algorithm, perplexity is a very important hyperparameter that controls the
number of neighbors that each point considers during dimensionality reduction, and you'll
see that actually made it pluggable in the code below.
So let's go ahead and do that.
So I've created one helper function called Gen, tSneak, data frame, or generate tSneak
data frame.
And what that does is it iterates through the vectors that VEC file, looking for a specific
person, and depending on that to the vectors list.
And then what we do is actually do a PCA reduction against a specific person, right?
So what we're doing here is a PCA reduction for a specific person who's passed into the
function.
Then we do a tSneak reduction as well against that, and return that as a data frame to the
next function, as we're going to call here called plot tSneak, and I'm going to go ahead
and plug that in.
So plot tSneak actually calls generate tSneak data frame, and you can see it down here
below.
This code is just setting up that plot lid.
And what we're going to do here, and this is the meat of it, is for each person, right,
that we define here in this dictionary, dad, child, and mom, these are just the colors
for the budget.
Don't worry about that too much.
We're going to call it generate tSneak data frame, and get back a data frame, and plot
it against our scatter plot, and then actually show the plot.
So let's go ahead and do that.
So make sure that we ran that.
Make sure we run that.
And then let's actually call it.
So there's one last parameter, sorry, one last function call of plot tSneak, and this
is the perplexity.
In this case, we're going to set it to 27.
Yeah, so it did exactly what I predicted that it would do.
So if you guys have ever done this before, you have to run it a few times.
So we're going to call plot tSneak with the perplexity of 44.
Definitely, I encourage you to play with that at home to see how it changes the scatter
plot.
I'm sure that the images of each of the dad, the child, and the mom do clusters together
well.
I would definitely discourage you from insinuating too much that the child and the mom are close
together in the scatter plot is that it doesn't actually show much.
Really what you want to do is look at the fact that the images are clustering within each
person on their own.
So in other words, each one of these dots represents one image of a dad, and they're all very
closely clustered together, which is great because each of the pictures of the dad should
look like the dad, same thing for the child, and same thing for the mom.
Okay.
Now that we visualize the data, let's set up pine cones so that we can store these values.
So what we're going to do is, same thing that we've always done, delete the index if it's
already there and then create the index again, so we have a clean slate.
So we'll go ahead and run that.
Now we're going to store the vectors, and all we're going to do is run through that file
that we created again one more time.
So open up the file, iterate through one line at a time, split it into the person in the
file name, eval, the vector that was actually stored in that file, and then absurd it into
pine cone.
So let's go ahead and do that.
Okay.
Let's see what we stored inside of pine cone.
Let's do it a describe index stats.
Okay.
We have 241 elements, and that looks correct.
We had 241 iterations from TQDM before.
Now let's compute the scores, and I'm going to paste this in because it's fairly large,
but we have a small routine called test, and what this is going to do is return the top
K similar images against the child.
So it's going to take in one of the parents, this is either going to be mom or dad each
time around, and it's going to return the top K, and what we're going to do is actually
calculate the average score from the return value from the query from pine cone.
So for each vector inside pine cone, we're going to get a score, and we're going to take
the average of the top K values that we got back.
So let's store that value, and then we're going to call it for each relationship that
we want to look at.
So we're going to call compute scores, and so what we're going to do is call test for
the dad in the child, and we're going to call test for the mom in the child.
And to do that, we're going to take a random vector, or random image, I should say, from
the dad in the mom, and use that to compute the score.
So let's go ahead and do that.
All right.
So the dad got an average score of 0.43, closeness to Prince William, who's the child,
and the mom got an average score of 0.35, closeness to the child that is Prince William.
And so what that means is that the dad is closer and similar to Prince William.
Now what we can do is use the fact that we know the dad is closer, and find which image
is the closest between them.
So what we're going to do is take one base image of the child, which will choose this
because it's probably the most representative of what he looks like, in my opinion.
This is what he looks like as a mature adult.
And what we're going to do is find a way to find which image in our data set looks most
like him from the dad.
So we're going to take that image, we're going to build an embedding out of it, and we'll
look at it.
That's the embedding version of that image above, and then we'll scroll down, and then
we'll query, we'll take the top three results that match it from the dad.
And then let's look at the response.
This is a typical JSON blob that comes back from Panko, and you get the ID of the vector,
you get the metadata, and you get the score.
Now what we're going to do is look at the image in particular that matches.
So this is the image that is most similar to Prince William.
So in this example, we built a facial similarity system that matches King Charles and Princess
Diana to their offspring Prince William to figure out which of those two looks most
like Prince William.
The data show that's actually King Charles that looks the most like Prince William.
And we, at the end, we figured out which image in particular looks most like Prince William,
and we saw that this image here at the end looks most like Prince William, which I would
agree.
They very much look similar.
I encourage you to play with this code with your own family members and see what results
you get as well.
In the next lesson, we're going to build an anomaly detection system again using Cisco
ASC log files in Panko.

## 07. Anomaly Detection
In this lesson, you will build another fun system.
This time, you will build a small machine learning model to detect anomalous log entries in Cisco ASA log files.
You will train the model using supervised learning against the small data set we prepared in order to make the lesson more accessible and keep training times down.
You will then use the model and feed it a sample input data set to find anomalous log entries.
Let's code and innovate.
We're going to build an anomaly detection system using Cisco ASA log files.
The way that's going to work is I've prepared a very small training set called training.text.
It's a label data so we can do a little bit of supervised learning and create a very small model that can detect anomalies against log files.
Then we're going to use a sample data set called sample.log that I've also prepared that has one small anomaly inside of it.
We're going to try to see if we can find the anomaly.
We're going to run the sample.log file data through our model and store the embeddings as the output inside a pine cone and use that to find the anomaly.
Let's get started.
We'll start by importing our warnings as we always do.
Then our package dependencies.
Then we're going to get our pine cone API key using the deep learning AI tools package here.
Then we're going to create our index.
Same code as always, we get a pointer to the pine cone instance.
Then we'll delete the index and then we'll create the index as well and then get a pointer to it at the end.
That's going to go ahead and run.
We've already downloaded the data for you in the lesson.
If you happen to be running this offline, you'll need to run this by uncommenting out this line.
That pulls down this file called training.tar.zip.
We've already done it ahead of time.
We also want to untar the data.
Again, this has already been done ahead of time.
Let's look at the data and see what's inside the sample log file.
We're going to look at head minus five.
We can see the first five lines.
This is an ASA log file format.
We have the date, the time, log, sort of identifier, and then the actual log message.
What I've done is gone ahead and prepared a data set that has been labeled for you based on that log file above.
There's a slight difference.
The format of this file is essentially a sample log file and then pat or carrot, another log file,
and then the label similarity between the two at the end also diluted by carrot.
We are going to only look at the top five here.
There are some that are labeled differently at the very bottom of the file.
I encourage you to look at the bottom of the file.
What this is going to say is that log part here, highlighted here by my mouse,
is similar by a score of one and so on and so forth.
This is similar by a score of 0.9 on a continuous variable range of 0.1.
Now we're going to create our very, very small model based on sentence transformers.
I've gone ahead and pasted this in because it's a lot of code,
but we're going to create a very simple model.
We're going to use the word embedding model, first here, based on birth, based on case,
which is sort of the base case model.
Then we're going to have a small pooling layer and then a very small dense layer to get it down to 256 dimensions inside of our embedding.
Then we're going to create our model and then we're going to output which device we have.
Again, I encourage you at home, if you happen to be lucky enough to have a coded device, you should try it.
It's a little bit faster, but we're going to be fine on the CPU here.
We have our model object and we've seen that we're on the CPU.
We're going to prepare to train our model.
Here we have an array called train examples that's going to have the data that's going to get stored.
We're going to open up the text file and read the lines.
And then for each one of the lines, we're going to strip off the white space characters and then split it.
If you recall, I said it's separated by a carrot or a hat.
And then we're going to append these samples.
We call it A and B. We had log one.
We had logged two into the similarity score, which I'm calling here label.
We're going to store that inside of the input example object.
We're going to pass it the label because this is supervised learning.
We'll do 100 warm-up steps.
We'll prepare our data loader and then we'll set up our loss function.
Okay, so let's go ahead and run that.
And now we're going to actually train the model.
So we're going to call model.fit.
That's going to do the actual training.
And then we're going to prepare our sample data at the same time.
So it's really two parts, but I've clustered it into one cell.
So there's the model training here, model.fit.
And then there's preparing the sample data that we're going to use to pass through the model later.
So this is going to train and this will run for a little bit.
So this is going to run for a couple minutes and we're done.
So our model is trained.
And let's generate the embeddings from our model.
So we remember we had that samples, the survey that we created earlier.
We're going to go ahead and store that in embed or EMB.
And now we're ready to test our output.
Let's first store our data in Pankoam.
You've seen this pattern before.
We create an area called prepped.
We iterate through all of our embedded data that's stored inside of samples.
We create a vector embedding here.
And we store it inside of prepped.
And we're going to absurd it into the Pankoam.
So let's go ahead and insert in the Pankoam and that was quite fast.
So now what we're going to do is take a known good log line.
So we're going to say good log line equals sample 0.
I happen to know that it's at the top.
And I can say print good log line.
That's a good log line.
I know that that's not a problem.
So now let's take that good log line and query it inside of Pankoam.
And we're going to get the top 100.
And that's going to come back and be stored and queried.
And we're going to get a pointer to the matched values inside of Pankoam and store that in results.
Now what we're going to do is look at the results.
And we're just going to have a small routine here to print that out.
So iterate through, look at the top 10.
And these are the top 10 matches for this line here.
Of course, this one matches and has a score of 1 because it's the same line.
And these ones look very similar as you can see.
So that's the top 10.
Now what we're going to do is take the last element in there.
And print that one out and see how that compares.
So last element in that one is very different.
And that you can see that it doesn't at all look like the other lines above.
This one's called sick fault detected in the matrix in the similarity scores 0.32.
So above, we query the top 100 and there's fewer than 100 here.
So we're going to get less than 100 results.
So what we're going to do is actually look at the last result, which should be the worst performing score.
And expect that that's our anomaly.
And that's exactly what we're going to get from this line here.
We're going to see that the worst performing score is 0.32.
And the log line looks sort of like a Cisco ASA log file except for this message that says sick fault detected in the matrix.
And so this is our anomaly, the worst performing score that came back from Pankoam when we searched for a very canonical looking line stored here.
So that's a very simple way to build an anomaly detection system using Pankoam and the Cisco ASA log file data.
I encourage you to try it with other types of data using sense and transformers as well and see if you can find anomalies in data yourself.

## 08. Conclusion
In this course, you learned about the essentials of building applications using vector databases.
You built simple search and recommendation systems, retrieval augmented generation, or RAG,
explored hybrid search, looked for facial similarities in the Royal family,
and finally found anomalous log entries in Cisco ASA log files.
We only scratched the surface of what's possible with vector databases,
and I encourage you to explore far beyond the material we covered today.
I'm looking forward to seeing what you've built on your own.
