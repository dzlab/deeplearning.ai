# Serverless LLM apps with Amazon Bedrock 


## 1. Introduction
Hi, and welcome to this short course on serverless LOM apps with Amazon bedrock built in partnership with AWS and tied by Mike Chambers.
Thanks, Andrew. It's great to be here.
Let's say you're working to prototype an LOM based service to help a customer service department to sunrise is customer phone calls.
This is actually a real application and a common use case that many companies have been working on.
When building the initial prototype, you might manually download an audio recording of a call that's been recorded with permission, then run automatic speech recognition or ASR software to transcribe it to text file and send that text to an LOM to summarize it and finally slow that summary in the searchable database.
But there's a lot of work to manually connect all these steps together and this is also a repetitive process that you might want to automate so as to have your software be automatically triggered to run whenever a new customer service call comes in.
This is as opposed to constantly rerunning your code by hand each time a new call comes in.
To connect AI services like ASR to your LOM application and also to automate this workflow, you can use event driven triggers that will run the workflow whenever new audio files get uploaded and deploy this entire workflow so that it runs on its own.
That's right, in this course you'll connect an LOM to other AI services in this case automatic speech recognition to enable the LOM to process data that isn't already text such as audio recordings of dialogues between customers and your customer service representatives and you'll deploy this with a serverless architecture as opposed to a traditional server based architecture.
Usually you'd have to do a lot of work beyond building the app to deploy and maintain the app like spinning up machines and installing dependencies.
Then there's even more work with the ongoing maintenance of that system adding security patches updating the dependencies and so on.
By starting off with a serverless architecture you can save yourself a lot of time and effort as you iterate on your app.
You'll use Amazon Bedrock which gives you access to a wide range of foundation models and also the ASR service Amazon Transcribe.
You'll connect these services into a workflow that triggers an LOM via AWS Lambda to automatically run once new files are uploaded to the cloud storage service S3.
The audio summarization use case is just one example but you really use the tools and concepts in this course to integrate your LOM with other AI services into a pipeline that can essentially run on the zone as input data comes in.
Many people have worked to make this course possible. I'd like to thank on the AWS site Aunty above Joe Fontaine, Chandler Martha Brutham and Benjamin Gruer.
And from Volcarium, who helped with the Jupyter notebooks, David Lin, finally from deepland.ai, Eddie shoe and the other entity have also contributed to this course.
In the first lesson, you'll get started with prompting an LLM using Amazon Bedrock and explore how it can be integrated with other AI services.
That sounds great. Let's go on to the next video and get started.


## 2. Your first generations with Amazon Bedrock
In this lesson, you'll use Amazon Bedrock to prompt a model and customize how it generates its
response. You'll also explore how you can integrate and elements with other services in order to
build a data processing pipeline. So let's jump in. And our notebook environment already has access
to the AWS services through the security configuration of this environment. So we can just start
writing code and start integrating with AWS services and Amazon Bedrock. And the first thing that we
need to do when we're working with Python and with Amazon is to import the SDK for Python,
which is Boto3. So I'm going to import Boto3 and run that cell just to load the SDK and all of
that necessary code in order to be able to work with AWS services. So let's go and use the Boto3 SDK
to connect with Amazon Bedrock and make our first generation. So to do that, we call on Boto3 and then we
create a client object. And that client object, we give it the name of the service that we're
looking to connect to. So in this case, it's going to be Bedrock, but it's Bedrock runtime. So I type
in Bedrock runtime. Now there is a Bedrock client as well. And we'll look at that a little bit later
on, but the Bedrock runtime client is the one that we need in order to be able to make generations.
And then we need to tell it the region that we want to use for this particular service. So we do
that by typing in region name and then we're going to give it the name of a region. Now we need to
make sure that we're selecting a region which currently supports Amazon Bedrock. And that is a
subset of all of the AWS regions in the world. So check the documentation to see which region you
want to work with. But for this particular course and with everything that we do in this short
course, we're going to do it in US West 2 and US West 2 is the Oregon region. And that fully
supports Amazon Bedrock. But we just need to put that into a variable. So let's put that into
a Bedrock runtime. So that's going to create our client object. But let's just put that into a
variable. So let's call it Bedrock runtime and load that into there. Now if I run this cell,
then the Bedrock runtime client will be available to us right here. So what can we do? How can we
prompt Amazon Bedrock? Let's create a text prompt and then send that over to Amazon Bedrock.
So I'm just going to paste this in here. This is the prompt I'm going to use. Write a one
sentence summary of Las Vegas. So just defining a text string there, that's going to be the prompt
that we send into a large language model and get it to make the generation off the back of this.
So with that defined, how do we send this then to the Amazon Bedrock runtime endpoint? Well,
we need to create some keyword arguments that we're going to pass into the service. So let's
just create ourselves a little dictionary here and start to build up all of the keyword arguments
that we need to use. First of all, we need to set the model ID. So I can do this here. I'm just
going to paste this in and we're going to choose the Amazon Titan text light V1. And we'll take a
look at these in more detail a little later. But this is the model that we're going to use for now.
This is a large language model from Amazon. Then we need to add in some other details to our keyword
arguments. So we want to put in the content type and the accept. So the content type is the
mind type of the data that we're going to send in. So the input in our request. So we're going to
send a JSON structure. And that is what the model is going to be expecting. And then we can specify
the mind type of the data that we're willing to receive back. Now by default, this will be
application JSON as well. In fact, they'll both default to application JSON. But I'm just going
to say here that we're willing to accept anything that the model gives us back. And that makes this
code pretty portable. So you could take this and work with it with other models if you wanted to.
Now the next thing that we need to do is we need to pass in a body or a body structure. And this
body is going to include the prompt that we want to send. Now this is actually a JSON string that
we need to put into here. And because we're working with Python and I don't want to write a JSON
string by hand, I'm going to use the JSON library to do this. And so I'm just going to hop back up
to the top here and I'm going to also import JSON, run that cell again. And now I can use JSON inside
of my code. So I'm going to say JSON and then dump that string. And we're going to put into here
the contents that we want for our body. And so we can still sort of work in our dictionary format
here. So let me just put in another dictionary here. And all we need to put in here and it's super
simple for this particular first generation is that we're going to put in the input text and the
prompt. So this is the input text is our key. And then we're going to put in the prompt which
will just load in our single sentence prompt. They write a one sentence summary of Las Vegas. And
those are all of the keyword arguments that we want to pass into the bedrock runtime client.
So let's set that up as some keyword arguments that we want. So I'm going to just load those into
a value there, run this cell and now I've got everything that I need. So this is the magic line
now that we're going to use to create our generation. So we're going to call on the bedrock run time
client, which is obviously the object that we created above. And we're going to call invoke
model. And the invoke model, it's all we need to do in order to pass those keyword arguments in.
So let's go and pass those keyword arguments in. We're going to put star star keyword
args. And that's going to unpack all of these keyword arguments here into the input to the function.
So we could in principle put all of this in here, but this just makes it a little bit easier to read
by having it this way around. Again, that's all we need to do. Let's go and load the response
that we get back from that into a variable so that we can see it. And let's go ahead and run this.
So I'm just pressing shift enter on the cells to run them. I've just done that here and it's
completed and it's returned. Let's have a look at the response that we get back. So this will not
actually be the text itself, but it's going to be a pointer to a streaming body response that has
the data that we want. So we have to tap into this streaming body response. It's a fairly
common thing that you'll see when working with the photo three library and when you're
interacting with those API endpoints from AWS is that you'll get this kind of streaming body
response output. So let's just write a small amount of code where we can unpack what this response
looks like. So what we can do is we can work with the response object and we can get out of it the
body. So we're going to grab the body, which is I guess this part here and we're going to call read
on that because we're going to read this photo call response streaming body. And so once we've got
that out, well, we're going to end up with a JSON structure. So we accepted anything, but I will
tell you right now it's going to be a JSON structure that we're getting out of this. So let's
wrap this in JSON load string. So we can wrap that in that. And then let's actually just say that
we want to store that then in our response body. So we're going to say response body equals all
of this. So we're loading out of this string. So we're grabbing out of the response, the body
we're reading that body, we're loading the JSON string and we're dumping that into this response
body variable here. So let's run all of that. And now we've got something we can actually take a
look at. So let's just plonk this into a cell and run this so we can see the output. And yes,
you can actually see it. And in there is the output text itself. So there is Las Vegas is the
29th most populated city in the United States and just a little bit of information about Las Vegas.
So if I just format that a little nicer. So I'm just going to dump that as a string out to the
screen and put some in dense in it. So we can see it. There we go. So you can see here that we've
got the input text token count. So the number of tokens when it's tokenized the input string using
the tokenizer for this particular Titan large language model, it's tokenized it and found nine tokens.
We have our result here. Our result includes a token count of 27 and then the output text that
we've got back from the model itself. Now this is really useful. These token counts are really useful
because the way that the model is charged when you're working on demand is token based. So being
able to see exactly what the tokenizer from the large language model is doing and how many tokens
are in the request and how many tokens in a response is super useful. We also have this, the completion
reason and at the moment it's saying completion reason finish and we'll come and have a look at
that in a little bit more detail in a moment. So if we just wanted to output just the output text,
of course, then all we'd need to do is just run something like this so we can print the body
response. We're going to dig into that and get the results, the first result and the output text.
And if we do that, then we get a clean sentence if you like, which is Las Vegas is the 29th most
populated city in the United States and the most populated city in the state of Nevada,
which I guess is a single sentence summary of Las Vegas. I'd encourage you to pause the video and
change the prompt to something else. For example, you could ask it to summarize a piece of text
since later in the course, that's what we'll be doing. We'll be summarizing transcripts from
audio recordings. Okay, so we've jumped right in there and we've taken a look at a little bit in
depth at how to make a generation with Amazon Bedrock using the Titan model. But there are more
options when it comes to making this generation. So let's take a look at some of the generation
configuration. So this time, I'm going to put a prompt in of write a summary of Las Vegas. You
notice this time I'm not saying a single sentence because I want it to create a little bit more than
just one sentence. So I run that cell and now I'm going to create some more keyword arguments. So
let's go ahead and do as we did before. So I'm going to jump ahead a little bit here and this time
we're going to use the Titan text Express version one, the light version that you saw earlier is the
smallest and fastest model. The express version on the other hand is larger and can be better for
more advanced use cases. So a couple of different Titan text models that we can use. We can talk more
about these a bit later and the same content type and accept and this time our body is going to be
a little bit more involved. So first of all, we're going to set ourselves up with the body
and get ourselves a dictionary to put in here. And in this case, we're still going to do actually
the JSON dumps. So let me do this. So we're going to do the JSON dumps and let's just go and put
some brackets in there. And so the next thing that we're going to put in here, in fact, the data
structure we're going to put in is called text generation config. And this is going to allow us to
add in some extra configuration details that you'll probably expect if you've worked with large
language models to influence the way that the large language model will generate text. So the
first one that we're going to put in here is a maximum token count. So if I just put that in here,
so max token count 100. So this is the maximum number of tokens that we'll get back from the
generation. So it's not necessarily the exact number we'll get back, but it's the maximum number.
So it's a way of limiting the generation that we're going to get back. So the next thing that we're
going to add in is temperature. And so temperature influences how creative the outputs going to be
basically by constraining that random number generator, which is on the output of the large language
model. And so we have that set here to a pretty typical 0.7. And that's absolutely something that you
can play around with depending on the type of generation you're doing. And then the last one
I'm just going to put in here is top P. So we're going to have a top P of 0.9. And again,
this is a way of constraining the number of options that the large language model has when it's
choosing what the next token is going to be as it iterates through. And we're not going to go into
these kinds of settings in a great deal of detail in this short course. If you want to know more
about this kind of thing, then look for the generative AI with LLMs in these specialization courses
on deep learning AI for myself and my colleagues went through all of this in a lot more detail.
So that's how we can pass in some of the configuration options. There are more that you can add in here
as well. But this is a typical set of configuration that you'd send in for the Titan text express model.
So let's do the same things we did before. Let's make sure that they are loaded into a variable. So
we're going to call it keyword arguments or quags. And then we're going to go and generate a
response back. So let me do that here. And I'm just going to paste in all of this in one go.
This is exactly the same code as we looked at earlier in the notebook. So we've got our response.
It's going to come back. We're calling bedrock runtime invoke model. We're passing in all of those
keyword arguments. Then we're getting our response body out from the response that we get back by
calling read on the body and then getting out of that the JSON data from the JSON string that we
get back from the model. And then we can see what that generation is going to be just by dipping
straight into that and getting down to the output text. And then we're just going to print that out.
So we're just basically short cutting all of the steps that we took before. So I just run that.
And when we do we get this back. Now this time again, we asked for a summary of Las Vegas,
not a single sentence. And so it's giving us a little bit more information about Las Vegas. But
notice something here, it's actually kind of stopped mid sentence as far as I can see there.
So let's go and see how we would be able to tell that. I mean, it looks like it to me like it's
finished mid sentence. But if we just get the actual raw data out, so we're going to look at the
response body, but rather than drilling into the output text specifically, we're just going to take
a look at the whole thing. So again, we looked at this before and you'll notice now the completion
reason is set to length. And that means that the model has finished generating because it's hit
the maximum token count. And you can see here, we've got a token count on the output of 100. If you
noticed that when we actually set the maximum token count, we set it to 100. So it's done that.
It's predicted all the tokens. It's got to 100 and it stopped because that's what we asked it to do.
And so we've missed out some of the end of what it would like to have generated for us.
So we can fix that up by going back up to here and maybe just giving it plenty more tokens
to play with. Let's say 500. I'll rerun this cell and then I'm going to rerun this cell and
we should get a new response come back. In fact, I'll rerun this cell as well. So they'll both run
one after the other. Takes just a moment longer to generate more text. But you can see now it's
got plenty of space and it's finished with a full stop, which means it's probably finished writing
what it wanted to write. So if we just look down here in the raw response body, our completion
reason this time is finish. So it's actually gotten right to the end of what it wanted to generate.
And it did it in 304 tokens, rather than the maximum of 500, which we gave it. Okay, so that shows us
how to make generations with the model inside of Amazon bedrock. It also shows us how to use some
of the inference parameters or the generation configuration when we're prompting using the
large language model. And that's the basis which we need for most of the rest of the course.
And again, before we move on from this point, why not pause the video here and experiment around
with this. You can change some of this code. If it breaks, you can always put it back where it was
just by getting the code out of the original source. So pause the video here. I'll see you in a moment.
What we are going to look at though is integrating Amazon bedrock and those large language models
with other data sources. So working with other bits and pieces of data. So let's take a little bit of
a preview about what that will look like. And what we're going to do here is work with audio files,
specifically the recording of a customer service call. So I can play this audio back for you now
inside of this notebook. So if I just import iPython Display, import audio, and this is just stuff
for the Jupyter Notebook. This is nothing more than that. But this is going to allow us to play the
audio file in the notebook itself. And we have an audio file inside of this environment. So if I
just paste this code into here, so audio equals audio, file name, dialogue, mp3, which is the
farming we have. And then if I call display audio, then it will show us a play widget inside of
the notebooks. If I just run that, well, there it is. Now, I'm not going to play all of this audio
file. You can do that in your own time. But let me just play the first 10 seconds. Hi, is this
the Crystal Heights Hotel in Singapore? Yes, it is. Good afternoon. How may I assist you today?
Fantastic. Good afternoon. Okay, so you get the idea there. It's a customer service call between
someone who's looking to book a hotel room and then the customer service representative from
that hotel. So a very typical kind of conversation that you might record from a customer service
center. And more importantly, there's a lot of data inside of there that we'd really like to be
able to extract out and use and process and report on. So that's the kind of thing that we're
going to look at here with the data processing pipeline. Now, what we're going to do is we're going
to create a transcript from this audio file. And the transcript will end up looking something like
this. So I'm just going to open up here a transcript.txt, which is again on the file system of this
notebook. So if I just run this and well, let's print out the dialogue text, then you can see
probably what you'd imagine. But here is speaker zero. Hi, is this the Crystal Heights Hotel in
Singapore, the file that we just listened to. So this is a transcription. So we're going to look at
how we can create this transcription from this audio file. And then we can take this transcription
and we can pass it into a large language model. And the way we start to do that is by doing some
simple prompt engineering. And we can look at a couple of different ways of doing that in this
short course. And but here what I'm going to do is I'm just going to paste in this section of code
here. So this is setting ourselves up with a string. We're using the Python formatted string syntax
here with triple quotes so that we can have a multi line string. And we're saying here in this
prompt, the text between the transcript XML tags is a transcript, and then just have to scroll over
of a conversation, write a short summary of the conversation. And then what we've done here is we've
got transcript and then dialogue text. And that dialogue text will get replaced with the actual
dialogue that we have up above here, because that's the name that we gave to this variable.
And we loaded it just out of the text file for the moment. And then it finishes up by saying here
is a summary of the conversation in the transcript. So basically giving the model, the large language
model, plenty of prompt space here to start to generate the summary for us or a summary of this
particular conversation. So let's run this cell. And then I can just quickly just print out this
prompt. So if I just say print out prompt, run that, then you can see what you might imagine.
There is the template now filled out with the transcript between these XML-like tags. And
this isn't necessarily something you have to do, but it's a way that you can indicate to the
large language model that the data that's between these two tags is something which you've now
essentially labeled. I've labeled it as the transcript of this conversation. So if the conversation
itself, I don't know, starts to talk about transcripts, then it shouldn't get too confused about
that. And it should know that everything between these tags here is the transcript from the phone
call itself. So it's a prompt engineering technique. Let's just have a go with sending that into
the TitanText model. So I'm just going to put my keyword arguments here. This again is exactly the
same as we've seen before. My prompt now there is going to be from this prompt that we have defined
up above. And then let's go and get our response out from our Titan model by calling invoke model.
And then let's just go and add in a couple of lines where we can unpack that and get our
generation. And I guess the last thing that we need to do is print that generation out. So let's
just put that in here and say, here we go. So here's the generation. Here is the summary of the
transcript of the phone conversation. So Alex is looking to book a room for his 10th wedding
anniversary, et cetera, et cetera. So we've been able to take the audio file, transcribe it,
extract some information out of it. In this case in a super simple way, we'll look at a little
bit more detail in subsequent lessons in this short course. And then the next lesson, we'll take a
look at how we can use Amazon transcribe to take the audio that we've got from our recording in the
call center and then extract the actual transcript out that we can then pass into the model as we've
done here. So we can dive in at a deeper level. So when you're ready, I will see you in the next lesson.


## 3. Summarize an audio file
In this lesson, you'll see and hear how to take audiophiles of conversations between two people,
in this case between a customer and a company representative, and transcribe them.
You'll then process this transcription and use an LLM to analyse the conversation.
This architecture could be adapted to work with other audio sources, such as recordings of meetings
or presentations, or to transcribe audio from video files. So let's tune in.
And before we get going with the actual code, let's just step through the process that we're
going to follow in this lesson. So we're going to use Boto3 again, and we're going to set ourselves
up with a couple of different clients. So not initially the Amazon Bedrock client, we're going to
connect to S3, and we're going to connect to transcribe. And these services are going to form
a very important part of the data processing pipeline that we set up. So with those services set up,
we can transcribe our audio file. We're going to do that. We're going to create a unique name for
that. We're going to then transcribe that audio file into a text output. Then we're going to need
to take that text output and convert it into a text format that we can use inside of the prompt
and then send it to the large language model. So it's a form of prompt engineering that we're doing
where we're converting the data ready to put into that prompt. And in this lesson, we will
actually send the transcription off to the large language model just to see the kinds of outputs
that we can have. And then in the subsequent lessons, we're going to start building that
further and further into our fully automated pipeline. So let's just remind ourselves of the
audio file that we're working with. And I'll re-import the library that we used in the last lesson.
So that's the ipython display audio control. So I'm just going to run that line and then run
a couple more lines that we've lifted again directly out of the last lesson where we're going to
put a widget inside of our notebook, which allows us to play dialog.mp3. So running both of those
lines, exactly the same lines as we had before, little widget loads up and just really quickly,
let's remind ourselves what that audio sounded like. Hi, is this the Crystal Heights Hotel and Singapore?
Yes, it is. Good afternoon. Okay, so it's the dialog between a customer calling up the hotel,
and then the hotel staff working on that call and working with that customer. So we've got this
audio file. And again, this is inside of our local environment, so this is running inside of the
notebook server that we're using. But what we want to do is we want to start moving to the cloud. We
want to start moving to a serverless architecture, which is running for us in managed services inside
of the cloud. And so the first thing that we're going to do is we're going to upload this object,
or so this file, into the simple storage service. So AWS is S3 service, the simple storage service.
So we do that using Boto3 again. So I'm just going to import that now. So the Boto3 library again
is the SDK for AWS that runs in Python. So let's run that cell so that it's defined. And then we can
go ahead and create ourselves some clients again. Again, if you remember from the last lesson,
we do Boto3 client. And then using that notation, we can create a client, which will connect us to
the various services, which run inside of AWS. Last time we did it for Amazon Bedrock. And in this
particular instance here, we're going to create a client for S3. And in the same way that we did last
time, we're going to point it to a specific region for this particular API endpoint. And we're going to
say the region name of US west two, just to keep everything consistent with the way that we're
running these notebooks. So let's store that in this S3 client object here. So we've got S3 client
and let's run that. So now we have an S3 client to find. Okay. So let's go ahead and upload the
dialog MP3 into S3. We've got an environment variable that is being set. And we can use the name
that's setting that environment variable to get access to an S3 bucket, which is already being
created for us in this course notebook. So I'm just going to import OS into Python so that I've got
access to the environment variables in this particular environment. And the way we get access to
those environment variables is to call on OS dot environment and then ask for a specific name
inside of the environment variables that it's got access to. In this particular case, we're going
to look for resource S3 bucket name. And this is just going to bring back a string. So let's go
ahead and store that string in this location at bucket name. Of course, all we're doing here is
just pulling the environment variable out of the notebook environment. And of course, if you're
writing your own code, then you can go ahead and have this as a hard code string or your own
environment variable or wherever you want to get the bucket name from. So let's go ahead and run
that. And now that's defined. So the value that I have in here right now will probably be different
from the one that you have in your running environment. But as long as we've got the valid bucket
name stored in there, then we're happy. So from there, let's go and define a file name for our object
that we're going to upload. So file name in this case, again, it's just a string. It's dialogue
dot MP3. So we've got that defined for when we upload the object. And it's simply this. So we're
going to take the S3 client. So let's say S3 client, which is the client object we created before,
we're going to call upload underscore file. And then into there, we can add the file name,
the bucket name, and then the file name again. And the reason why we're doing this is because we're
pointing to the local file, then the bucket name, and then the name we want to give the object
once it's uploaded into S3. So we can go ahead and run this cell. And so now our object is uploaded
into S3. So now we're starting to have a cloud-based architecture. All right, now we can
create and perform a transcription of this audio file. And to do that, we're going to create ourselves
a another client object. So for this client object, we'll go to photo 3 again, we'll load up our
client. And this time, we're going to create a transcribe client. So transcribe, so Amazon transcribe
is the name of the service. And then we'll have a region name once more pointing to
US west two. And again, we're just going to store that in a variable name. And that variable name
can be transcribe client, just trying to keep some consistency there. And I'll run that. And now I've
got my client defined. And so now I'm going to use this service and send it a command, an API
command asking it to transcribe our audio file. And we're going to tell it the location of that
audio file inside of S3. So the transcription job needs a unique name. So I'm going to use UUID
in order to provide that for me. UUID stands for universal unique identifier. It's a unique
string of letters and numbers. And that's what this UUID library helps us to generate. So let's just
import that library. And so then let's create a string using UUID. So we can build this up. So
we're going to have UUID and we're going to use UUID4. UUID4 is one of five methods for generating
these unique identifiers. UUID4 is good enough for what we need in this particular application.
This will output a long and hopefully completely unique set of characters before we can't use it
like that though. So we want to wrap that up in string so that we've got something a little bit
more usable yet again. And so this is getting useful. We could use this kind of approach to get a
unique name for the transcription job. So what I'm just going to do is just put in the beginning of
this that we want our job name to be transcription job, hyphen, and then whatever this is. Lots and
lots of different ways of concatenating strings together inside of Python. This one will do for now.
So if we take a quick look at our job name, then we've got a unique name that we can give to
the transcription job. All right, so let's get into the guts of how we actually call this. And so
we're going to call the transcribe client and then we're going to call the method on that which is
start transcription job. So if we just give ourselves some space here to put in some arguments so
we can pass some details about this job. So the first thing that we're going to do is we're going
to pass in the name that we've defined. So we're going to say that add transcription job name is
job name up above. Then we're going to give it the audio file that we want it to work with. Now this
audio file has been uploaded again into s3. And we've got a bunch of variables that we've set previously
about the location in s3. So if I just paste in this command here, we're pointing to where the
media is and we're saying it's a media file URI. And then we're doing this formatted Python string
here to show the location inside of s3. So this s3 colon forward slash forward slash the bucket name
and then the file name or the object key is the way that we can specify exactly where that
piece of media is. So once we've got that, we can add in a couple more settings here. So this is
our way of telling it the media format. So we're sending it an MP3 file and then giving it an
idea about the language. So it is possible for it to detect the language automatically. But we
just make it easier for it if we're telling it. So we're going to tell it that this is US English.
And then we supply a bucket name for the output bucket. In this particular case, we're going to
use the same bucket as the input bucket for the output bucket. So this is where we're going to
deposit the output of the transcription job. And it's really useful to define where the output of
this job goes so that we can start to look at event-based architectures later on in the course.
And then finally, we're going to put in a couple of settings here for the AI model itself. So it's
going to identify different speakers who are speaking inside of this transcription. So we're
basically telling it that we know that there's going to be more than one speaker and we want it to
label those different speakers when it finds them. And we're telling it to look for a maximum of
two. The transcription service can look for a maximum of 10. But just putting into here means
because we know it's going to be a conversation between two different people. Okay, so this is all
pretty much set up. We want to grab the response out from this call to this service. So let's go and
store the output from all of this in response. And then we can go ahead and run this cell. And before
we move on, take a look at this code and make some adjustments to it. So play around with it, tweak
it a little bit and see what you can make it do. So maybe pause the video here and have an experiment.
So it took us a while to go through the process of building up that transcription job. And you might
notice that it actually returns immediately. But it hasn't necessarily finished yet. So let me go ahead
and paste the whole of this next cell in. I'm going to run it and then we're going to talk about
what it has. It's already finished by the time I've managed to do it. But if you just run this cell,
all this is doing is it's basically waiting every two seconds and going to check that transcription
job to see if it's actually complete or potentially if it's failed for some reason. So when we start
the transcription job, it's sending a command to the service to go ahead and transcribe the file.
But it's not coming back with the actual output in response. It's coming back with a pointer to
the ongoing job. Now this particular audio file isn't very long. And so in the time it took me to go
and copy and paste this piece of code in here, it had already finished. But you can see the point
if this is going to be a much longer transcription job, then we want you to give it just a few more
seconds or a minute or two to run through all of that. So this is just a simple loop, as I say,
just pausing every two seconds to go through and look for the status. And it gets the status by just
looking at the client and then asking for get transcription job. And then we pass in the name
of the transcription job that we created previously. And so it just goes through there and looks to see
if it's finished or not. Anyway, it has. So we can go ahead and carry on with our code. And we can
take a look at the output of that transcription. So let's import Jason so that we've got some Jason
that we can work with. So we've got the Jason library that we can work with. And let's go and grab
back the text from the transcription. So I'm just going to build up some code here. And this is
why I spent essentially I'm going to paste this first line here. So I'm basically going to check one
more time to make sure that it's complete. That means that if we do run this code in the future,
then it'll only run if it's actually going to work if you see what I mean. So let's then go and
put in some values here. I'm just going to paste in a block here. And there we go. And this block is
going to derive the transcription key. In other words, the file name that's inside of s3 that
will have been created. And the naming format is the job name and then dot Jason. So again, another
reason why we have that unique file name. And then it's going to grab the actual object itself using
our s3 client. It's going to use get object. And we're going to tell it which bucket and which key,
in other words, which file name. And it's going to be able to go and grab that particular object
for us. And then we can read that object. So we're not even saving it to disk or anything. We're
just going to read it using that dot read that we looked at before from the streaming body return.
The similar thing that we saw in the last lesson. And we're also going to run decode on it as well.
And then we should have our text there. Of course, that text then is going to be a Jason structure.
So we're going to load that. And then finally, we're going to have the transcript Jason object here.
So let's run that and then take a quick look just by dumping this into the cell and running it.
And we should see there we go. We have the output from our transcription job. And you'll notice
that there's actually a lot of information in here. And so we've got the again, the job name.
And then we've got the status that it's completed. And then we've got the results itself. And so we
got transcripts here. And this is all of the text. You'll notice how there is no differentiation
between who's speaking in this particular part. If we scroll down and down and down and down,
then we start to see some more information here. So we've got speaker labels. And we can see the
different speakers. So we have speaker zero. And all of these different timestamps from when they
were talking, speaker one and timestamps for them as well. If we scroll down even further,
then we start to see a different kind of data. Here we go. So you can see the individual
words. So let's have a look. Hi. Is this the crystal heights hotel? And those words there
are coming from speaker label. There we go. So speaker zero initially, speaker zero said this
and this and this and this and this. So using this part of the output, which is inside of items,
then we can get the particular word and who said it. And we can start to build up the transcript
in a format that we want to see and we want to use. So let's go ahead and do that. Let me just get
rid of this for one second. And let's expand out this code that we were working on. We saw
in there that the information we were interested in was items, which is inside of results. So let's
go ahead and pull out just the items. And then let's just go and loop through those items. So let's
say for item in items, we're going to go and print the item. So this should get us a little bit
closer to what we're after. So I just run that and now we're straight into just this section of
the document where we've got the individual words and who said them. So this is getting us a lot
closer to what we want. So instead of this, let's go and pull out the speaker label. So the speaker
label is here and we've also got the content itself and the content is the word. So we want the
content, the word high and the speaker label of who it is. So those are things that are useful to
grab. So let's instead of printing that, let's remove that and set up some a couple of variables
for ourselves. So speaker label and content where we're getting that speaker label out, we're getting
the content, which is actually inside of alternatives. As you can see, it's the first one in the list
and then content. So that will give us the speaker label and the word, which is said. So that's
useful. Again, let's step by step through this, make sure that we're getting what we want to see.
So let's just print that in a formatted string. And yeah, so we can see we're getting what we want
here. So the speaker and what he said, we want to have just speaker zero and then the entire thing
that speaker zero has said and then speaker one and the entire sentence that speaker one said. So
we just need to do a little bit more formatting of this code and then we can get that. So let's set
ourselves up a couple of variables up here before we get to the passing. We're going to set up the
output text. So this is somewhere we can store the final output and we'll store the current speaker
as well. So we know who it is that's currently speaking and we can just traverse through this list
and concatenate together the string for us. So let's go and replace this line here with some code
here, which is going to look at the speaker label that's come through and see if it is the same as
the current speaker or not. And if it isn't the same as the current speaker, then we'll set it to
the current speaker and we'll start that output line by using this formatted string and saying,
okay, well, this is the current speaker here. And then obviously what we'll need to do is concatenate
the rest of that content. So we can do that by just pasting in this line here. So output text,
we're adding to the end of that output text, the content. So this is going to loop through all of
these lines. It's going to look for a speaker whether it's different or not and it's just going to
start concatenating these strings together. Now, the other thing that I'm just going to add in here,
and it's one of those things that I'm just going to add in, it's sort of like trust me, this will
help just to make sure that the formatting is nice is just something to deal with some punctuation.
So you might have noticed when we looked through the different item types before, we have got the
words, but we've also got some punctuation. And so we're just going to strip the output from that
and remove the trailing space. And it's probably not entirely necessary actually seeing as we're
passing this to a large language model, but for the sake of making it easier for us to read, then
let's do that. All right, so with all of that together, if we run that cell again and now look at
output text, then we should see that we've got our transcription looking more or less like it's
in order. Now it's not printed out very nicely, so let's just do some print brackets around that
like this. And now we can see our transcription coming together. So speaker zero says what they say,
speaker one, then speaker zero, and through it goes. So this is looking pretty good. The only
thing that we want to do now is we want to save this transcripts to a file because we don't want to
just print it down on the screen. So just by using the open command from Python, we're going to save
this into a text file locally, and we'll use the job name again as a unique identifier locally for
this particular text file. So let's run that. So with this code, obviously there are bits in here
that we could actually change. So if you'd like to have a go at doing that, then pause the video now,
change some of those bits inside of the code, see what difference it makes. And when you've had
enough, then I'm pause the video and I'll see you in a moment. We're going to call
Boto3, we're going to go to client, and we're going to go ahead and load the bedrock run time
so that we can make some inference or make some generations with our model. Let's remember to put
in our region name, and we're going to put in us west two, of course, again, and we're going to save
all of that into a bedrock run time variable. So let's do that and run that. Okay, so now we have a
bedrock run time client ready, but we do need to actually sort of following through as though we
hadn't saved it to the disk, I suppose, or so we're working with this as a sort of standalone
piece of code. I'm going to just put some standard Python in here to open up that file name and
read that file in to a variable that we're going to call transcript. So again, just sort of like
simulating, if you like, that we hadn't done the previous step or that previous step had happened
at a different time. So we've now got our transcript loaded in. Okay, so let's go ahead and prompt
with this. Now we looked at a way of doing that in the last lesson where we just did some fairly
simple string concatenation to put a simple prompt together and get a summary of this transcription.
But this particular case, I want to do something a little bit more sophisticated than that.
So in previous examples, you've created prompts within strings within your code,
and there are many ways to do that in Python, such as basic string concatenation and F strings.
But for production cases, such as here in this serverless cloud architecture and in this course,
it can be more helpful or more manageable to define a template in a separate file. In this case,
we're going to use ginger, the ginger templating library. And by doing this and defining the template
in a separate file, you can look to apply version control specifically to the template. And by
doing so, separate out the prompt, which might be being code developed by the business user
from your application code. This makes it possible to modify the prompt and to swap in different
prompts during production. So whilst the code's actually live. And so what I'm going to do is I'm
going to use this magic notation with Jupiter notebooks. And we'll see this again in a later
lesson as well, where we're using this right file. And then we're giving it a file name. And so
if we do that, then any text which we enter into this part of the cell down here, when we run
this cell, it will save that text into the file name, which is written here. In this case,
would be prompt template.text. So I don't want it to be just text. I want this to be the,
obviously, the template for my prompt. So I'm going to paste in something here and we can
step through this and take a look at the way it fits and where it works. So this template,
this prompt template that I've got, again, it will be saved as a text file here. And I can just
put a couple of line breaks in here so we can read it a bit better. So I need to summarize a conversation.
The transcript of the conversation is between the data XML-like tags. And there we have the data
XML-like tags and the notation, which we'll add transcript in here. And so we're going to have a
look at a slightly different way of doing our prompt templating using the ginger library. And then
I've got some more instructions here. The summary must contain a one word sentiment analysis and
a list of issues, problems or causes of friction during the conversation. The output must be provided
in JSON format shown in the following example. And then we give an example of the actual output
that we'd like. And then we say write the JSON output and nothing more. And then I say here is
the JSON output. And so we're hoping that our next token predictions will build up an output
from the LLM, which will give us exactly what we want in this kind of format here. So that's a summary
of how this particular prompt will work. And it's outputting our summarization or it's outputting
information about our transcript in a slightly more system usable format in this JSON structure.
So let's run that. And by doing that, you can see that it will, in this particular case, it's
overwriting it because in this environment, I've already done this to make sure it'll work.
But it should say that it's written the text file for you there. And if you decide that you'd like
to make some changes to this particular prompt, then you can go ahead and do that inside of the
Jupyter Notebook, run this cell again. And it will overwrite it for you as well. And overwrite
the prompt template.txt. So once we've got that, we can use the ginger library, in the case ginger
two, in this particular case, to fill out the information in this template. Now it's a little
bit overkill because this is such a simple template that we've got. But it's a useful technique to
be able to do this. And it's a very capable templating library. And so we can use it, and we will
use it a little bit later in the course as well. So let's import ginger two and import template
from ginger two. And then we are going to need to open up the template that we've just saved.
So all of this that we just typed in here saved this template out to prompt template.txt. So
again, with our standard Python notation to load in prompt template.txt and load that into a template
string. So let's run that. And again, if we look at the template string, it'll be exactly as we
think there is the template as defined up above. So let's get rid of that.
Okay, so the next thing that we need to do is we need to define the data that we're going to
insert into our template. And in this particular case, there is only one thing that we want to put
in there. But we're still going to define it in the same way as I would do if we had a lot more
data. So let's go ahead and say that we're going to create a data dictionary and we're going to
transcript. So this is basically pointing to this particular name here. So I'll copy and paste
it to make sure it's exactly the same. And we want to put into there the transcript that we have
loaded out of our file before, which happens to be called transcript. So let's go and use that
and put that there. So now we have our data dictionary ready to insert into our template.
So I can go ahead and run that. So that's defined. And then I can call on template and I can
load template with template string. So this is the ginger to template object here. And we're
going to create one of those using the template string. So that's the format that we've got. And we
will call that template so that we can go ahead and use that. So let's define that. And then we
just call render on that template. So we can say template dot render. And then we pass in that
data. And that's all you have to do. I do want to, of course, save the output of that somewhere
or just have that stored somewhere. So let's store that inside of a string called prompt.
Okay, so a lot of different lines to do something fairly straightforward. But the architecture
there is super powerful. So we'll be using that later on as well. If we take a look at prompt
now and print that out. In fact, let's just go ahead and print it out rather than dumping it out
raw. And then there we go. There is our current prompt. So exactly as we saw before with the data tag,
then all of our transcripts going in there. And then the end of the data tag and the rest of our
instructions all set out for us in our prompt. So all that now remains for us to do is to pass that
into the Titan text model using Amazon bedrock and take a look at what the output is going to be.
So let's go straight in and paste in the keyword arguments exactly as we've seen in the last
lesson. So we're calling on the particular model we're going to use. In this case, it's the Titan
text Express V1. Everything else should look similar from the last lesson. In this case, though,
I've turned the temperature right down to zero. And that's because I don't particularly want
this to be creative in its output. And I want to try and ensure that it's going to get a nice
syntactically correct JSON output. So by turning the temperature down, we stand a good chance of
that working for us. So let's run that. So we've got that defined our keyword arguments. And we'll
just put the boilerplate code in that we've seen before. So bedrock runtime invoke model passing in
those keyword arguments. We get our response back. And you'll notice this is taking just a moment
or two to run. And while it's running, let's get the next code ready to go. So we're just going to go
again, we're going to go into the response. We're going to get the body out. We're going to read it.
We're going to load the JSON out of there all in one line. Get the response body. And then inside
of that response body will get results zero and output text. It is worth noting actually, while I
mentioned this, that the response body format will be slightly different depending on which of the
large language models you use from Amazon bedrock. In this particular case, because it's the
Titan model, then we know the location of the output is like this with results zero and output text.
But if you use something like the Claude models or the AI21 labs models, then the output and the
location of the output text or the generated text will be wherever it has been defined by those
development teams. So let's run this. So we get our generation and just print our generation out
and take a look at what we've got. And we should have, in indeed, we do have our JSON structure,
which provides us with a single word sentiment analysis. So it's positive. Overall, the transcription
was positive. And then we have a list of all of the issues which have been identified. And at the
moment, it's pulled out a whole bunch of different issues, talking about what speaker one has done.
And in most of these cases, they seem relatively positive. We can filter this more by being a bit
more specific in our prompt templating and also in our prompt writing. And we'll have a look at
how to do that in a later lesson inside of this course. But for now, what we've done is we've
used Amazon transcribe to transcribe the audio file using serverless services inside of the cloud.
We've stored the actual audio file in S3 in the first place. And then we've used Amazon bedrock
again, serverless service, fully managed service with that large language model, the Titan model
to come out with this generation. And so we're really pulling together the different building blocks
that we need here to be able to implement this fully serverless, event driven architecture,
data processing pipeline. Now, we've looked at the code in some detail here, including the prompts
that we're using. And we've printed them out to screens so that we can see them too. What you
want for that is an effective logging solution. So for audit and compliance and for the all
important debug logging is super useful. And that's what we're going to have a look at in the next lesson.


## 4. Enable logging
So far in this course, you've started to integrate LLMs and other AI capabilities.
Part of developing and managing a deployed application is addressing security, audit and compliance.
And one of the key tools for auditing your compliance is logging.
In this lesson, you'll learn how to enable logging for all of the calls that you make to LLMs
that are within Amazon Bedrock and how to review these logs. This is extremely powerful for
audit and compliance, and you'll use this to monitor for error conditions. So let's take a look.
So it would be completely possible to put in lots of logging code into your application stack and
start to push out various bits and pieces of what you're sending to and getting back from the
large language model in there. But sometimes that's not as easy as all that, especially if you're
using libraries such as Langchain, for example, and Amazon Bedrock does work natively with Langchain.
And so one of the things I certainly found with Langchain as brilliant as it is, is it can sometimes
be a little bit tricky to know what's happening behind the scenes with all of that prompt engineering
it's doing and sending data to and fro from the large language model. And so what we can do here
is we can actually set up Amazon Bedrock to globally log all of the calls which are made to and the
responses you get back from the model when it's making its generation. So let's have a look at how
we can do just that. So the first thing I'm going to do is just import a bunch of stuff that we're
going to need for this particular lesson. And it's all stuff that we've seen before. And so we can
just go a little bit faster through this. So importing boat 03 Jason and OS again so we can get our
environment variables and then we're going to go ahead and create ourselves a bedrock client.
So this is not a bedrock runtime client. It's a bedrock client because we're now going to be
looking at and configuring the Amazon Bedrock service itself rather than making a generation
just in the first instance. So let's run that cell so that we've got all of that up and running.
We then have this other cell as well and you'll see this in the code that we've supplied over on
the other side of the screen. But we've got these helper files and I've just written a quick
cloud watch helper. So Amazon cloud watch is the service which deals with logs and metrics and
just the general health and the what's going on inside of your AWS account. And just to make it a
bit easier to work with I've created some helper functions here and in this particular case it's
a cloud watch helper function to help us see what's going on inside of cloud watch. You can also see
what's going on inside of cloud watch through the console as well. And that's probably the more
logical place to go. We're going to have a look at both of those. Okay, so now that we've got everything
set up we can start to write some code. And the first thing that I'm going to do is I'm going to
define for us a log group name. So inside of cloud watch logs it has these log groups and you get
to define what that naming structure looks like so that you can save essentially where we want to send
our log files to. So in this particular case I'm going to send it to this log group name of my
Amazon bedrock logs. Now it is important actually for this particular lesson to use that specific name.
And a reason for that is because we've got some permissions set up inside of this account and
the permissions are expecting to work with this particular log group name. So if you do want to change
it you can but you'll just need to change those permissions as well if you're running this inside
of your own account. Now just a word on what those permissions are. They are IAM permissions IAM
stands for identity and access management which give the Amazon bedrock service the ability to put
logs or to create log streams and to put log events into cloud watch cloud watch is the part of AWS
that aggregates all the logging and metrics and logging and you can use it to add monitoring and
logging to your applications. So let's take a bit of a look at that quickly just with this slide overlay.
So this is the JSON structure of what the permissions look like and you can see here that they have
this statement structure where we have the effect of allow and the action here that we're going to allow
is to create log stream and to put log events and you can see here that we got it's locked down
to this particular resource. So in other words we're allowing the Amazon bedrock service and only the
bedrock service model invocations here to go ahead and put those logs into the log stream.
And so the other part to this then is the trust policy and just quickly this trust policy basically
means that this policy can be assumed by the bedrock service. So both of these things have been created
for us already inside of this account and this is just a quick note to what we've already done
and the fact that this role exists. So now that we've defined our log group name of course this is
just a string. I'm going to go ahead and use my helper function which we've loaded in as a cloud
watch here and we're going to call this command here to create the log group. So this is just going
to send an API call off to the cloud watch service to create the log group called my Amazon bedrock
logs. And you can see it's just returned back here that that's being created for us successfully.
Now that's really easy to do inside of the AWS console and so that's why we've not gone into
that in a great deal of depth. What I wanted to show you is how we configure Amazon bedrock to send
those logs into cloud watch logs. So let's have a look at how we do that. So we're going to create
ourselves a dictionary of parameters that was going to be essentially be our logging configuration.
And we're going to send this into Amazon bedrock. So let's do this now. Let's just say this is all
going to be our logging configuration. So the first thing that we need to do is set up a cloud watch
config. So let's go ourselves a dictionary for that. And then we're going to have our log group name
that we pass in. So this is defined as we have already here with log group name of my Amazon bedrock
logs. So this is telling Amazon bedrock where to put those logs. And then we need to tell it about
the role. So this is the role that we saw the outline of just a moment ago. And there was an
environment variable inside of this notebook currently running, which will point us to the ARN or
the Amazon resource name for that particular role that's already been set up. So we've got that
there. And then we can define how we want to deliver particularly large objects. Now this is
because Amazon cloud watch it's really good at storing sort of log entries. So text entries. But
imagine for a moment that we've got a massive generation, which has been created or if we've used one
of the text to image models, so we're actually generating some images. Well, if we want to log all of
that, then we're going to need to give it an output location, which isn't just cloud watch,
something which is capable of storing larger objects. And the clue there is in the name objects,
we're going to put this inside of s3. So let's go and paste in this configuration here and just
have a look through this. This is the large data delivery s3 config. And we're giving it the location
of a bucket that we want to use to store those large objects. And we can also give it here a key
prefix as well. So inside of that bucket, let's give it a particular prefix. So it's all
sectioned off into a particular area. In other words, this is a bit like having a folder called
Amazon bedrock large data delivery. You could call that whatever you want. And in this particular case,
again, we've got a bucket, a logging bucket, which we've got the name of that bucket defined
inside of our environment already. So that is the cloud watch config there. Now with the Amazon
bedrock logging options, we have the choice of being able to put logs into cloud watch then also
with a large objects going to s3. But we could instead put things straight into s3 and not use
cloud watch at all. So it's just another one of the options. And we can do that with this configuration
here. So in a similar way to having cloud watch config, we can have s3 config. And so we can go
ahead and put the logs directly into s3. So this is different from this section up here. This section
is just where to deliver large things, which relate to the cloud watch logs. This is where we're
going to put just anything. And in this particular case, with both of these things enabled,
it means it will put the data inside of cloud watch and inside of s3. So we're just doing both
just for the sake of it in this particular lesson. Then we get to say what kind of data that we
want to have logged. In this particular case, we're working with text data. So we'll say the text
data delivery is enabled. And we'll set that to true. And that is our logging configuration.
So I can go ahead and run that. Of course, that won't do anything at this point. It's just
a dictionary, which has been defined. But I can now go ahead and call on our bedrock client agent
that we had before. And I'm going to call a much longer method name than we've called before. So
this by the time it's going to be called put model invocation logging configuration. And then we
are going to pass in our logging configuration in there. And so we can do that just by passing in,
the logging config is the logging config. So in other words, we're going to point the logging config
to this data structure that we just had up there. So that's all we need to do now to go and apply
that logging configuration to Amazon bedrock. So let's run that. And then we get a response back. And
the response is code 200. So it's all positive. It's been applied. Now while we're reviewing the
different calls that we can make to the bedrock client, let's have a go at this. So bedrock get
model invocation logging configuration. So as opposed to put, so this should return to us the
actual logging configuration and what it looks like. So if we scroll down here, then indeed we see
all of those values that we sent up into the model. And in my particular case, I've got bucket names
which relate to the environment variables that I have inside of my account and yours will be
different. So with the logging enabled, it's now enabled globally across our entire account for
any invocations that we make to those text models inside of our account. So let's have a go at
making a text generation so that we can see what it looks like. And I've just pasted the entire line
in here. We've seen it many times before. This is the bedrock runtime client that I've created inside
of us west two. So let's run that cell and I'm sure you can guess what's going to come next. Well
almost I'm just going to paste it in in one cell because we've seen it so many times before. Here's
our prompt. In this case, write an article about the fictional planet Fubar. Then we have our keyword
arguments. Much as we've seen before, oh, the temperature is a bit low. Let's set that temperature
a bit higher because we are doing something a bit more creative here. And then we're going to go
get a response, the response body, the generation, and then we're going to print the generation out.
And before we move on, take a look at this code and make some adjustments to it. So play around
with it, tweak it a little bit and see what you can make it do. So maybe pause the video here
and have an experiment. Of course, the reason why we're doing this is because we want to perform
some generation so that we can then go and have a look in the log files and we should be able to
see that it's in there. So once this is finished writing, it's awesome article about the fictional
planet Fubar. I've got one more thing inside of the helper functions that we created in order to
help us see what's turned up in the logs. Now, we do need to wait just a moment before we run this.
So this is just the helper function, the cloud watch helper function that we've got inside the lab.
And then it says print recent logs. What this is going to do by default is it will look in our log
group name, which is defined up above. And it will pull back all of the entries from the last five
minutes. So if you are running this multiple times, then if you're not seeing things, then it's
because it's only pulling back things from the last five minutes, just to keep things sort of simple.
But let's go ahead and run that now. And fantastic. We actually get to see the output. Now we'll go and
take a look at this inside of cloud watch itself in the console in just a moment. But this string here,
permissions are correctly set for Amazon bedrock logs. This is actually something which is sent into
the logs by the backend system. So when we're giving it the ARN and ARN stands for Amazon resource
name. And these are unique identifiers to every instance of a resource inside of AWS. So in your
account, all of the instances of things have individual ARNs for the role that needs to be able to
perform the logging. And then we set those logging parameters and that logging configuration. The first
thing that Amazon bedrock the service will do even before any generation is made is just go and check
to make sure that all works. And so that's all that string is there. You'll see that once and then
it will disappear in the logs and you'll never have to see it again unless you set it up another time.
So I then just do this quick line so we can see the difference between the different lines here.
And here is the entire JSON structure of the particular call that was made to the service. And you
can see here the input text write an article about the fictional planet Fubar. You can see all of the
parameters that were sent in as well. Super super useful for debug super useful and maybe essential
for audit and compliance. You can see the input token count and the output token count. And of course
you can see the output text as well. So there is a chance that this output could get quite large,
especially if you're doing things other than text based models. But this is super useful for
being able to log globally. What's happening inside of your AWS account using Amazon
bedrock. And for me, more than anything else, this is really, really useful for debugging all sorts
of applications, no matter what libraries you're using, no matter what architectures you are using.
So if I want to see those logs inside of the AWS console, well, it's probably a little bit
easier to see than it is inside of a notebook. So I can go to my AWS console page and you can do that
by following the link that's provided in the notebook environment. I can click on the magnifying
glass here and I can type in cloud watch. Now you'll notice that this is quite narrow on the screen
here. So you'll probably see more than this if you're doing this inside of a completely separate
window. But let's navigate to cloud watch. And then inside of cloud watch, I can go to log groups here.
So if I click log groups, the menu should collapse away. And then I'll get a list of all of the log
groups that I have. And here you can see my Amazon bedrock logs, which was the log groups that we
created just a moment ago. And if I navigate into there, then I will be able to see the log streams
right down the bottom here. And if I click here, I've got AWS bedrock model invocations. And so there
I have exactly the same view as we were looking at before. So here are the model invocations which have
arrived into cloud watch logs inside of the console. So that's how to log. So in the next lesson,
we're going to get back to our event driven pipeline. We're going to put together all of the code
that we've been doing inside the notebook now in the cloud in serverless managed services. And
it's going to be a fully automated data pipeline working on our transcript. So when you're ready,
I'll see you in the next lesson.


## 5. Deploy an AWS Lambda function
In this lesson, you'll learn about AWS Lambda functions, a serverless type of compute,
and one of the easiest ways to manage code. You'll see how you can use them to run your LLM
powered workloads. Let's jump right in. And by now, maybe just now, you've realized
how long this lesson is. Yes, it is quite long because there's a lot for us to go through. So
in a break from tradition, let's actually take a look at what we're going to cover in bullet
points in this particular lesson. And what you can do is you can skip to the bit you want,
or maybe more practically, you can stop at some point and come back later if you want to.
You've got all of the code, of course, as per all of the other lessons, it's already there in
the notebook. So if you want to, you can just rapidly run through some of the cells and progress
forward if you're coming back after having taken a break. So in this lesson, we're going to,
first of all, take a look at some of the helper functions we've got available.
So I have written help helper functions that will help us get some things done faster. Yeah,
believe it or not, that's a shortcut. So once we've got those, then we go to take take a look at
some of the data that we're going to be deploying into our Lambda function. So we'll take a look at
the prompt and the prompt template and how that's been set up for us. Then we'll take a look at
the Lambda function itself. And this is going to be the majority of this lesson. So we take a look
at the Lambda function and piece by piece will build the Lambda function up and we'll look at the
different code and how it works and why it's been structured in the way it's been structured.
So that will take the significant amount of time. Again, you've got all of the code right there.
So if you don't want that detailed walkthrough, you don't have to have it. Or if you want to come
back to the lesson another time, then you can always go into more detail at that point.
Once we've got the Lambda function defined, we deploy it and then basically we're going to do a test.
And so we're going to see the Lambda function working and then we're done with this lesson
and we can move to the next lesson. In the next lesson, then we bring it all together into the
overall pipeline. And it's a much quicker lesson because we've kind of built this foundation if you
like inside of this one. Okay, so if that's all good with you, let's get started. Now as you'll
notice from the code over on the left hand side, there's a bunch of helpers and things that we've
got to help us through with this so that we don't have to do complicated deployments and things
inside of the Jupyter Notebook. I've got some helperscripts. But let's start off with the usual.
So we're going to import Boto3. I'm also going to import the OS, the operating system library from
Python so that we can get access to environment variables and things like that. So let's shift
enter on that, run that cell so we've got the necessary libraries there installed. Now I'm going
to import the helpers. So these helpers here, I'm just going to paste them in. These are scripts
that I've written to help us inside of the Jupyter Notebook environment to do more complicated
things easier so that we can focus and concentrate on the things that we're really interested in.
So I'm just going to import these but we do also have to create instances of each of these as well.
And so let me do that and I'll explain what each of them do so there's no hidden surprises or
anything. So first of all we've got the Lambda helper here. So what we're going to be doing in the
detail of this code is we're going to be deploying a Lambda function into the AWS account that we're
currently working on. Now there's lots of different ways to do that. You can do it through the AWS
console, you can do it through cloud formation or terraform or many, many different ways that
more than I can list here. And it's not necessarily that usual to do it from a Jupyter Notebook but
you can do it and I'm going to do it here and we're going to use this helper function that I've
written to do it. So we're going to create an instance of the Lambda helper and just to let you know
there are a couple of different methods that we can call on this. So we've got deploy function
and add Lambda trigger and we'll see what they do in more detail when we actually go ahead and use
them. So I'm just going to run that cell. So I've got myself an instance of the Lambda helper.
The next helper that we've got is the S3 helper. And so S3 is the simple storage service. It's the
most cost effective place to be able to store data. It's very, very simple. It's in the name,
right? Simple storage service. And this is just going to help us do this too. So if you read the
Boto3 documentation interacting with S3 is not that hard, but this just abstracts us away from
having to do some of this. So we've got a method to upload a file to download an object from S3
and to list objects that are in there. And again, that's just going to help us. If you are
experimenting with this in your own environment, you might choose to do this through the AWS console
or through the AWS command line. So once that's run, the last one for us to just create an instance of
is this one, which is the display helper. So this is a set of helper functions that I've created
to help us to be able to display large text documents inside of a Jupyter notebook environment.
And there's a couple of different methods that we can call. We've got text file and JSON file.
I think in this we're only actually going to call JSON file. We used text file earlier on in this
course. You may remember. Okay, so I've got all the helper functions set up. They're just sort of
now out the way. These are extra tools that we can use just to make it easier for us when we're
deploying our code. And the last thing that we need to do is just go and grab ourselves the name of
the S3 bucket, which has been created for us as part of this lab environment. So if I just
paste in this line here, we're just using the operating system library from Python. We're getting
an environment variable called learner S3 bucket name text. So it doesn't really matter what it's
called, but it's grabbing the name of a bucket, which has been already created for us in this lab
environment. And each bucket is going to be called something slightly different. So if you do see
this bucket name in my output, it will be different to yours because S3 buckets need to have a
globally unique name across all AWS accounts. So no two buckets will have the same name. And so this
just is a way that we can set that up for this lab environment. So once that's set up, we've got
basically everything that we need to get started. And we can deploy our Lambda function that can
integrate with Amazon bedrock. And you're going to construct the prompt in the way that you did
earlier in the course, which makes it easier to swap in new prompt templates when you're deploying
this into the production environment. So what I'm going to do is I'm going to create a text file
here. And we can do that inside of the Jupyter notebook environment. We're going to use this
sort of magic syntax of a couple of different percent signs. Then we're going to say right file
and then the name of a file. And now anything that we type in this cell, when we run this cell,
it's going to create that text file in the operating system in the file system of this Jupyter
notebook environment. So by running this cell, it's not going to run the code as such. And it's not
any code anyway. It's just some text, but it's going to save this text out to the disk.
So let's paste in my prompt that I'm suggesting that we use. And of course, you get to change this
if you want an experiment around with this. But let's use this for the moment. So this is the text
prompt that I want to use for the summarization of the transcript between the in the conversation
that we have from our audio file. So I need to summarize a conversation. The transcript of the
conversation is between the data XML tags. We've seen things like this before. But this is a special
piece of syntax here. This is the ginger templating syntax. This basically is a placeholder
for where the transcript will go. And then we have more information about what we want from the
output. And here we have an example output. So we're saying this is what we want to see as the
output. We want to see some JSON format. And we've set it up like we might have specified
by our application development team or if it's ourselves by what we want to see coming out of this
analysis of the phone call. And so we want an overall sentiment, sort of statement about sentiment
of the overall conversation. And then we want a list of any of the issues which have been detected
in there. And so post processing, whatever is going to pick this information up after we've
finished processing the data, can summarize or can extract this knowledge out of the phone
conversation. And of course, if we're doing that at scale across many different phone conversations
from a call center, then this becomes really valuable analytical insight as to what's happening
in those phone calls. And so we've got that. We're also going to add in something here. We're going
to say that we're interested in issues which align with a certain list of topics. And so these
topics again are something that we can actually insert into the template. So we've got this iterative
for loop written in Python. The percent symbols here are special syntax that the ginger library
recognizes so that the Python code will loop through the variable topic, which is itself a list
and each item of the list will get put into the final string of the prompt for topic in topics.
So when we pass this to ginger, we pass a list of topics and it will render each of those in this
sort of bullet list inside of this section of the prompt. And then it's going to say here,
write the JSON output and nothing more. Here is the JSON output. So this is prompting techniques to
be able to get the large language model to output the kind of text that we want. So I'm going to suggest
that that's the template we're going to use for now. And we're going to provide this to the Lambda
function and package it up with the Lambda function and deploy it into the serverless compute
environment into the Lambda environment in AWS. So for now, I'm going to run that cell. And so now
I've run that cell. We've now got this prompt template dot TXT file saved on the disk of this
Jupiter notebook environment. In fact, we could go ahead and actually print this out to the screen.
Why not? Let's go and use a display helper function that we defined a little bit earlier. So let's
say display helper. And then we can use the text file method on there. And then we're just going
to ask it to print out our file that we've just created just to prove the fact that it has actually
just created it for us. So here it's saying this is the file and here's all the data. So exactly
as you would imagine, but it just shows that we've actually just created that file and saved it to
disk. If you run this lab again, this lesson again, try running that beforehand. It won't work
because the file hasn't been created. So trust me, yes, we've just created this file. There is
our prompt template. Okay, now let's go ahead and create the Lambda function itself. Of course,
this is the main part of this lesson is looking at Lambda functions and how they work. Now we're
going to be writing some code here. But again, we're not going to be executing this code inside of
the Jupiter notebook environment. We're going to be defining it here just like we did with the
text file and then we're going to upload the code and we're going to execute it inside of AWS
in the serverless compute environment, which is Lambda. So we're going to do the same thing as we
did before and we're going to basically write a text file, but this text file will actually be a
Python file, a Python function that we can upload. So the first part that I've gotten here, I'm just
going to paste this in again here because I think it's useful and of course you'll see it as well
in the code that's been supplied. It's just a reminder that this isn't going to run here. This is
just a text file that we're creating. All right, let's get into the architecture then of the Lambda
function itself. Now the Lambda function is essentially yes, it's a Python file and the Lambda
environment, the execution environment will look for a specific function name to call. Now you can
change this, but very often, not quite default, but very often that's called the Lambda handler.
So we create a function definition inside of our file, call the Lambda handler and this is our
entry point. This is what will be called. Okay, so before we get too deep into that, let's include
some of the imports that we need. So we're going to need, oops, we're going to need to import
Boto3 because again, this code will execute somewhere else. So we need to in that environment include
Boto3. We're going to need to import JSON and we're also going to need to import the ginger to
templating library. As we saw before, we're using ginger to template. So let's import that now
so that we've got all of that set up. The next thing that I'm going to do as well, outside of the
actual function handler itself is set up an s3 client so I can get access to objects which are inside
of s3 again from this function. So let's do that. I'm going to call it an s3 client. That will then
be available sort of globally for everything that runs inside of this Lambda function and we're going
to call the client and we're going to build it up just as we've done many times before and there's
our Boto3 client for s3 and then we're also going to need our bedrock runtime as well. So I'll paste
that in because we've typed it many times before, but there's our bedrock runtime. It's the Boto3
client for bedrock runtime in the particular region that we're working in. So now we've got all
those set up. Then we can start to look at making use out of those. So let's fill out the rest of
the Lambda handler. Now if you do accidentally run this cell, it's fine. You can just re-run it again.
So I'll do it right now. So I just press run. It's written Lambda function. In fact, it's already
it's overriding Lambda function because I've already done this mistake several times of running this
cell. That's fine. We can just keep adding text to it and when we're ready, we can just finish it
there and we don't have to write anymore into it. All right. So let's put some content into here.
Now when the Lambda handler gets called from the execution environment, from the Lambda environment,
it gets passed a couple of different variables here, one called event and one called context. And we
rarely do anything with context. It's a vent that we're interested in because event carries with it
the payload, which is basically what caused this Lambda function to run. And there are many reasons
why Lambda function might get called. You might call it directly and to say I want it to run,
passing it some variables. And if you do so, those variables will arrive through event. But it might
be that it's an API gateway, which is being extended to be able to work with a Lambda function. Or
it might be that an event has occurred somewhere else inside of AWS, such as an object or a file
being put into s3. And that's what we're going to be working with here. So when this Lambda function
is called, it's going to be as a result of a file being put into s3. You'll hear me say file and object
sometimes interchangeably. When a file is uploaded into an s3 bucket, it becomes an object. So data
that's inside of an s3 bucket, that's called an object. So we're going to have an event triggered
when the object has been placed inside of s3. And we're going to get those details inside of this
event variable right here. So we can use that and extract out of that the information that we're
interested in. And we're going to be interested in knowing which bucket was it. So where was this
data put? And we're also going to be interested in knowing the name of it, which is the key of the
object, the file name, if you want, or the key of the object, which is inside of s3. So we can go and
grab both of those things. So let's first of all go and grab the bucket. And it's going to be inside
of event. An event is essentially a dictionary. So we can look up inside of there. We can go and
look at the records. And of course, if you want to, you can experiment around with this and just
maybe even print the events out so that you can get a sense of what the event looks like. But I'm
going to explain to you exactly where these values will be stored. So inside of event inside of
records, there's a list. We get the first one in the list because we're only going to be triggered
by one thing. And then we look inside of a key called s3. And then we look inside of bucket. And
then we grab from that the name. And of course, there's lots of other information inside of that
structure that you might want to get as well. But that's how to get the bucket name out of the event.
The other thing that we want to get is the key. And in a very similar way, we can get the key out
of that event out of the records out of the same part, the same list in s3. But this time we're going
to get the object key. And remember the item that's inside of s3 is called an object and the key is
the name. So if the file is in the root of the s3 bucket, then that's just going to be literally
just the file name of that object. Okay, so we've got now access to the location of the object that
which has been placed inside of s3, which caused this lambda function to run and to be executed
rather. So now we can go ahead and actually open up that file, take out the contents of that file
and use that for our generation. So let's add in some more code. You'll notice actually on the code
that's been supplied, the stuff that we're going through, I've got some big blocks of code here.
There's a lot of code here. So I will paste some of it in sort of directly and we'll talk about
what it does and why it's been included. And this is the first part that I'll include. And it's
got a comment in there actually, which also explains what this is for. This is essentially just a
small piece of error checking really to make sure that the object that we're responding to is the
object that we actually want to respond to. And so as you'll see later on in the architecture that
we put together over the next couple of lessons, the file that we're going to respond to has hyphen
transcript.json. It's going to have something before that, which is going to be a unique name.
And if it doesn't have that, if the file name doesn't end in that, then I don't want to run because
there's challenges with that, right? We don't want to get given like an image file or an audio file.
We just need to get rid of that. Okay, we're not working with that. So that's just a quick check
to make sure that we're doing the right thing. And so once that's done, we can now go ahead and grab
the actual content of the transcription.json file. And you'll notice that it is a JSON file. It's not a
text file. This is going to be the raw output from the transcription service from Amazon Transcribe.
So previously we've worked with that and we sort of saw that working at a high level.
Now we're going to get a little bit more in depth and we'll see what that looks like in a moment
or two. So let's go and set up a try block. So we've got some error handling here. And we're going
to, first of all, create a location for where the file content is going to live. So because we are
basically want to load this in as a string. So let's set up somewhere that it can be. And then
we're going to use the S3 client that we set up before to grab the data out of that S3 object.
So I can call that much in the same way as you've seen other things being called from Boto3.
So we've got a response that we're setting up and we're basically saying with our client,
we want to get object. What object do we want to get? We want to get the object, which is at this
bucket location and this key location. It's as easy as that. And once we do that, we get this
response object back and we're going to unpack that in much the same way as we've done with other
things, for example, with Amazon bedrock. So let me grab what that looks like and I'll paste that in
trying to get my indentation right. And so the file content, which is of course what we were setting
up before is inside of the response. It's at this point body. We're going to read it and we're
going to decode it. So in a similar kind of way to when we were working with Amazon bedrock and
we had to read the streaming body object response that we got back after a generation. So once we've
done that, now we've got the file content. So we've got the content of this JSON file now in
string format ready for us to do something with it. Now if you remember when we were at our earlier
lesson and we were working with the transcribed service, it actually responded back with a JSON
object, which is this JSON object. And that was quite let's say complex. It has a lot of data inside
of it with all of the different detections. It's got a just a block where we've got the basic
transcription. And then it's got each individual word broken out with who actually said it.
And we needed to scan through all of that file, pass through all of that file and create the
transcript, which was more human readable, I'll say, but it's readable for the large language model.
So we need to do that here again. Now I don't want to do that inside the lambda handler. I would
like to try and keep things a little bit organized. So let's go and create a quick function and this
function is going to do that manipulation for us. And it's going to use exactly the same code
that we looked at in the earlier lesson when we were dealing with the response from transcribed.
So for that reason, I'm just going to paste it in here. We will talk about it, but I'm not going
to build it up slowly. So I've just pasted it in here. Now I'm calling this function extract,
transcript from Textract. We didn't have that function defined before, but we did have this code.
So this function is going to accept some file content, which is going to be this file content in a
moment. And let's make a little bit of bigger gap there so we can see the difference. So this
function will take that raw JSON string in. And the first thing that it'll do is it'll use JSON
loads to actually load the content from that file into a Python object that we can then actually
do something with. And so that's going to be called transcript JSON. So from there, this is exactly
from the previous lesson where we basically iterate through that particular JSON file. We're looking
at the structure of it and we're constructing a line by line transcript of first person said this,
second person said this, first person said this, second person said this all the way through to
the end. And it's going to output here, this output text. That's obviously a string. And it's
going to pass that back obviously to whoever called extract transcript from Textract. I guess this
is abstract transcript from Textract formatted file, but this name of this function is already
getting pretty long. Now we've got that in our file. We can go and call that and bring the response
back so that we've now got a pure transcript that we can send off to our large language model.
So let's call this. So I'm going to paste this in. So this is now our transcript. So we're going
through the processing. We've started off with it raw. And then we've gone and done this conversion.
And so now our transcript text, it's calling that function with the original file contents. And
now we have a string of that transcript, which is great. Now I'm keen on having some kind of
logging. This is something that you can check and have a look at inside of CloudWatch logs.
Just as we did before, when we were looking at CloudWatch logs with Amazon Transcribe, you can also
do that with the running Lambda function. And so having an occasional logging line can be super
useful. Maybe we also want to print pass in here or log in here. What the transcript actually is.
So let's log that out as well. And if we wanted to, we could have a look at that. Now, of course,
you need to think carefully about what you log and what you don't log and have maybe a more
comprehensive logging policy than just printing everything out because it could be confidential
information in there that you don't want to have in the logs. But for now, for this particular demo,
I think this is going to work fine. Okay, so now we've got that set up. We need to send that off
to Amazon Bedrock to do that summarization of the text. And again, I'm going to create a separate
function for that. So that's an interesting distinction here. I keep talking about how a Lambda
function actually can contain multiple functions. But there's just going to be one entry point.
So we have this. This is our entry point. And it can go ahead and call multiple other functions if
you want to. So let's define another function. And for this function, I'm going to call it
Bedrock summarization. And the code that we're going to put in here will look very similar to
code that we've seen multiple times before inside of this course. And it's going to use the
ginger template. So the majority of this code really is going to be about sort of hydrating.
I think that's the terminology putting our data into that ginger template. So let's go ahead and
do that. This is pretty simple. You will see different ways that you could definitely
extend on this and make it better. But first of all, we just need to open up the text file
that's being created. So it's going to open up prompt template. Now, for the keen IUD amongst you,
obviously, this isn't running on this server. As we've said a few times, this is going to be running
inside of the Lambda service in the serverless environment up inside of AWS. So when we send our
Lambda function, we also need to package up at the same time, the prompt template and send them
both. And we absolutely will do that. And so when we do that, when the Lambda function runs,
prompt template.txt will be saved on disk or in the file system right next to this Lambda function.
So it's fine to do this. And we can just open up this text file and go ahead and grab ourselves
the template string. So that's going to be a text string holding that template as we've defined
before. So then we need to hydrate it. We need to put the data into the template. So to do that,
I'm going to create a dictionary called data. And then I'm going to load this up with the data
that's required by the template. Now, if we scroll back, I'm just going to scroll quickly. You
don't necessarily have to do this. I'm scroll back to look at the actual template that we created.
There are a couple of variables. We have transcript that needs to be inserted into the template. And
we have topics. And inside of those topics, they're going to be a list of individual topics. So we
need topics and transcript. That's what we need to pass in. And that's what we need our Lambda function
to pass in on our behalf. So let's set up transcript. And into transcript, of course, we're going to
pass in the string, which has been sent into this function. So that's super easy. And then we need
to pass in some topics. And this is really just a sample. So let's go and write a list of topics
directly into here. And so I've experimented around with these to see what works kind of well.
So let me paste in this list and put my colon back in there. So charges, location, availability,
these are the topics that we're interested in having an itemized breakdown of what's happened
inside of the transcript. And so you could put more in here. You could have less in here. Or you could
have a different way of setting up your template if you wanted to. So that's our data that we have
ready to insert into our template. And so then the next set of lines, it's really just about the
ginger process. And so this is using the ginger templating library. We have to set up this template
object. So template here is the template object from ginger. So if you scroll back up for a second
through a half completed code, you'll see here that we're calling this this template object here.
And so we're creating a version of that. We're passing in the template string as it wants in
its creator. And so we'll have this template to ginger template object here. And then we can
use that to create our prompt. So we can type in prompt and we can call template, which is this
variable here, this object here. And we're going to render it. That's the terminology from ginger,
hydrate, render. And we're going to pass the data in. So that's all we need to do. And by the end
of this, this prompt variable here will contain a text string. And that text string will be our
prompt template, which has been populated with all of our data. Now obviously that's a very simple
version of this. You can see how you could use ginger to do something more comprehensive,
potentially, and even doing things like version control of prompt templates and all kinds of
things like that, which can be super useful, especially in this sort of new era of large language
models where the prompts are really important. And they contain essentially business logic inside
of those prompts. So being out of version control, those is really handy. Okay, so we've got our
prompt set up now. Again, we can do some quick logging. If we want, let's print that prompt out.
We probably won't see that this time around. But again, if we needed to debug, then it's going to
now be inside of our cloud watch logs. And now it's just a case of sending this prompt off to Amazon
bedrock to the Titan text express model is the one we're going to use here and go and grab our
generation back. And so as we've seen many times before, we're going to set ourselves up. Yeah,
some keyword arguments. So let's say that these are our keyword arguments here. And it'll look very
similar to many times that we've used this before. But let's just cover it one more time. We're calling
the particular model that we want. And then the key things in here are that the prompt gets passed
into this text input. And then we've got our generation configuration options here with maximum
tokens, stop sequences, temperature and top P. Notice how the temperature is really low here. So
we're not necessarily interested in the model being particularly creative setting temperature to zero
will make the model more consistent, meaning that for the same transcript and for the same set of
topics, you'd expect the model to generate nearly the same response. So again, you can experiment around
with these parameters. You could expose them out to environment variables. There's lots of things
you could do. But for now, I think these variables are reasonably set. So then we just go ahead and
call our response back from the model, which we've done many times before. And we can go ahead and
return the summary that we get. And so let me again save you having to watch me write it one more
time. Here is our summary where we're going to do some JSON loads. We're doing it all in one line
here from our response. We're going into the body. We're reading. We're getting results. We're getting
the output text and we're returning that in the string here summary. Okay, so that is the bedrock
summarization function that we've added inside of our overall lambda function. So we're making great
progress here. There's not much more to go with this lambda function. And I do appreciate that we
are building up a whole bunch of code. And we're not actually running it here. So it's just a way
for us to step through it step by step. And you can see what's happening. All right, let's go back
to our lambda handler where essentially following the stages of the execution of this function,
we're almost there. We now have the ability to do our summarization using Amazon bedrock. So we
just have to go and call for it. So let's go and grab a summary back from bedrock summarization
where we passed in transcript. And that's going to do all of those things for us. It's going to
populate the template and then actually do the generation and then pass it back in summary. So
this should be at this point, the JSON return that we want to see that we've asked to have
returned from our generation from Amazon bedrock. And what do we want to do with the output? So
where do we want to put that? Now, we could just print that out. But if we did that, it would just
end up inside of cloud watch logs. And that's not very useful. Nothing that's actually called the
service will actually be able to see that output. So, and you know, a reasonably common way of
outputting from a pipeline function like this is to save the output to s3. And so you'll notice that
with this we put an object into s3 and then we process it and we save an object back out to s3.
s3 really can be used as this sort of staging point for data manipulation like this in an event
driven architecture. So let's go and do that. We've got an s3 client that we've already created
and we can go and use that client again to upload the object back into s3. And so we can do that.
I'm going to paste that in here. So this is s3 client. We're using put object and we're going to put
it into the bucket. We're going to write a specific key here, of course, because we don't want to
overwrite the JSON file. And we're going to call this results dot txt. Obviously that's reasonably
hard coded and probably would need to be extended. But for now, it will be fine. In the body, we're
going to put the summary. So that's the actual data itself goes into the body. And it's a plain
text file. So we've got a mind type there when we call the s3 client put object. So that line
on its own, of course, it's multi line, but that single command there will upload our result
into our s3 bucket. And so it's going to put it back into the s3 bucket that we originally got
something from. For this example, since you're looking for the transcript file and also writing
the summary to the same bucket, if you don't have some code to check what type of file is being
added to the bucket, then the Lambda function may get stuck in an infinite loop. What I mean is when
the Lambda function writes its summary text file to the bucket, this will trigger the Lambda function
to run again. If you don't check that the file name and the file type is not what you're looking for,
then the Lambda function may attempt to summarize its previous summary. So to avoid this loop,
we can add this statement and make sure that the Lambda function only proceeds when a new
transcript.json file is added to the bucket and not when the summary is or anything else is
uploaded to the bucket. Okay, let's finish off this tri block that we've got here. I'm just going
to do that. It's boilerplate Python really at this point. So we're going to have an exception. We're
going to log the exception if it does occur. Otherwise, we're going to return with a status code
200. This return block here is what you would see if you were calling this Lambda function.
Now as it happens, it's going to get called by the event that happens on the s3 bucket. But if you
were to extend this and run this in a different way, then you would get this output coming back,
saying hopefully everything's run successfully. And so at this point, the code should be complete.
We have our whole Lambda function there. And at this point, all we've done again is we've just
populated this cell. But if we just review through all of that and make sure it looks fine,
you can just take the code that's already there and just run it. You don't have to build it up
step by step. It's up to you. But I'm going to press shift and enter on here. And that should
now write this Lambda function out to the disk. And obviously it's overwritten it because,
you know, I've already I've already done that once. So now our Lambda function has been written.
The next part of this is now super easy because we've got all of those helper functions. So we're
going to use the helper functions to actually upload this as a Lambda function and start to
actually see it working before we move to the next lesson where we tie all of it together into a
fully automated pipeline. So to use the Lambda helper, let's go ahead and call the Lambda helper
and deploy function. Do have to spell it correctly. And the things that we pass into here,
well, we're going to pass in everything that we want to have sent to the Lambda service.
And so we have a list of files. And so we have the Lambda function.py. Let's go and break this
out over a couple of lines so that we can actually see what we're doing. So the first thing that we've
got here is a list. And this is just a list of all of the files that we want to upload and actually
zips these up actually. So it takes the Lambda function itself and our prompt template it zips it
up and sends that to the Lambda service. And we also need to have in here the name of the function.
So this just makes the code a little bit more portable. So we've got a function name and we're
going to call this the Lambda function summarize because that's what this Lambda function does.
Okay, so that's literally all we need. And if we run that so shift enter, you'll see it's
sort of going through the various stages. So it's zipping up the function. But what it does is actually
looks to see if the function already exists because it's compatible essentially with you re-uploading
and doing it again. So if you want to experiment with the code alter the code then you can do and then
you can just run this code again. And rather than creating a new function it'll just update the code
from the function that's already been created. So it'll do all that kind of stuff for us. So in this
particular case it didn't exist because we've not run it before. So it goes ahead and creates it
and it says done. So thumbs up everything's looking great. Now the next thing that we have to do with
this point we have a Lambda function inside of AWS. It's sat there. It's ready to be run. It needs to
be told to be run. Now we've written the code in such a way that it will work when it's been triggered
by this object being passed to S3. But we actually have to set that trigger up. Now that the function
exists we need to tell the Lambda function to be triggered whenever something is uploaded into
that S3 bucket. Now it's an S3 bucket which is already being created for us inside of this lab
environment. And if I scroll back up seems like quite a long way now we have this this this bucket
here this bucket name text this is the actual bucket that we actually want to put the trigger on.
So whenever the object is put into this bucket we want to trigger our code and we can do that again
with the Lambda helper. So if I just call Lambda helper and this time we're going to add the Lambda
trigger and we pass it in the name of the bucket. And just before we run that don't run it yet
just just before we run that I also would need to do to set something else up for the Lambda trigger.
I just want to set up this this is the filter rules suffix. So remember before I'm setting how we
have multiple safeguards to make sure that the trigger only works under certain circumstances. So
to prevent this trigger from working when a text file is uploaded as this Lambda function actually
will do to itself we're setting the trigger to only work with JSON files. So that's going to pass
configuration parameter into this Lambda trigger and now we can run it. Okay so again this is just
the helper function doing this for us and once we've done that we get back to here it says trigger
has been added so everything's successful there. Now if you do have an error message after you've
run this it's possible that you run this too quickly after deploying the function it does need
just literally half a second in between deploying the function and deploying the trigger maybe
a couple of seconds. So just just wait just a beat just before running all of these cells and
you should find it's fine you can just rerun it a number of times and it will it'll work eventually.
Okay so now we've got our trigger set up so now we're ready we're actually done if we upload a JSON
object into that bucket it will actually go through and it'll do the summarization for us. Now
realize that we're actually uploading the result here that transcribe will make for us so we've only
done half of it and we will complete this in the next lesson but let's prove that what we've done
here does actually work and so to do that we've got a demo transcript sat on the file system here
let me actually just use the display helper again this time using JSON file which will hopefully
nicely print out this data to the screen for us. Now this is enormous by the way so we're not
going to scroll through all of this and in fact what I'll probably do is I will comment out this
line and run it again so that we sort of get rid of it and get more space back I just want to show
you that this is here so if I just run this yeah this oh it's put it into this sort of sliding box
here so you can see this is the demo transcript or JSON or the job name of when it was originally done
this is an example that I ran on my own machine a while ago and we can use this just to show if
you like that the Lambda function works and so you can see all of the bits and pieces in here and
this goes on forever and ever and ever and ever and there's all the different parts of the speech
and who said it and the start time and the end time and this is essentially what we're passing through
with that code that we looked at before to extract out something more readable which has actually got
the you know speaker one speaker two speaker one speaker two I just wanted to show you that it was
there I'm going to comment that out and run it's just a useful way of getting rid of all that
output so what we need to do is we need to upload that JSON file into our text bucket that we had
and we can use our handy little helper functions to do that we're just going to call upload file we're
going to tell it the bucket that we want to put it in and we're going to tell it the file which is
local to us that we want to upload so let's go and run that and it says that it's been uploaded and
there's a nice big typo in there we'll fix that up at some point let's then use the next
helper from the s3 helper which is to list objects so if I just run that and then we can see here
we've got what we have just uploaded their demo transcript and we can also see already results.txt
and the reason why that's there with a timestamp which is very close by you notice it took three
seconds I'm sorry six seconds so this results.txt is the result from our lambda function
hooray it worked okay so our lambda function has been uploaded and it's actually been triggered
and it's actually worked and so essentially this is the end of the pipeline again in the next
lesson we're going to come back and put in place the beginning of the pipeline and you'll see
everything working back to back or all the way through is that's what I mean to say so let's go
and grab that so if I just run the s3 helper again we're going to download the object again if
you were doing this inside of maybe your own account you might just go to the s3 console and just
download the object but this is going to be download results.txt which is this object there and it's
going to make it into a file when it downloads it yep bucket downloaded it was successful that's
all good so now it's just a case of showing that file and yeah let's use a helper again we could
write something but let's use display helper so display helper and this is what this is a text file
and we just pass in results.txt so wait a minute what are we expecting to see here we've passed
in the transcription data from the audio transcription and we've asked it to produce a json output
highlighting anything that might be of concern or of interest inside of the phone call so now let's
look at what we got and here we go so we've got a nicely formatted json structure we've got a
sentiment which is positive and it says yeah there's this topic of charges the preauthorization
is a standard procedure to cover any incidental expenses you may carry on your stay however
it is only a hold and not an actual charge so it's picking that out of the actual conversation of
an issue of a topic of charges and if you were to experiment around with the prompt and with the
different topics that you want out you could get it to do different kinds of things of course
there are other models that you can use with amazon bedrock as well but that's a really good example
of how we can create the end part anyway of this pipeline that we're going to put together
in the next lesson we'll complete all of this now and we'll add in the automated transcription
component and see it working end to end.


## 6. Event-driven generation
Now that you've deployed your code into a Lambda function,
in order to automatically trigger it whenever an audio file is uploaded,
you can set up an event-driven architecture.
So let's do that. So you're doing really well,
congratulations in getting this far. Let's complete this code off and
include the beginning of our pipeline so that we've got something end to end.
Now inside of this lesson environment, you probably don't have any longer
the Lambda functions we deployed from the last lesson. So go back and do it all
over again. Now I'm only kidding, I'm kidding. We don't have to do it again,
already deployed inside of this environment to the same functions that we did before
with the template and all that kind of stuff. So you can just carry on from where we left off,
all that stuff is there for us. So let's rapidly get into this and get this up and running and working.
And the first block of code is just to import the libraries much in the same way as we did before.
So we've got bow to three and OS so that we can get hold of our environment variables.
Then we've got a Lambda helpers, which we used so much in the last lesson and we are going to
grab ourselves a couple of different environment variables for the same reason as last time.
All S3 buckets need to have unique names globally. So mine will be a different name to yours.
And this just allows us to be able to create these buckets inside of the lab environment and then
have a standard name for them that we can use a reference. So let's go ahead and run that cell.
And once we've got that, well, we're back now into deploying Lambda functions.
And the reason for that is that we need a Lambda function to be triggered when the audio file is
uploaded. So if we look at this from an architectural perspective, we're going to have our audio file
and the system that creates those or in this particular case, it's going to be us is going to
upload that into an initial audio S3 bucket. And so that S3 bucket is where those audio files
and then when that happens, when the object gets placed into the bucket, it's going to cause
an event to trigger and that event will call a Lambda function. Now it's exactly the same
architecture if you like as the last lesson, but this time it's going to call a Lambda function,
which is going to find the location of that object in the same way that we did before. And this
time it's going to pass it to the Amazon transcribe service. We did this by ourselves by hand if you
like inside of the Jupiter environment. Now we're going to get a Lambda function to do that for us
so that it's fully automated and we don't have to have obviously running code set in front of us.
It's serverless. It's all happening for us and it can run at scale inside of the AWS cloud
environment. So that's what happened so that that Lambda function does that. Now what it's going to do
is it's going to provide the configuration for transcribe to say once you've done that, once you've
finished, put the transcription Jason file in this next bucket, which is the text bucket. And from
that point, it's going to get picked up by what we deployed in the last lesson. So we're putting
on the beginning of our pipeline here and we are splitting this across multiple S3 buckets. In this
case, two different S3 buckets, one which contains our audio files and the other which contains our
text files. Yep, we need to write another Lambda function. And so we're going to use the same method
that we did before where we're going to basically create this text file inside of this cell. And
this time, as this code is a lot less to specifically to do with generative AI, let's paste the
function in and let's just step through it. So again, I've just included this section at the top
so that whenever it's seen, yes, this isn't running inside of this notebook, we're creating the
file inside of this notebook. So we import when it's being run inside of the Lambda environment,
we import Jason, both three UUID and OS. So OS again, because we want to have access to
environment variables inside of the Lambda environment, UUID, because we want to be able to make a
unique name for the transcription job as we did before. We're going to create an S3 client,
and this time we're going to create a transcribe client. So there's no Amazon bedrock here.
It's using the Amazon transcribe service, which is a pre-trained AI model for doing the transcription
job for us. And so with those clients created, we're now going to go ahead and create our Lambda
handler. And if you remember, this is the entry point for this code. When it's run inside of the
Lambda environment, this is what's going to get called. And as well, if you remember, it's the
event, which is going to contain the information about the event that caused the Lambda function to
be run. And so we can go into that, we can inspect that, we can grab out of there the information
about the object, which has been placed into S3. So we can grab the bucket, so the bucket name.
And of course, that means that this Lambda function could be triggered by objects being
uploaded to different buckets, like it would be, it would work. It's not just tied to a single bucket.
Anyway, so we get the bucket in this particular case and the key or the file name in this particular
case. And again, we're only going to work here with dialog.mp3. When we could change that and say
if it's only MP3 files, et cetera, we're only looking for dialog.mp3, which is the audio file,
which we've been playing around with earlier in this course. So it's the same one. Of course,
you could use a different one. And so with that, we have our object as long as it is dialog.mp3,
then we're happy to move forward. And then we can call our transcription job. And we've seen
this code run before in a previous lesson, but now it's wrapped up inside of our Lambda function.
So we create a unique transcription job name using the UUID, because it needs to be unique for
the transcription service. And then we call start transcription job using the transcribe client.
And we pass into that the name of the transcription job that we've just created where the media is.
So we know where the media is. It's in this bucket location here. Notice that we're not downloading it,
like we did before. We downloaded the JSON object and manipulated it. We don't need to do that here.
We just need to know where it is so that we can tell transcription to where it is, the transcribe
service where it is. And so that's what we do here. It's in this bucket location. It's in this
key. We give it the media format. We give it some help about which language it's got to look at.
And then here's the key part. So this is the output bucket name. And we're going to pass in an
environment variable. So this is actually an environment variable inside of the Lambda environment.
So this is an environment variable we need to set up alongside our Lambda function. So we'll use
that with the Lambda helper will help us do that. And then we have an output key. And so this
will hopefully will be familiar from the last lesson and from the code that we deployed in our last
Lambda function. We're going to output the transcription as the job name that we defined with the UID.
And then hyphen transcript.json so that it will pass the necessary filters and checks inside of
the other code that we created. And we're going to ask it to show the speakers and we're going to
look for two different speakers inside of this conversation. And then we're going to have some
error exception handling kind of stuff down the bottom as is reasonably good practice.
I'm not claiming my code is the best code ever but it's not too bad. So let's go ahead and shift
enter on that. So we've run that and we've run that Lambda function. And now we can get into
deploying this code using the Lambda helper of course. Let's go ahead and do that. Let's step through
it line at a time. And again, the Lambda helper is going to abstract us away from some of the
complexity but what it's doing is important to know. So let's paste this line in here.
So this is the Lambda helper which is actually going to create some Lambda environment variables.
It doesn't actually create them at this point but it sets the values of them for when
the function gets created. And so we're going to set something called bucket name text and then
we're going to pass in the value that we've got loaded into our environment. And if I scroll up
then you can see of course that's the same as this. So we're creating an environment variable
inside of the Lambda environment which is called this and it's going to have the name of our bucket.
And so it's just a way of passing that information along. Rather than having to pass it into the
event somehow, we can define this as an environment variable for our Lambda function. So once that's
set, let's paste in the next line as well and run these both at the same time. Let me just paste
that in there. Okay, so this is again the deploy function and we only have one file to pass this time.
So it's a list of just one. We don't have any prompts or anything here this time and then we're
going to give it a name Lambda function transcribe and that's that. So let's just run that shift
enter and yep, the Lambda function does not exist. It's fine if it does because you're just
re-running it again. It will work just fine and it's just deployed our Lambda function for us.
So it's just taken the code that we wrote above, packaged it up, sent it to Lambda and it's now
sat there waiting to run. And exactly as we did in the last one, it's now needs to be triggered.
So we're going to trigger it. We're going to set up our filter. We're going to only run this trigger
if it detects an MP3 file because that's the only time we want it to run. We don't need it to run
if it sees a JSON file, a text file or anything else. So a file with at least the extension MP3 and
then we are going to actually go ahead and create that trigger then. And so here we go. Lambda helper,
add Lambda trigger and we're going to add the trigger on the bucket, the audio bucket and we're
going to trigger our Lambda function transcribe function which of course is what we've just defined
above. So let's go ahead and run that and it's worked successfully again as per the last one.
If you race through this and then run this cell above and then immediately run this cell,
you might find that it's got an error. So just give it a few beats and then run this and it should
be fine. Okay, so now we are set up with our architecture. So we have everything set up. We've got our
audio bucket which has got a trigger on it where that trigger is fired. It's going to cause our Lambda
function to run that Lambda function is going to call the transcribe service and the transcribe service.
It's going to do its thing. It's going to listen to the audio file and do the transcription into
that JSON format which it will dump into our text bucket. The fact that that has been dropped into
that text bucket will cause the event to run which causes our summarization function to run which
uses Amazon bedrock. So it will go and look at the transcript JSON data. It will create a text
version of that output. It will then put it into the prompt. It will then send it to the large
language model and it will then get its response and it will put that response into the S3 bucket.
So enough talking. Let's see it actually working. So we have our MP3 file set on the file system
here along with us. So we're going to use S3 helper to upload it to the audio bucket that we have.
So let's run that and hopefully it comes back with something looking successful again with
our typo. We'll change that. So there the file has been uploaded. So it's uploaded that successfully.
So we can go and check to make sure that the files there are going to take a look. So let's use
our list objects on the audio bucket and indeed it's there. So there's our dialog.mp3 and it's
just been uploaded. Now that's all that's going to happen inside this bucket. Nothing else will
happen here. What's happened is it's triggered that lambda function. And so the next thing we should
see is inside of the text bucket. That's the next evidence that we'll see that it's working.
So let's call list objects on the text bucket. Right. So there are a few things in here. We've got
right access check file temp. This is actually something that transcribe creates just to make sure.
Before it starts to do the transcription, does it have the right access to be able to write to
this bucket? It did. So it moved on to the next part. Now what can we see here? We've got this
transcription job with this UID name here. Transcript.json. So that is the transcript that got created
by the lambda function that we've just created earlier in this lesson. So that's successfully
worked. And then we can also see that just a few seconds later it has finished and created results.txt
which is the output from our summarization. It's the output from our generative AI. So let's go
ahead and take a look at that. I can call my s3 helper and I can say that I want to download
an object. We need to get it from where do we need to get it from? We need to get it from the
text bucket. And we want to call what do we want to get? We want to get results.txt. So let's place that
in here. Let's run that. Excellent. That's downloaded. And we can use the display helper to show
that. But we didn't actually load in the display helper. So let's just do that quickly here.
So let's load display helper. And we need to get ourselves an instance of display helper.
So that's going to be a display helper there. Right now we can use display helper. Obviously
you could just download this file as well. So display helper. It's text file, isn't it? That's
what we have to call. And we're going to call text file on results.txt which should be downloaded
into our file space. Okay. And there we have it. Very similar output of course to the last time
that we ran this in the previous lesson. But this time we've actually seen it all the way through
end to end working from the actual original input which just up here just a moment ago
was actually the MP3 file. So we've gone in this pipeline all the way through from our MP3 audio
recording. It's been transcribed and we've taken it all the way through to our summarization
the end. Now of course this architecture is there to be manipulated and changed and you could do
all kinds of things with it. You could apply it to your own particular kind of business problem
that you have. And there's many different ways that you can extend this much in the same way.
As there is many different things you can do with Amazon Bedrock. We've really just scratched
the service here. And so here is a set of resources you can connect to to be able to see some of
the latest things you can do with Amazon Bedrock. Okay. We've made it to the end. So let's go to
the next video.


## 7. Conclusion
In this course, you've seen how to use Amazon Bedrock with a simple and powerful
event-driven pipeline to use LLMs to process the recordings of customer interactions.
This architecture can be extended and adapted in many ways to work with your specific requirements.
And what we've seen so far only scratches the surface of what you can do with Amazon Bedrock.
You can also deploy more sophisticated architectures such as Agents. Agents allow LLM-powered
applications to interact with external systems such as your other applications and third-party
services. You could extend this architecture from this course so that the LLM-powered agent
could automatically update customer records in a CRM or send an escalation email to management.
Amazon Bedrock enables you to build Agents in Lambda functions just like the ones we saw in this
course so that the agent workflows are all hosted in the cloud and work at scale.
Amazon Bedrock also supports retrieval augmented generation or RAG. Using a service known as
knowledge base, you can set up RAG in just a few clicks. You can integrate customer support
documentation into the application, enabling LLMs to find relevant and up-to-date information
quickly. To find out more about these architectures and Amazon Bedrock in general,
take a look at the resources at the end of the previous lesson. So it just remains for me to thank
everyone who is involved in putting this course together, the awesome team at Amazon,
here at deeplearning.ai, and of course you for spending your time with me. I can't wait to see
what you build with this technology and Amazon Bedrock, so please reach out and let me know.
Thank you and goodbye.
Great. Are you sure you don't want some tea?
Go on then.
You have some too.
