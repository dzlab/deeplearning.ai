# Open Source Models with Hugging Face

## 01. Introduction
Welcome to this short course, Open Source Models with Hug&Face, built in partnership with Hug&Face.
Thanks to Open Source software, if you want to build an AI application, you might be able to
grab an image recognition component here, an automatic speech recognition model there,
and an LM somewhere else, and then string them together very quickly to build a new application.
Hug&Face has been transformative for the AI community in terms of making it easy for anyone
to do this by making many, many open source models easily accessible, and this has been a huge
accelerator for how many people built AI applications. In this course, you learn directly from the Hug&Face
team how to do this and build cool applications yourself possibly faster than you might have
previously imagined would be possible. For example, you use models to perform automatic speech
recognition or ASR to transcribe speech into text, and then also text to speech models or TTS
to go the other way to convert text into audio. These models combined with an LM give you the
building blocks you can use to build your own voice assistant. You also see how to use Hug&Face's
transformer library to quickly pre-process as well as post-process outputs of machine learning models.
For example, pre-processing audio, like controlling the audio sampling rate in the ASR or TTS
examples I just mentioned, as well as pre-process or post-process data such as images and text.
The notion of grabbing open source components to build something quickly has been a paradigm
shift in how AI applications are built, and in this course you get a feel for how to do this
yourself. Delighted to introduce our instructors for this course, Eunice Baikodar,
a machine learning engineer at Hug&Face has been involved in the open source team where he works
at the intersection of many open source tools developed by Hug&Face, such as transformers,
perhaps efficient fine tuning of PEPFT, and TRL, which stands for transformers reinforcement learning.
Markson, also a machine learning engineer at Hug&Face, is part of the open source team where he
contributes to libraries such as the transformers library and the accelerate library. Maria Hallusover
is a member of technical staff at Hug&Face, and she leads the educational projects at Hug&Face
and contributes to cross-library efforts to make save the art machine learning more accessible
to everyone. Thanks, Andrew. We're excited to work with you and your team on this.
First, you will create your own chatbot with open source LLM. You will use open source LLM from
meta. The same code can apply to more powerful open source LLM's when you have access to more
powerful hardware. You will use open source models to translate text from one language to another,
summarize documents, and calculate sentence and bearings in order to compare similarity between
two sentences. Next, you'll use transformers for processing audio. What audio tasks do you think
a voice assistant might be performing when you ask it for, say, a weather forecast? It knows to wake
up when you say its name. That's classification. It converts your speech to text to look up your
request. That's automatic speech recognition. And it replies to you. That's text to speech.
In this course, you'll classify arbitrary sounds, transcribe speech recordings, and generate
speech from text. The computer vision applications of transformers are plentiful. You learn how to
detect objects in images and segment images into regions called semantic areas. For example,
you can apply this code to detect that a puppy exists in an image and also segment the part of the
puppy that makes up its ears. After you've learned to handle text audio and image tasks,
you can combine these models in the sequence to handle more complex sounds. For example,
if you want your app to help someone with a visual impairment by describing an image to them,
how could you implement that? In this course, you apply object detection to identify the objects,
image classification to describe those objects in text, and then speech generation to
narrate the names of those objects. You also use a model that can take in more than one day to
type as input. These are called multimodal models. For example, you build a visual question on
string application in which you can send an image to a model, as well as a question about that
image, and your application can then return an answer to that question based on the image.
You also use the Gradio library to deploy an AI application to hugging face spaces so that anyone
can use your application to perform tasks by making API calls through the Internet.
Of course, the goal of all of these examples isn't just for you to build these specific examples,
it's so that you learn about all these building blocks so that you really combine them yourself
into your own unique applications. Many people have worked to create this course. I'd like to thank
on the hugging face site, the entire hugging face team for the review of the course content,
as well as the hugging face community for the contributions to the open source models.
From deep learning to AI, at issue, it also contributed to this course.
In the first lesson, you learn how to navigate thousands of models on the hugging face hub
to find the right one for your task and how to use the pipeline object from the Transformers Library
to start building your applications. That sounds super exciting. Let's go into the next video and get started.

## 02. Selecting models
Today, thousands of open source models are available to you,
and many are released every week on the Hagen-Face Hub.
Hagen-Face Hub is an open platform that hosts models,
datasets, and machine learning demos that are called Hagen-Face Spaces.
How do you find a model that you need for your project?
Let's head over to the Hagen-Face Hub to find out.
You'll find models suitable for many tasks on models page.
The number of models here may seem overwhelming.
It's a good idea to begin your search by identifying what task you're working on
in machine learning terms.
In this course, you'll see plenty examples of tasks.
Let's say I want to do automatic speech recognition.
Let's choose it from the left side panel.
There are still many models to pick from,
but you can narrow your search down further.
Let's say you want a model to transcribe speech in French.
You can choose your language here.
And let's say you want a model with a permissive license.
By permissive, I mean a license that allows you to use the model
for most kinds of applications, including commercial use.
This leaves you with much fewer options.
You can sort by downloads.
If you want to find models that are commonly used for this task,
or if you'd like to try a recent model that the community is excited about,
sort by trending.
Before picking one model or the other, check out their model cards.
A well-written model card is like a readme file for a model.
It contains a lot of useful information,
such as models architecture, how it was trained,
what limitations it has, and so on.
As you can see here, models can have checkpoints with very number of parameters.
So we say that this type of model comes in different sizes.
Checkpoint refers to the saved model, including the pre-trained weights,
and all the necessary configurations.
We often say we load a model, but technically speaking,
we load a model checkpoint.
Some checkpoints have dozens of millions of parameters.
Others have a billion or a few billions of parameters.
Depending on your hardware, you may not able to run the largest checkpoints.
So let me show you a rule of thumb that I use to estimate how much memory I will need for a model.
We'll go to files and versions.
Here, you can find a file called PyTorch model bin.
This file stores the trained weights of the model,
and you can easily see its size.
Multiply that size by 1.2, in other words, at 20% on top.
And this is approximately how much memory you'll need to run this model.
Now, let me quickly show you an alternative way to find a model for a task,
a dataset, or a demo.
Let's go to the tasks page.
On this page, you can learn about different machine learning tasks.
Let's choose a task that we're interested in.
Again, let's go with automatic speech recognition.
On this page, you can learn about the task itself.
So this is a great way to discover machine learning tasks
that you have not worked with yet.
You can also find suggestions for models that will work with this task.
Data sets, you can use,
and find some demos where you can play with models that perform this task.
Note that Whispering by OpenAI is suggested as the top choice here.
Let's go back to this model's page.
To load this model from the Hagen-Face Hump,
you can use the Transformers library.
Notice the Using Transformers button.
If you click it, you'll find two helpful code snippets
showing how to load the model checkpoint.
In this course, you'll be working with models using the pipeline object
as in the first example.
The pipeline object offers a high-level abstraction to solve tasks.
It also takes care of complex pre-processing of inputs
to match the model's expectations.
For example, some audio models expect the input audio
to come in the shape of a log-mell spectrogram.
Text typically needs to be converted into so-called tokens,
and images often need to be properly resized and normalized.
With the pipeline, you won't need to do any of this pre-processing steps by hand.
Now that you know how to find models for your tasks,
and where to find the pipeline code snippet,
let's build your first application.
Let's go on to the next lesson.

## 03. Natural Language Processing (NLP)
Hi, in this lesson and the next few lessons, you will use the HikingFace Transformers library
to perform various natural language processing tasks. In this lesson, you will build your own
chatbot using an open source model built by Meta. Let's get started. We talked about natural
language processing, but what is the definition? NLP is a field of linguistic and machine learning,
and it is focused on everything related to human language. We saw significant progress in this
field thanks to the transformer architecture from the well-known Peter attention is all you need
in 2017, and since then this architecture is now the core of many state of the old machine learning
models nowadays. In this lesson, we will be using the transformer library and in particular the pipeline
function. So as you can see, I just imported the pipeline function from the transformer library.
For this classroom, the libraries have already been installed for you. If you are running this
on your own machine, you can install the transformer's library where running the following. We don't
need to actually run this command, so we can come in and out. Now we have everything we need
to create our own chatbot. As you can see, we created the conversation on pipeline using a model
from Facebook called Blenderbox. We decided to use this model because the model is very small
and performs quite well. We only need 1.6 gigabytes to load it. We can't use them to any other
big models since we won't be able to load it. As you can see, we just created a conversational pipeline
using a model from Facebook. This function pipeline takes as the first argument the task. As you can
see, we've put conversational. For the second argument, we need to pass the model and we selected
a model from Facebook. We'll go through why we decided to use this model later on, but note that
one of the main reason is because the model is very small, only needing 1.6 gigabytes. We can't use
any other big models such as LAMATU because it will exceed the four gigabytes that's available
in this classroom. Now that the chatbot is loaded, let's pass a user message.
I will ask the chatbot what are some fun activities I can do in the winter. To pass the user message
inside the chatbot, we need first to put it into a conversation object. So let's import this object.
You just need to pass the user message inside the conversation object and you need everything now
to pass the conversation to the chatbot.
As you can see, the assistant responded. I like snowboarding and skiing. What do you like to do
in winter? Feel free to change the user message asking, for example, ideas about a birthday,
a thing to do during the summer. Anything you ever wanted to ask a chatbot.
Now let's step back and review what other NLP tasks you can perform with Open Source model.
You will also see how you can search the Hagenface Hub for a suitable model for various tasks.
NLP has many applications. You must have heard about ChatGPT, the OpenAI chatbot,
or the Open Source equivalent Hagenchat, which lets the user select any open source language model.
Such as Lama2 or Mistral and Converse with it. NLP is also present in many tools we use every day.
We can think about other completion in document, translation tool, or even spam filters.
NLP has many tasks. You will go through a few of them in the next few lessons,
but feel free to try the rest on your own. You will go through the text generation
a task to build a chatbot, sentence similarity, summarization, and machine translation.
One question remains, which model should we choose? There are so many open source model.
To browse the model, you can use the Hagenface Hub. I will show you how I decided to select the
Facebook model for our conversational pipeline. We go to the Hagenface Hub on the model section.
We can add filters, so we will choose the conversational task.
Right here, conversational task is the task of generating conversational text that is relevant,
coherent, and knowledgeable, given a prompt. This model has application in chatbots and
as a part of voice assistance. As you can see right now, the model
are sorted by trend. We can change that to the most likes. As you can see, we have so many
different models. Since in the classroom, we can only fit four gigabytes. We need to choose
a smaller model. The second model seems to be a great choice because it's quite small,
since it only has 400 million parameters. We see that it was normally a lot of times,
and it has a lot of likes, so let's check this one. We can go to the files. You can see that the
model is only 730 megabytes. Now we can just click using transformers to learn how to load it,
and this comment should be pretty familiar with you. As you can see, he's telling us to use the
pipeline object to use the conversational task and load the model by putting the name of the model.
Here. For example, let's ask the chatbot, what else do you recommend?
As you can see, the assistant answer has nothing to do with winter activities. It doesn't remember
the earlier conversation. The right way to do that is to add a message to the conversation.
L&M's don't actually keep a memory of your previous messages, but when you're using
transformal object, you can add follow-up messages and the conversation object will keep your
previous prompt as well as the L&M's response so that the L&M's will converse with you as if it
remembers your earlier conversation. Let's try that. As you can see, we've added a new input to
our conversation. What else do you recommend to the earlier conversation? Now let's pass this
conversation to our chatbot. As you can see, the assistant's answer shows some memory of the
previous conversation about winter activities. The response may not be very insightful because the
model is very small. There are many states of the open source model that can do quite well if you
have the hardware to run them. Let's take a look at some of these on HagenFace app. At HagenFace,
we have the open L&M leader rules that enables users, companies to evaluate open source L&M's
and chatbot. Let's see what are the most performing chatbot. We will only select the pre-trained one.
First, the pre-trained model or model that are trained from scratch. Most of the models that you
will see at the top of the leader world are coming from large companies since not everyone has the
compute capability to train those big models. As you can see, you can see some familiar names with,
for example, the new mixture models, the Falcon model, the E model, and the Q and model.
There's a lot of debates surrounding this leader world. People are skeptical if the benchmark
we choose or representative of the performance of the models. We know that they are not perfect,
but we will be adding more and more benchmarks so that we are able to evaluate fairly all the models.
But if you want to have a look at another leader board, you can take a look at the LIMSIS
chatbot arena. This leader board collected over 200,000 human preferences vote to rank L&M's
with the LO ranking system. We have proprietary models such as GPT-4, a board or a cloud,
but as you can see in this list, we also have open source model, which are quite a catching,
quite fast, the proprietary models. As you can see here in the list, we have the
mixed row model, which have a patch 2.0 license. What it means is that it allows you to use,
modify and distribute the models. You can commercialize it, use it everywhere. In the next lesson,
we'll learn how to summarize long document and translate text from English to French. Let's go
on to the next lesson.


## 04. Translation and Summarization
Hi, in this lesson you will be translating text from English to French and summarizing long documents and using open source models from Meta.
Let's get started.
For this classroom, the libraries have already been installed for you.
If you're running this on your own machine, you can install the Transformers library by running the following.
Peep install Transformers.
Similarly, you can install PyTorch by running the following.
Peep install Torch.
Since the libraries are already installed in this classroom, I wonder when these comments.
Let's import the pipeline function from Transformers and also import Torch.
Now that the pipeline is imported, we can create the translator pipeline by typing.
Translator equals to pipeline.
We need to set the task to be equal to translation.
And we need to set the model.
For this particular task, we will select a model from Meta.
This model from Meta and LLB stands for no language left behind.
It is capable of translating 200 different languages.
The last argument we will add is the Torch D-Type.
We will set it to Torch Bifload 16.
By setting the Torch D-Type to Bifload 16, we are able to compress the model without any performance.
Now that the translator is loaded, we can start to translate the text.
So the text that we will be translating is the following.
To translate the text, you can pass the text to the translator.
You also can set the source language and the target language just like that.
The source language is English, so you need to pass the following string.
As for the target language, we said that we will be translating the text to French, so we need to pass this string.
For each language, you have a specific string that you need to pass to the translator.
You can find the code for each language at the following link.
Let's check the translated text.
As you can see, the text has been successfully translated into French.
If you know a little bit about French, you can see that Papier was correctly translated to Rochio.
But there are still a few mistakes that were made.
As you can see, Lama was translated as Lama, which is wrong.
It should have been just Lama.
Now, I invite you to stop the video and try yourself with your own text or by changing the target language.
Before moving to the summarization task, let's free some memory by deleting the model and calling the Corvich Collector.
To do that, we need to import G.C.
We need to delete the translator.
And we need to call G.C.Collect.
Summarization is also very simple to perform using pipeline.
You just need to set task as summarization.
Let's do that.
To perform the summarization, we will use a model from Meta called Bord Large CNN.
Lastly, we will also set the TorchDType to Bflow 16 to compress the model.
Now that the summarizer pipeline is loaded, let's summarize the following text.
As you can see, the following text is a description about Paris.
And let's put that into our pipeline.
To get the summary, you just need to pass to the summarizer tool that we just loaded the text.
You can also set the minimum length and the maximum length of the summary just by setting the minimum length and max length arguments.
So in our example, we set it the minimum length to 10 and the maximum length to 100.
Let's see what we get.
And as you can see, we indeed have the summary of the text we passed earlier.
Now I encourage you to pause the video here and try to summarize some of the piece of text that you find on the internet or maybe even summarize an email.
In the next lesson, you will measure the similarity between any two sentences.
And this task is very useful for many applications that involve searching for relevant pieces of text.
Let's go on to the next lesson.

## 05. Sentence Embeddings
Hi, in this lesson, you will measure sentence similarity using the sentence transformers library.
Sentence similarity measure how close two pieces of text are, for instance,
the phrase I like it ends and we love cats have similar meaning.
This sentence similarity is particularly useful for information retrieval and clustering or grouping.
Let's get started.
For this classroom, the libraries have already been installed for you.
If you are running this on your own machine, you can install sentence transformers library
by running the following, pip install sentence transformers.
Since the library are already being installed in this classroom, we don't need to run this
cell. So I will just comment it out. Let's load from sentence transformers library,
the sentence transformer class. This class will enable you to load many models.
To load the model, you just need to pass the name of the model inside the sentence transformer class.
For this lesson, we decided to use this particular model to perform sentence similarity.
This model was actually functioned by the open source community. Back in 2021,
Hagenface and Google organized an event called Community Week where anyone can join online,
use the provided hardware and train MLP and computer vision model.
The Mini-LM sentence embedding model was one of the models that came out of the event.
Sentence similarity models convert input text into vectors or so-called embeddings.
These embedding captures semantic information. Let's encode the following sentences.
The cat sit outside, a man is playing guitar and the movie are awesome. To encode these sentences,
we will use the encode methods. So for these sentences, we can get the embedding this way.
Model.encode. We put the sentence C's
and we can also add an argument to make sure that we get tensors at the end
through the arguments convert to tensor equals true.
Now let's print the embeddings. As you can see, we manage to encode the text to an embeddings
or vectors. Let's do the same thing with another list of sentences.
So we have the dot plays in the garden, a woman watches TV, and the new movie is so great.
We do the exact same thing and we get the embeddings to for the sentences to.
And now we can calculate how close these sentences are between them.
To do that, we will use the cosine distance, which is a measure to calculate how close and how
four two vectors are. To do that, we need to import the utils function from the sentence
transformers library. Then all we need to do is to use the cosine methods from the utils.
We can pass the two embeddings.
This will compute the cosine similarities and let's print them.
Notice that these are pairwise similarities for every sentence in the first list
to every sentence in the second list. If you look at the diagonal of the matrix,
you will get the similarities between the first sentences of both lists. The second
element of the diagonal will be the second sentences of both lists and the third element of the
diagonal will be the similarities between the third sentences of both lists. Now let's output the
score of each pair in both lists. You can see that the cat sits outside and the dot plays in the
garden, gives a score of 0.28, which suggests that there is a similarity between those two sentences.
The model probably picked the fact that cat and dogs is quite similar. For the second pair,
we have a man is playing guitar and a woman watches TV. These two sentences do not have any
similarities, hence the very low score. The last example, the movie are awesome and the new movie is
so great, shows a score of 0.65. This is indeed correct since these two sentences are very similar.
That's it for the lesson. I hope you learned a lot and feel free to try other tasks in the
list. We have text classification, zero short classification and Q&A. Let's go on to the next lesson.


## 06. Zero-Shot Audio Classification
Audio classification has many applications.
For example, you may want to identify the language someone is speaking,
or perhaps you want to know what birds are in your area.
Let's build a sound classifier.
But there's a catch.
In the traditional classification,
a model predicts a label from a predefined set of classes it was trained on.
If there are no models trained on your specific set of classes,
you would have to collect the data set and train to the model.
Here, you'll use an alternative approach that doesn't require fine tuning.
For this classroom, the libraries have already been installed for you.
If you're running this on your own machine, you can install the transformers and data sets
and other required libraries by running the following.
I will command the mouse because they're already installed here.
We're going to need a sound to classify.
So let's load an audio data set from the Hagen-Face Hub.
The ESC 50 data set is a labeled collection of five-second environmental sounds,
such as sounds made by animals and humans, nature sounds, indoor sounds,
urban noises.
You're not going to need the whole data set, so we're just going to load a few examples.
This example is labeled as dog. So it's likely a recording of a dog barking.
Let's give it a listen.
Sounds like a dog to me.
Let's build the classification pipeline.
For this kind of audio classification, you will need a pre-trained
clap model. At the moment, it is one of a kind architecture available for this task.
So you can find it on the Hagen-Face Hub by filtering models with feature extraction
multi-model task and then filtering by the name clap.
To classify your audio example, you only need the array of audio data.
However, the example has to have a sampling rate that the model expects.
Let's then back and talk about the sampling rate.
A sound wave is a continuous signal. This means it contains an infinite number of
signal values in a given time. But the audio your computer can work with is a series of
discrete values known as digital representation. To get the digital representation of a
continuous audio signal, we first capture the sound with a microphone.
Then, the analog signal is converted into an electrical signal.
Then, the electrical signal is sampled to get the digital representation.
Sampling means measuring the value of a continuous signal at fixed time steps.
As a result, the sampled wave form is discrete with a finite number of values at uniform intervals.
A very important characteristic of the digitized audio is the sampling rate.
It is the number of samples taken in one second and it is measured in hertz or killer hertz.
For example, 8 kHz is the sampling rate of audio in a telephone or a walkie-talking.
16 kHz is a sampling rate that is good enough to capture human speech without a sounding muffled.
A sampling rate of 192 kHz is something that you can expect from professional, high-definition,
audio-requering equipment. But why is the sampling rate important when working with AI models?
Consider an example. A 5-second sound and a sampling rate of 8 kHz will be represented as a
series of 40,000 signal values. The same 5-second sound sampled at 16 kHz will be represented
as a series of 80,000 signal values. And at 192 kHz, it will be represented with almost a million
values. For transformer model, these three arrays are very different. Transformer models treat input
as sequences and rely on attention mechanisms to learn audio representation.
There are trained on data sets where all examples have the same sampling rate and they do not
generalize well to other sample rates. So for a transformer model trained on 16 kHz audio,
a 5-second high-quality audio that is expressed in nearly a million values will look like a 60
second recording. Let's get back to the code. If a transformer model has been trained with
audio samples, each recorded at a sampling rate of 16 kHz, it's going to view any input
as if it was recorded at the same sampling rate. So let's take a one-second sound,
which has been recorded with 192 kHz sampling rate. How many values will the array representing
that sound have? 192 thousand values. But if we take a model that has been trained on the audio
examples with 16 kHz sampling rate, is it going to see as one second or more? Let's find out.
The model is going to see 192 thousand values and it expects that one second contains 16 thousand
values. So for this model, the recording is going to look like 12 seconds. So what if now we have a
5-second recording at high definition, meaning 192 kHz and the same model that was trained with
audio samples at 16 kHz? How long will the 5-second recording at high definition look like to this
model? We have 5 seconds times 192 thousand values per second. So this sample will be an array
of 960 thousand values. The model expects each second to contain 16 thousand values. Let's divide
the number of values that we have by 16 thousand. This way we'll see how many seconds the model will
think that this example is. So as you can see, the original sound was only five seconds, but with a
lot of samples per second. But for a model that has been trained with a lower sampling rate,
this exact audio will look like a 62nd recording. Now let's get back to our task and check what
the sampling rate or the model in this lesson expects. We can get this information from the pipeline.
So this model was trained on audio examples recorded at 48 kHz. Let's check the sampling rate of
our example. In this case, this is not a large difference in the sampling rate and the model will
likely do okay, but this is not always going to be the case as you'll see in other examples.
So let's see how you can automatically cast the whole data set to the correct sampling rate
when loading it with data sets library. Let's check the first sample again.
Now it has the same sampling rate as the model. When you load the data set this way, all of the audio
examples will have the correct sampling rate. So the audio sample is now ready for the model. However,
you also need to provide the pipeline with the candidate labels. Clap takes both audio and text
as input and computes the similarity between the two. If you pass a text input that strongly
correlates with an audio input, you'll get a high similarity score. Conversely, passing a text
input that is completely unrelated to the audio input will return a low similarity score.
So let's define some candidate labels to compare the sample with.
Pass the audio sample and the candidate labels to the pipeline and see what label is the most likely.
Now try more than two candidate labels and then try some completely unrelated labels. See if
you can gain some intuition for the limitations of this approach. Let's try some completely unrelated
labels. We'll use the same pipeline with the same audio sample. Remember that was a dog barking.
So as you can see here, the candidate labels now have nothing to do with dogs or barking. Yet
the model still tries to find the most plausible label among given options. In the next lesson,
we'll do automatic speech recognition.

## 07. Automatic Speech Recognition
automatic speech recognition ASR is a task that involves transcribing speech audio recording
into text. Think meeting nodes or automatically generated video subtitles.
For this task, you'll learn to work with the whisper mile by OpenAI.
Just as before, all the necessary libraries have been installed for you,
but if you're running this on your own machine,
you will need the same set of libraries as before, plus read your interface.
Let's load a speech dataset. This time, we'll take library speech.
It is a corpus of approximately 1,000 hours of data derived from the rated audio books.
Often times, audio datasets are very large, so it's useful to know how to load them in streaming mode.
This way, the examples will be loaded as needed, one at a time. For a streamed dataset,
you can access the examples one by one, and this is how you can take the first example.
By the way, if you want to access more than one example, let's say the first five,
you can do it with the take function.
Inside this list, you can access individual examples with their intuses.
You can pick whichever example you prefer, but for now, let's stick with the first one.
Just like before, you will only need the audio part of this example.
Let's listen to the narration.
Chapter 16.
I might have told you of the beginning of this phase on in a few lines,
but I want you to see every step I wish we can.
There are thousands of pre-trained models for automatic speech recognition available
on the Hagen phase hub. You can find them by selecting the automatic speech recognition
task. However, Whisper by OpenAI remains one of the best models for this task. Whisper was
pre-trained on a vast quantity of labeled audio transcription data, 680,000 hours to be precise.
What is more, 117,000 hours of this pre-trained data is multilingual or non-English.
This results in checkpoints that can be applied to over 96 languages.
Here, for the sake of efficiency, we will use the distilled version of the model that only works
for the English language. By distilled, I mean a smaller model that was trained using the responses
of the full Whisper model. This checkpoint is over 10 times smaller, five times faster,
and within 3% word error rate of the large model. Just as before, let's check the sampling rate
that Whisper expects. Now let's see what sampling rate our example has.
Hooray! This time they're the same. So we can pass the audio as is to the pipeline.
It worked. Now let's compare this to the transcription that came with the example.
Notice that unlike the transcription that came with the example, Whisper returns
transcription with proper capitalization and punctuation, which makes it much easier to read.
Now let's build a simple transcription demo and we'll use Gradio for this.
Let's create a transcribed speech function that will be a wrapper around our pipeline.
Next, you can create a tab interface where one tab will let record audio from the microphone,
and the other tab will let the user upload audio files.
So here's how we create the tab that will let the user record the audio from the microphone.
We need to create a Gradio interface past the transcribed speech function,
define where the audio input is going to come from, in this case it's microphone,
and what the output should look like. In this case it's a text box.
If you would like to learn more about Gradio, there is a course about Gradio on the learning app.
And we'll create the tab for uploading files in the same fashion.
Now just bring everything together and launch the demo.
Try the demo, try recording yourself, speaking into the microphone, or try uploading audio files and see
if the transcription matches what you say.
Let's test if whisper can transcribe what I am saying right now.
Ta-da, and try speaking for about a minute and see what happens to the transcription.
You may notice that this demo only transcribed part of what you were saying if you were speaking
for longer than 30 seconds. This is because whisper expects audio samples to be under 30 seconds,
and everything else will be truncated. Realistically you may want to transcribe longer recordings,
say a whole meeting. You can still do that with this pipeline, but you will need to provide a few
additional arguments. So let's illustrate this. Let's get a longer audio example first. We're
going to stop the demo otherwise it's going to prevent us from running anymore cells. To stop the
Gradio demo, click on the square icon that interrupts the kernel. We're going to be using the same
automatic speech recognition pipeline. Let's check again what the sampling rate the model expects.
Now, this is a big difference in sampling rate.
The error says that the model expects a single channel audio input and we must have more than one.
So we are probably working with the stereo audio. Stereo uses two channel audio. This helps create
a sense of space and directionality in sound, which enhances the listening experience. Stereo is great
at adding spatial component to audio. So when you're listening to music you get a better experience,
but for transformer models it's usually not needed. Most transformer models work with
mono channel audio. This is because you don't really need to have the spatial information to
identify whether the sound is of a dog barking or a cat meowing. You don't really need to know where
the speech is coming from. You just need to know what has been said. At the same time, stereo audio
has two channels so that's twice amount of data and it just increases complexity for the computations
without really providing any benefit. Let's see how we can convert this audio to mono. Let's check
the shape of the audio array. As you can see there are two channels in this audio, but we need just one.
We're going to use a library called Librosa to convert this audio array from stereo to mono.
Librosa expects the shape of the audio array to have a number of channels first and then the data.
So one more step before we can do the conversion is to transpose this array.
Let's check the shape again.
Now let's convert this audio to mono.
Now that this is done, let's listen to the example.
Hey Rob, hey Bob, I see the glory shining of their eyes.
Those words put me back in place. Good bye, Bob.
All right, let's try to pass this audio directly to the pipeline and see what happens.
Oh no, the text generated by the model does not really match the audio.
Why is that? This is because the sampling rate of the audio example does not match the sampling
rate of the model. The model tries to guess what the generation is, but fails to do so.
Let's double check this. Let's see what the sampling rate of this example is.
The sampling rate of the example is 44,100 hers and let's see what the pipeline expects.
The pipeline expects the audio to be sampled at 16 kilohertz.
Let's fix this. We can re-sample an individual file using Librosa.
Now the audio example is ready for the pipeline.
For the pipeline to be able to transcribe the longer video, we will need to pass a few arguments
to it. But first, let's talk about how automatic speech recognition pipeline handles longer
recordings. Because Whisper can only take in up to 30 seconds at the time to transcribe this
longer example, the pipeline will split the long file into chunks. We can specify the chunks
length for Whisper. 30 seconds chunks are optimal since this matches the input that the model expects.
And each segment will have a small amount of overlap with the previous one. This allows the
pipeline to accurately teach the segments back together at the boundaries, since it can find
the overlap between segments and merge the transcriptions accordingly. Because the audio
is split into chunks, the pipeline can transcribe the chunks independent of each other and then
combine the results. For this reason, you can transcribe batches of chunks in parallel.
Let's see the arguments that will be passed into the pipeline.
Next, the length for the chunks. 30 seconds in this case.
Next, we'll specify how many chunks we want processed in parallel using batch size.
In this case, the original file is only one minute and 21 seconds, so there is no need to have
a batch size larger than 3 or 4. In case of a larger file, the batch size will depend on your hardware
and memory available to you. If you try a large batch size and get an out of memory error,
you know that you need to try a smaller batch size. So in the first introduction lesson,
I gave you a roll of thumb to estimate how much memory you would need for a model to run.
To estimate the batch size, think of it as a multiplier for that memory.
So how many models you can run in parallel? So if you have the hardware to do that,
you can have a large number of batches. In this case, four managers,
but we also don't think that any more because 30 seconds splits with a bit of overlap for it is
probably even redundant. Probably we can do three, but just in case three or four for this case is fine.
Try an experiment with batch size and see what your hardware can handle.
Finally, you can set return timestamps to true, and this enables predicting segment level
timestamps for the audio data. These timestamps indicate the start and end time for a short passage
of audio, and they can be particularly useful for aligning the transcription with the input audio.
To output the transcription with the timestamps, you can print the chunks part of the output.
Now we get the transcription for the full audio, and we get the timestamps. Now let's see how
we can modify the demo to accept longer audio recordings. First, let's copy the original demo.
Let's copy the transcribed speech function. Here, we're going to modify
how we're calling the pipeline. Instead of giving it only the audio file, we'll pass additional arguments.
Like so.
The only thing that we need to update in the interfaces is the name of the function.
The code snippet launch in the demo doesn't change, so it's ready to go.
Try uploading an audio file that is longer than 30 seconds, or record yourselves
speaking into a microphone, again longer than 30 seconds, and see if it works.
In the next lesson, you'll learn how to go in the opposite direction and go from text to speech.
Let's go to the next lesson.

## 08. Text to Speech
In this final audio lesson, we'll tackle text to audio generation by converting text to speech.
Text to speech is a challenging task, because it is a one to many problem.
In classification, you have one correct label, maybe few.
In automatic speech recognition, there's one correct transcription for a given utterance.
However, there's an infinite amount of ways to say the same sentence.
Each person has a different way of speaking, but they are all valid and correct.
Think about different voices, dialects, speaking styles, and so on.
Despite these challenges, there are open source models that can handle this test really well,
and you are about to use one of them.
We'll use a VITS pre-trained model from Kakao Enterprise.
This is one of the two models that can fit in this environment.
And this model has a permissive license.
Once you have the pipeline, all you need to do is to pass some text to it.
Let's write some text.
Now let's pass this text to the pipeline.
Let's give it a listen.
Researchers at the Allen Institute for AI,
I think the face Microsoft, the University of Washington,
Carnegie Mellon University, and the Hebrew University of Jerusalem developed a tool that measures
atmospheric carbon emitted by cloud servers while training machine learning models.
After a model's size, the biggest variables were the server's location and time of day
and was active.
And just like that, you can convert text into an irradiated audio recording.
Feel free to paste your own text and play with the pipeline.
In the next lesson, Eunice will show you how to build an object detector.
Let's go on to the next lesson.

## 09. Object Detection
Hi. In this lesson, you will explore computer vision models and play with some of them to build a
cool application. In particular, you will build an assistant that can help a visually impaired
person understand what is in a picture. Let's get started. Okay, so just a quick heads up before
starting the lesson. So for this classroom, the libraries have been already installed, but if
you want to run the notebook by your own on your own machine, make sure to install the required
libraries, which can be installed through the following command. So yeah, let's get started.
So first of all, we want to import the utility methods that we're going to use for this library.
So make sure to import from helper the methods called load image from URL and render results on
image. So we're going to quickly do that before starting. So for this lesson, we're going to
focus on a specific task in computer vision called object detection. So first of all, we're going
to load a pipeline object as we've been doing it so far from our previous lessons.
So we're just going to import pipeline from transformers and load the object detection pipeline
that we called OD pipe that we will call using a model from Facebook called DTR resident 50.
So just run the following to get started to load the pipeline. And before moving forward,
I wanted to give some insights on this specific task we're going to focus on. So what is object
detection? So the task of object detection simply consists of detecting objects of interest in
a specific image. So for example, as you can see on this image, the object detection model is
able to detect all relevant objects in an image. And one thing to notice is that object detection
combines two sub tasks, which are classification, but also localization, because for each object
that we detect in an image, you also have to provide the label of the instance, but also the
localization of the detected object. You may be wondering how did we choose this DTR resident 50
model from Facebook, because today we have many state-of-the-art object detection models that you can
use from the AI ecosystem in general. So for that, you can simply browse the hugging phase hub and
use the filter object detection that you can see on the left and get all the available object
detection models that you can freely download from the hub. And you can easily determine some
metrics that you're going to use to select your models. So for example, you can select your model
based on the number of downloads or number of likes, and also sometimes the authors provide the
evolution metrics of their models on some specific datasets directly on the model cards. So those are,
I would say, the important metrics that you can use in order to select the model that you're
going to use for your task, and for this lab, we're going to use the Facebook DTR resident 50 model.
So now that we have loaded the pipeline, let's directly start using it. So let's directly use our
pipeline by loading an image that we have prepared for you. So this is a recent image that we took
altogether in the restaurant for filming our course. So yeah, we're going to see what are the
objects that the model is going to detect in this image. So I would invite you to take a few seconds
to make yourself familiar with the pipeline. Make sure to pass the image into the pipeline,
get the output from the pipeline, and you can use the render results in image function that we
have provided in the helpers in order to render the results directly in the image. Okay, so to get
the results from the pipeline, simply call OD pipe on the hero image. So we're going to do that
right now. So pipeline output equal OD pipe on the hero image. So yeah, once we got the prediction,
we'll just call render results in image by passing the image and also the pipeline output.
And let's see the final results. So yeah, as you can see, the model has accurately
predicted all the persons that are on the image. So with the corresponding mounting box,
you can also see the confidence score of the predictions for each predicted instance.
The model has also predicted the battles, the cups, and forks.
Matea looks like the model was not able to predict the file files, but maybe with some fine training,
we'll be able to do that. So yeah, now I invite you to pause the video and try that on your own,
with your own images, can be some local images that you load using pill, but you can also pass some
image URLs if you want to use an image that is on the web. Okay, so let's see how we can make that
a bit more user friendly. So let's say tomorrow, you want to show that to your friend and
make a nice demo using the model. So we're going to use a library called Gradio in order to expose
a simple interface. So that's pretty much the final demo will look like something like this. So
a simple interface where you can pass an image here, we pass a prompt, but you can also pass an
image. And then you get the results of the pipeline right next to the original image. That way,
you can share easily the demo with anyone and anyone would be able to try the model out of the
box using their own image. Let's get started. For that, we'll first import Gradio as follows.
So the Gradio interface expects to have a method where you pass the input and return the output.
So we'll create a method that will do everything under the hood for the users given an image.
So for that, we'll define a method called GetPipelinePrediction. So Def,
GetPipelinePrediction that will take a pill image as an input and we'll split it up in two steps.
So the first step will be to get the pipeline output given the pill image and we will use the
globally defined audit pipe that we have loaded beforehand so that we won't have to load the pipeline
each time we'll call this method so that it will be much more efficient. And then we will use the
helper function that we have used before, render results in image directly on the original image
using the pipeline output. And then we turn the final processed image. All right.
And actually, the code to make the Gradio interface is going to be pretty much straightforward.
So once you have that method defined, you just have to create an object of Gradio.interface
that will take as input this method, GetPipelinePrediction. And we just have to properly define
what is the input and what is the output and we'll just have to call demo.launch to launch the demo.
So if you run this code snippet, you'll be able to define the demo and we will call
demo.launch with share equal true so that you can also share a link for the demo to anyone.
Okay. So once you have the demo up, you can load any image from your local computer and pass it
to the model. So let's try out with this image and click on submit.
Yeah. So as you can see in this really good picture, the model was able to detect both cats,
the remotes, but also the couch, but we can't really see it because of this icon. But yeah,
feel free to try it out with your own local images. You can also pass, as I said, if you want to use
a raw pipeline, you can also pass a URL to the pipeline directly. And also you can share this link
to anyone so that they can try out the demo on their computer as well. You can also send this
link to your friends so that they can try out the demo on their own, but please keep in mind to
let your computer open and running because the demo will be running on your computer.
Yeah. So for the last part of this lab, we'll see how to make an AI part assistant using two
different models. So let's say you have an image, which is here. You just have to pass it to the
object detection pipeline to get all the relevant objects in the image with their labels. And then we
can perform some sort of post-processing of the output of the pipeline so that we have a more
natural text that describes what's in the image. And we can pass that image to the text to speech
model, which is also going to be another pipeline, which is going to generate an audio that will
narrate the text that we created in this step. And then we'll have the audio saying in this image,
we have this, this, and this. So this is what we're going to do right now. And yeah, let's get started.
Yeah. So we can combine the object detector model that we have just used with a text to speech
model to help us indicate and dictate what are the objects that are in an image with their number
of occurrences. So first of all, if you try to inspect the output of the pipeline, you will get
something like this. So it's an array of dictionaries with each dictionary corresponding to one
detected object with the label, the coordinate of the bounding boxes together with the confidence
core. So if we create a simple method that processes this array and returns a string of the
summary of what's in the image, we can pass that summarize string to a text to speech model to
dictate us what's in the image. So let's try that out. So we will use the same image that we use
before and the same pipeline. So we're not going to rerun everything again because we will use the
audit pipe that we have already loaded, which is here. And we will just have to import that helper
method called summarize predictions natural language from the helper's function that we have
provided to you. Let's try to see what the text would look like if you pass the pipeline output
to that method. So if we run this method and print out text, we'll get in this image, there are two
forks, three battles, two caps, four persons, one ball and one dining table, which is a quite
accurate representation on what's in the image. So you can also take your time and try to inspect what's
in this method, but all the logic behind this method is pretty straightforward. It just
tries to combine all the output that is in this array and try to make a sentence out of it.
All right, and what model are we going to use to generate or generation of the image? So we're
still going to use pipeline, but as you have seen in the previous lesson, we're going to use the
text to speech pipeline with this specific model from Kakao Enterprise. You can read more about the
model directly on the model card of this model, but yeah, we're going to use this pipeline. And let's
get the narrated text from the text to speech pipeline given the text. Perfect. Yeah, so to in
order to listen to the narrated text, we'll need to run this small snippet that uses iPad and audio.
So let's try out right now and try to listen what's in the narrated text.
In this image, there are two forks, three bubbles, two cups, four versus one bowl and one dining table.
That's quite good and quite accurate. We can wrap the whole pipeline in a single method so that you
just have to pass an image and it will return you directly the narrated audio. You can also wrap
that in a graduate interface so that you can let your friends try that out on their own.
So yeah, I'll invite you to post the video and try that out with other images, maybe also other
models that you can find on the hub so that you can compare different performances across different
models and also maybe try to wrap everything in a graduate demo so that you can share it with your
friends.

## 10. Image Segmentation
Hi! In this lesson, among other computer vision tasks,
you're going to perform segmentation and something called visual prompting.
By that, I mean that you will simply specify a point on a picture
and the segmentation model will then identify a segmented object of interest.
Let's see all that together.
Welcome everyone to the lab session of lesson 9.
We're going to see the segment anything model from Facebook AI,
now called meta AI.
And yeah, use that model to build cool applications.
First of all, make sure to run this cell before starting the lab,
so that we won't get the warnings from customers
and also make sure to install the required libraries that we provide here.
So make sure to run these comments first before running the notebook.
Yeah, so let's get started.
So yeah, as we've been doing so far for all our labs,
we're going to use the pipeline object for this lab
and we're going to focus on a task called mask generation.
So I'm going to explain in detail what mask generation means
and how does it differs from the classic image segmentation task.
So let's first import our pipeline object
and start to initialize the pipeline object.
So some for segment anything model, pipe equal pipeline, mask generation.
So in the image that you can see here,
we simply performed image segmentation on the image that you can see on the top.
The model predicts pixel wise labels for each pixel of the image
with the corresponding label of the pixel.
So for example, the blue pixels will refer to the sky.
The red pixels will refer to the class bridge
and the other pixels irrespectively refer to the ground and the mountains.
And in segmentation mask generation,
the difference is that users can perform what we call visual prompting
by guiding the model on the location of the object of interest
in order to predict the segmentation mask of that object.
So segment anything model, some from Facebook,
expects as an input 2D points, as you can see here,
but you can also provide bounding boxes as input
and the model will predict segmentation mask of that object of interest.
And in contrary to classic image segmentation,
the predicted mask won't have any label.
The only label that you can extract from that mask
is that that object corresponds to the object of interest
that you have specified to the model.
There is also one thing that you could do with segment anything
is the automatic mask generation pipeline.
So by that, we simply sample some points from the 2D image
and try out different combinations of 2D points together
because you can also prompt multiple points per mask
and filter out the predicted output with the highest scores
to get the most relevant segmentation masks in the whole image.
If you pass that image and use the automatic mask generation pipeline,
you will end up with results such as this one
where you'll have the masks for each object of interest.
So for example, you have the road, you have those small pieces
of the train, windows of the building, and things like that.
So if you want to read more about segment anything model,
you can just check, check out the paper,
some paper from Facebook AI or the original repository
of segment anything model.
And for our lab, we're going to use a distilled, compressed version of the model
called SlimSam, which basically does the same thing as Sam
with similar performances, but is much smaller.
So this model is going to be useful for us
because we're going to run our lesson on a small hardware.
So we will be able to run segment anything model
without the need of having a high compute requirement.
Now that we have more or less understood,
what do we mean by a mask generation and how it differs
from classic image segmentation pipeline?
Let's try our hands on loading the model
and try to play with the model.
Once we have instantiated the pipeline with mask generation,
we're going to pass the path to the segment anything model.
So we're going to use this model from the hub,
which corresponds to the SlimSam model that I've presented you before.
So let's load this model.
All right, now that we have the pipeline,
let's try to import an image that we have prepared for you.
So we're going to try to predict some segmentation masks
on this image where you have some people
and some cool lambas, as you can see here.
For the automatic mask generation pipeline,
you can pass this optional arguments
called points per batch
and you can get different results based on this value.
So higher points per batch means more efficient pipeline inference
for smaller hardware,
we recommend you to use a smaller points per batch
so that you won't run into any hardware issues.
So yeah, you can just run this command.
It will take some time for some computers,
but yeah, just wait for the results
and we're going to see the results together.
Okay, so now that the pipeline has finished its execution,
let's try to visualize the results.
So for that, we've prepared a helper function for you
called showpipemasks on image.
So we're going to import that and use it straight away
on our original image.
Very nice.
So as you can see, the model was able to segment
all small regions of interest in the whole picture.
So for example, it was able to segment all the heads
regions for each person,
was able to segment almost all the lambas individually themselves,
also the closes,
yeah, the wall behind and small items on the back as well.
But the problem with this pipeline is that you need to iterate over all the points
and post-process degenerated masks,
which might be a bit slow for some use cases and applications.
So we're going to focus on one specific use case
where we're going to infer the model with an image and a single point.
So let's try to do that right now.
So instead of using pipeline,
this time we're going to import the model class itself from transformers
and some processor.
So we just need to call some model that from pre-trained
to checkpoint name that we used before
and we're going to do the same thing for the processor.
We can also print out the model to see its architecture
for those who are curious.
So there is a positional emitting, vision and color,
transformer layers and so on.
And for this exercise,
we're going to use the same image
and let's say we're interested in segmenting this blue shirt from Andrew.
So for that, we're going to pass any 2D points
from the blue shirt in order to segment that region.
So we're going to give this location.
The point 00 starts from the top left.
So 1,600, 700 should be somewhere here in the shirt.
So for that, you need to first encode both the image
and the 2D points.
So we just have to pass to the processor the image.
Input points equal input points
and mention that we want to return PyTorch sensors.
And we're going to perform a simple inference
on the model with the Torch no-grad context manager
so that we make sure we don't compute the gradients.
So import Torch.
All right.
Okay. So once you have retrieved the output from the model,
we need to pass process to predicted masks
in order to resize them to the size of the original image.
So for that, you can simply call imageprocessor.postprocessmasks
in order to get all the predicted masks.
And I wanted to quickly inspect the size of the predicted masks.
So if we do land predicted masks, we have one
which corresponds to the number of images.
But if we would have passed many images,
then we would have as many masks as where we do have images.
So let's just consider the first mask and inspect its size.
For our predicted mask, we have a tensor of size one.
So batch size three and then size of the image.
So I just wanted to give a quick heads up
on why do we have three on the second dimension.
So if you check again, the overview of the sum architecture,
as you can see from this figure,
the model predicts three segmentation masks
together with their confidence scores.
So that's exactly what's happening here.
So we have all the three masks.
And we can also inspect the prediction scores
by getting outputs dot iu scores.
And as you can see,
the first mask seems to be the mask with the higher confidence.
But let's see on our case
and print all the predicted segmentation masks
given the visual prompt that we have tried.
So very nice.
So in two cases over three,
we were able to accurately predict the segmentation mask
of Andrew's shirt.
But for the first mask, it was able to segment Andrew entirely
instead of just the shirt.
But to get better results for this specific use case,
one could also pass multiple points
to get the segmentation mask,
which corresponds to the region of Andrew's shirt.
Yeah, so you could try to pass multiple points
for the same mask.
You can also try to pass a bounding box
that encapsulates the region of Andrew's shirt.
But yeah, you can try out many combinations.
Feel free to try them out and also try out the combinations
that are suggested in the official documentation
of some model in hugging faces transformers.
Now that we have seen how to use the segment anything model
in order to segment any object of interest
given an image and some 2D coordinates
and or bounding boxes,
I wanted to also present you another model
called DPT,
which stands for dense prediction transformer.
So DPT is a model that you can use
to perform death estimation given an image.
So death estimation is a common task in computer vision
that is also widely used, for example,
in autonomous driving.
So for the demo, we're going to use a model
called DPT Hybrid Midas from Intel.
And we're going to use pipeline as usual,
but this time we're going to call the death estimation task
for the pipeline.
So let's import pipeline from transformers
and define our death estimator using death estimation.
And for the model, we're going to use
DPT Hybrid from Intel.
Okay, so let's first inspect the image
that we're going to use for this demo.
So as you can see, it's a small Tamagotchi
that is standing in a road in Vienna,
apparently, according to the title.
So we are just going to estimate the death
of this image.
Let's see how it goes in terms of code.
So yeah, as we've been using pipeline,
it's pretty straightforward.
So we just have to call death estimator.
That's what we called our object of the raw image.
And then if we inspect the outputs,
we have a dictionary with the key predicted death,
with the raw death tensor of the predicted death of the image.
So we can't display that tensor as is.
So we need to first post-process the image
by first resizing it to the size of the original image.
So for that, we're going to use a function from PyTorch
called interpolate from Torch.nnd.functional.
And we're going to consider the predicted death from the output,
but we're going to unsquease in order to add a dimension
on the first axis.
If you inspect the shape of the predicted tensor,
so it has one, so batch size, number of images.
And then interpolate also expects to have the number of channels
in the second dimension.
So we're just manually going to add it here.
So just want to show you how it looks like if you call that method.
So it just adds a new axis on the second axis.
And then we're going to resize it.
So our target size is going to be the size of the raw image.
We're going to use the BQ big mode.
So those are the things you shouldn't worry too much about.
Those are just the best, I would say, best hyperparameters
that you can use to resize an image.
And align corners equal folds.
All right.
And then if we print out the prediction tensor,
so now we should have the same shape as the input image.
So that's great, but there is still one thing that we need to do.
So the values cannot be displayed as they are,
because for an RGB image, the pixel values
need to be between 0 and 255.
So we need to normalize the prediction tensor,
so that the values will stay between 0 and 255.
So we're just going to call that block.
We remove one dimension calling squeeze,
we convert it to a numby tensor,
and then we normalize it between 0 and 255.
And convert the image, the tensor in int 8,
and use that converted tensor using peel from array
to get the final def.
And let's see how it looks like.
So here is the predicted output from the model.
So as you can see, the Tamagotchi that is in front of the picture
has strong value towards white pixels.
So it's very close to the image.
And then the elements that you can see far behind
have pixel values that are close to black pixel.
So yeah, I would say the model was able to quite accurately
predict the depth of the image.
So you can try it out with your custom images
or images that you find on the internet
and just make sure to resize the output using these formulas.
So let's say now you want to showcase this model
through a simple demo,
and you want to share it with your friends or colleagues.
You can use Gradio that we used on the previous lesson
and share the link of the Gradio demo
to your colleagues or your friends.
So yeah, I just wanted to quickly show you how to do this using Gradio.
So it's pretty much straightforward.
So we're going to import Gradio.
We're going to use transformers pipeline
and we're going to use the depth estimator object
that we have defined here beforehand
so that we don't have to load it each time.
So similarly as the demo that we showed on the previous lesson,
Gradio.interface expects a method
that does everything for you
given the input that the user will pass.
So we need to define a method
that we take the input image as input
and we'll do everything for you under the hood
to the user and returns the predicted image
with the peel image format.
So we just have to write down all the steps
that we did here in a simple method
so that we do everything in one go.
So this is how it would look like.
So we get the output from the pipeline
given the input image.
We resize the prediction using the snippet we used above
and we normalize the output using also the snippet here.
And then we define the Gradio interface
that takes the method as input,
explicitly defined the input as a peel image
and the output as a peel image as well.
We're going to run that cell and call interface.launch
with the argument share equal true
so that you also have access to a shareable link
so that you can share it with anyone.
So I'm just going to quickly try it on a local image.
Very nice.
It is able to predict the depth of the image accurately.
So yeah, feel free to try that out again
with your local images or images
that you can find on the net.
You can showcase this bundle to your friends
and your colleagues.
In the next lesson you will learn with Mark
how to use multimodal models
where you can pass both image and some texts
in order for example to ask some questions
about some specific images.
So yeah, let's move on to the next lesson together with Mark.

## 11. Image Retrieval
In this lesson, you will work with multi-model models to perform image text matching
using the open source model blip from Salesforce. So if you take a picture of a woman
and a dog on the beach and also provide a text, my sister and her best friends,
the model outputs a matching score to indicate how similar the text and image are. Let's get started.
But first, what exactly are multi-model models? When a task requires a model to be able to take
as an input model one type of data, let's say an image and a sentence, we will call it multi-model.
You may come across other definition of multi-model, but we will stick to this one in this course.
When you think about multi-model models, you immediately think about chat dbt with dbt4v,
where you can send text, image and even audio. Now, if you want to try an open multi-model's chat dbt,
you should also definitely try the fix. In this and the next few lessons, you will go through some
common multi-model task. We will perform image-to-text matching, image captioning, visual Q&A,
and zero-shot image classification. For the first three tasks, we will be using the blip model
from Salesforce, and for the last task, the zero-shot image classification, we will be using the clip
model from OpenAI. The first text we will be looking into is the image, text, retrieval, or matching.
The model will output if the text matches the image. For example, you can see that in this example,
we passed a photo of a man and a dog, and the input text is the man in the blue shirt is wearing
glasses. The model should return that the text does not match the image. Let's cut it. For
this classroom, the libraries have already been installed for you. If you are running this on your
own machine, you can install the transformers library by running the following. Since in this classroom,
we have already installed all the libraries, we don't need to launch this command, so I'll just
comment it out. To perform the task, we need a few things. We need to load the model and the
processor. First, to load the model, we need to import blip for image text retrieval class from
the transformers library. Let's do that. Then to load the model, you just need to call the class
we just imported and use the front pre-trained methods to load the checkpoint.
As said before, I will be using the blip model from Salesforce to perform this task,
and this is the related checkpoint for this specific task.
As for the processor, it's practically the same. We need to import the auto-processor class from
transformers. Then to load the correct processor, we just need to use the front-read train
methods and past the related checkpoint. The processor role is to process the image and the
text for the model. Now, let's get the image and the text that we will be passing to the processor.
The processor will modify the image and the text in such a way that the model will be able to
understand it. For the image, we will be using the following URL link and to load the image, we will
be using the image class from the pill library. The pill library is installed by default when you install
Python. We also need to import the request library in order to perform HTTP requests to get
the data from the image. In order to get the raw image, we need this code. In short, this line of
code downloads an image from the specified URL, opens it, retrieves the raw binary data, then
converts it to the RGB color mode. And if you print the raw image,
you should be able to see the image. Now that we have the image, we will check if the model can
successfully return that the image matches the following text. An image of a woman and a dog
on the beach. We need to get the inputs that the model can understand. So to do that, we need to call
the processor and we need to pass a few arguments. The first one is the image. So image equals to raw
image. The second one is the text. And the last one is return tensors that we need to set it as
PT for PyTorch so that we get PyTorch tensors at the end.
Let's print inputs to see what it looks like.
And as you can see, we have a dictionary of multiple arguments. We have pixel values, inputs,
ID and the attention mask. Now we have everything to get the output. We just need to pass the inputs
that we have right here to the model. Note that we need to add a double store since we are passing
a dictionary that contains the arguments. Now let's print the scores.
As you can see, these numbers doesn't mean anything yet because they are the
logits of the model. And to convert these values into something that we can understand, we need to
pass them into a softmax layer. The output of these numbers into a softmax layer will give us
the probability. To get the softmax layer, we need to import Torch. Then we need to pass the
scores that we got into the softmax layer. Let's check what we get.
Now this number makes more sense. The first value is the probability that the image and the text
are not matched, so the probability is very low. And the second one is the probability that your
match and from the value, it shows that indeed the text and the image are matched with a high
probability. As a conclusion, we can say that the image and the text are matched with a probability
of 98%. Now is a good time to pause the video and try it on your own image and prompt.
Let's move on to the next task. Image captioning. You will use the same model that don't know different
ways that were trained specifically to take an image and output texted describe that image.
Let's do that in the next lesson.

## 12. Image Captioning
In this lesson, you will perform image captioning and you will use the same model blip,
but with different weights. Let's get started. The next task we will be looking into is image captioning.
For the image captioning task, we asked the model to return the description of the image.
For example, the model should return a man and a dog are reading a book together. We can put
the start of the output text. For example, we can let the model know that the output text should
start with a dog on a couch with something else. Let's move to the code. For this classroom,
the libraries have already been installed for you. If you're running this on your own machine,
you can install the transformers library by running the following.
Since in this classroom, we have already installed all the libraries. We don't need to run this,
so I will comment it out. To load the model for this specific task,
we need to import the blip for conditional generation from the transformers library.
Then to load the model, we will use the form pre-trained methods
and we will be using the following checkpoint.
Just like in the previous lab, we also need to import the processor.
To load the processor, it's the same as the model. We will use the form pre-trained method
and pass the right checkpoint.
Now we have all the elements to perform the image captioning. We are just missing
two small pieces, the image and the optional text. Let's get the image first. To do that,
we will use the image class from the pale library. To load the image, you can use the open
methods from the image class and pass the pass to the image. Let's do that.
Let's check that we were indeed able to load the image. And as you can see,
we have the picture of a dog and woman on the beach. Let's first perform
conditional image captioning. What it means is that we can pass a text that will be the start
of the output of the model. For example, we can pass a photograph of
then we need to process the text and the image. To do that, we will pass the text and the image
to the processor. We can also specify the return tensor's argument to be equal to pt
for PyTorch. This way we will get PyTorch tensors at the end. Let's check the inputs.
And just like in the last lab, we have a dictionary of arguments such as pixel values,
inputs id and attention mask. Now to generate the description of the image, we need to use the
generate methods. Just like in the previous lab, we also need to add the double stores since it is
a dictionary of arguments. Let's check the outputs. As you can see, we get as output a list of
integers. These numbers are token IDs. These are usually how the model understands the text.
Each token represents a part of a word or sometimes a single word. To decode these tokens,
we need to call the decode methods from the processor. Let's do that.
We put the output as the first argument. This is optional, but we can skip the spatial token
by setting that argument to true. And let's print the results.
And we get a photograph of a woman and her dog on the beach.
Now let's try the unconditional image captioning. What it means is that we don't pass any text
and we let the model start the description.
As you can see, this time we didn't put any text and let's generate the text using the generate
methods just like before. So the output is equals to model the generate double store inputs.
And let's decode. And now we get a woman sitting on the beach with her dog.
Now is a good time to stop the video and try to upload your own image and put your own
conditional text to the model. For the next lesson, we will be testing visual question and
answering. For this test, you can ask the model a question about an image and the model should
be turned and answered. Let's go to the next lesson.

## 13. Multimodal Visual Question Answering
Now, you will reuse the blip model for question-answering of an image.
This means you can give the model a picture of a dog and a woman on the beach,
and ask a question such as who is at the beach,
and the model will answer your question based on that image.
Let's do that now.
For the visual question answering task,
you can ask the question about the image to the model,
and the model should return an answer.
If you ask the model what the dog is wearing, you should get a pair of glasses.
Now, let's call this.
In this classroom, the libraries have already been installed.
If you're running this on your own machine,
you can install the Transformers library by running the following.
Since the libraries are already installed in this classroom,
I won't run this, and I'll just comment it out.
Just like the last two lessons to perform this specific task,
we need a few things, the model and the processor.
So, let's import blip for question-answering class from the Transformers library.
Now that the class is loaded, let's load the model
by using the FormPretray methods,
and by passing the related checkpoints for question-answering,
let's do the same for the processor.
So, we're loaded from the Transformers library,
the class auto-processor,
and we will load the processor using FormPretray also,
and we just need to pass the same checkpoint.
And that's it.
Now, let's load the image that we need to pass to the processor
to get the inputs, and we will pass these inputs
to the model to generate the answer.
To load an image just like in the previous lab,
we will use the pill library and the image class to do that.
To load an image, we will use the open method,
and we just need to pass the path to the image that we want to open.
Let's check the image.
And here we are.
And you can see that we need to have a picture of a dog and woman on the beach.
Now, let's ask a question to our model about this specific image.
For example, let's ask it how many dogs are in the pictures.
We need to pass the inputs to the model,
so we will use the processor to process both the image and the question.
And we will return the tensors to be PyTorch tensors.
Just like the previous lab, we will use the generate methods
from the model to get the outputs.
And we need to use the processor to decode the output.
And you can see that the model was able to,
and so correctly to the question that there is indeed only one dog in the picture.
Now, I invite you to stop the video and ask other questions about this specific picture,
or you can even upload your own pictures and ask whatever question comes to your mind.
In the next lesson, we will learn about zero-shot classification with clip model from OpenAI.
Let's move on to the next lesson.

## 14. Zero-Shot Image Classification
Now, you will use the Clip model to classify images with zero-shot classification.
It's zero-shot because the model will be able to classify the image from a
moment list of any labels that you do give it.
This is great because you don't have to fine-tune the model to recognize specific categories.
You can just use the model out of the box. Let's try it out.
For zero-shot classification tasks, you will use the Clip model from OpenAI.
Clip is a multimodal vision and language model.
The zero-shot image classification task consists of classifying an image based on your own labels during inference time.
For example, you can pass a list of labels such as playing card, dog, bird, and the image you want to classify.
The model will choose the most likely label.
In this case, it should classify it as a photo of a dog.
Yeah, the photo is a little bit small, but this is indeed a dog.
Let's see how to do that.
For this classroom, the libraries have already been installed for you.
If you're running this on your own machine, you can install the transformers library by running the following.
Clip install transformers.
Since the libraries are already installed for us in this classroom, I won't be running this command.
Just like in the previous lessons, we need two things. We need the model and the processor.
Let's first load the model from the transformers library.
So we need to load the Clip model from transformers.
To load the model, we will use the from FreeTrain method.
And we need to pass the correct checkpoint for this specific task.
Now that the model is loaded, let's load the processor.
So we will import all the processor from the transformers library.
And we will use also the from FreeTrain methods to get the correct processor.
Now that the processor and the model is loaded, let's get an image.
To do that, just like the previous lessons, we will use the PL libraries and import image class.
We will load the image using image.open.
And we will specify the path to the image.
Here's our image, two lovely kittens.
Let's try to classify this image. Let's create the labels.
So let's put a photo of a cat as the first label and the second label we can put a photo of a dog.
Then we need to create the input that will go inside our models.
To do that, we will use the processor.
We need to pass the text, which are the labels.
The image.
We will make sure that we get the pipe watch sensors,
where specifying PT in the return tensors arguments.
And we will also add the last argument padding to be equal to true,
encased the labels length or not the same.
Then we can pass this output to our model.
So outputs equals models, double store inputs.
Since the inputs is a dictionary of arguments,
and let's check what we get.
We get a very big output, but the thing that we are interested in are the logits per image.
So let's print that.
And as we've seen in the previous lessons, to get the probability,
we need to pass these logits into our softmax layers.
So let's get the probability.
Just like that by calling the softmax on the output,
let's get the softmax of the logits.
And we take the first element, which is a single tensor.
And as you can see, now we have something that looks like a probability.
The first element is the probability that the image is indeed a photo of a cat.
And the second one is the probability that the photo is a photo of a dog.
So to conclude, we see that for the label a photo of a cat,
the probability is practically 100%, whereas for the second label,
the probability is near zero.
Now I invite you to pause the video and try a couple of things.
You can either maybe change the label so that you put labels that doesn't have anything to do with the image
and try to see how the model responds to that.
You can also upload a new image or just change the labels.
In the next lesson, Eunice will show you how to deploy the blip model using hugging phase bases.

## 15. Deployment
Hi, you now have a broad overview of the type of tasks that you can achieve within the Hanging
Face ecosystem. In most cases, for hosting demos and practical applications, it would be nice
to have your application running without leaving your computer on. In other words,
offloads the whole computer requirements outside your local machine. In this lesson,
you will leverage Hanging Face spaces to deploy your demos and use them as an API. Let's get started.
So welcome to this lab session about deploying ML models in the Hanging Face ecosystem using
Hanging Face spaces. So at the end of this lab, you will learn how to host a ML model as an API
on Hanging Face Hub and call that API through a simple command. So for this lesson, we will deploy
the blip model that you have covered on the multimodal lab session. Recall that the model has been
fine tuned on several multimodal tasks, making it possible to perform three different tasks,
image captioning, visual question answering, and image text retrieval. In our case, we're going
to focus on image captioning. So let's get started. So first of all, you need to create an account
at the Hanging Face website, hf.co, and make sure to connect to the main website. Then you just have
to navigate here on the top right of the window and go to a new space to create a new space. Let's
decide for a name for the space. So we're going to call it blip image captioning
API. We're going to put the default license, select the video as the space SDK and select the
basic hardware and put it public so that everyone can use it. All right. So once you have created
that space, you need to create two files, which are the requirements.txt file to list all the
required libraries that you need to run the space and the main file that you need to call app.py.
So let's do that right now by creating first the requirements file and put all our requirements
inside that txt file. So we'll add transformers because we need transformers by torch and gradient.
All right. So before creating the app.py file, let's quickly go back to our lab and create a new cell
and first try out the demo locally before pushing it on the hub. So as seen in the previous lab,
we're going to leverage the pipeline objects from transformers and use gradient as a demo by
leveraging the gradient interface. So as usual, load the pipeline. But this time we're going to load
the image to text pipeline and we're going to load the blip image captioning based model in order
to perform image captioning. All right. So now that the model has been loaded, as we've been doing
so far for all our gradient demos, we need to define a method that will call launch here that we
take the input, call the pipeline, get the output and get the generated text from the output.
And note here, we're using the globally defined pipeline that we have loaded here so that we
won't have to instantiate a new pipeline each time we're calling launch. That way we just have
to define the initialization of pipeline once at the beginning of the script so that it's loaded
only once and we use that globally defined model or pipeline inside the launch function.
Then let's define our grader.interface that will look like this with input being a grader.image,
output being text and we call interface.launch with share equal true so that we can also have a
shareable link that you can share with your friends and colleagues. All right. Yeah, let's quickly
try it out right now. So we're going to try that out on this image that we have locally of two
kittens lying on a couch with two remotes. So let's see how it goes. Perfect. Two cats sleeping on
a couch. That's pretty much a good description of the scene. All right. So now the question is what
if I want to deploy a larger model or what if I don't want to deploy that model locally on my local
computer because I want to use my computer to do something else. This is possible using spaces and
that's what we're going to see in our lab. So once we have confirmed that the app works locally,
we're going to export that app directly into the created hugging space. So we're just going to
copy paste all these blocks and paste it in the newly created app.py file on the space that we
have created. So let's just do it right now. So we're going to create a new file app.py
and copy paste everything. Note that alternatively you can also get clone the space locally
and do everything through Git. Here we're just doing everything through the command line through the
UI for easiness. Here we're going to remove sharing control
and we'll just have to wait the app to build. Okay. So if everything went well you'll end up with
something like this. So if the app is successfully running you should have that on your
a newly created space. We can also test it out right now to see if it works.
All right so we got the same results as the test that we've been doing locally. Now how can I use
that space as an external API? So if you look into this window there is use via API select box
here that you can click and we just have to you know follow the instructions here. So first
pip install radio client. So that's that's something we already did in our lab and we just need to
retrieve this link which is a temporary link that gets changed. I think each 24 hours.
So you just have to instantiate this client object with this correct link and called client
predict then local path to the image or a URL or also a pill image object and make sure to call
API name equal predict and then you can directly print the results. So let's copy paste this snippet
and try that out in our lab. So let's try that out.
Perfect. So if we inspect the input image. So indeed it's a red bus with some blue stripes on
the side. So if you want also to further inspect what's in your API you can also call the view API
methods directly on the client to get more information on the API. So it also gives you some
information on how to call the API. So through API name equal predict the expected parameters and
also the expected return type of the API. Perfect. So we were able to call our model as an API outside
our local machine. So everything is done on the cloud. Feel free to try that space out or build a
new space for your own use case or try out new models new inputs and so on. So before moving
forward there is also another feature that I wanted to show you. So just to complete my explanation
about the client API you can also make the space private in case you want to host a private model.
We just have to make sure that the space is private when going to the setting here. So by making
the space private and then you just have to pass your hugging face token when you instantiate the
client. So in terms of code it would look something like this. So you will instantiate your client
with the argument hf token your token. So that way you'll have access to the private space as an
API. So how can we push this further? So we've been deploying this demo so far on a CPU instance
on the hugging face space. But what if I want to deploy a much larger model that cannot fit on
the CPU instance or what if the CPU instance is a bit slow and I want to deploy it on a GPU instance.
So there is a feature you can use within the hugging face ecosystem called GPU 0 space where you can
basically spin some free GPUs on demand for your spaces. So let's see how to do this. We need to
first go to this hugging face organization called 0 GPU explorers. Just have to browse 0-gpu-explorers
on hugging face hub and request to join this organization in order to get access to this feature.
Once your request has been accepted you directly have access to the 0 GPU feature. Whenever you
create a new space you'll have the option 0 GPU appearing here with dimension free.
So to demonstrate the 0 GPU feature I've created a space on my personal account for the lava
model. So lava model is an image to text model which is quite large so the lava 1.57b that is
deployed on the space the demonstration space has approximately 16 gigabytes and needs quite a
lot of GPU RAM to be run. And the space leaves here and as you can see it says running on a zero
device. So let's use that as an API. So we just have to click use via API here. Okay so we can
copy paste this snippet and try it out on our local machine. And instead of using the default prompt
and image we're going to use a custom image that we have prepared for you. So remember this image
of the instructors having dinner all together in Palo Alto here. So we're going to use this image
and ask the model if it's able to predict what we are all holding in our forks.
So let's try that out right now. So we're going to prompt the model in such a way that we're
going to explicitly ask what we are holding in our forks. So we're just going to say these
people are having dinner in a Mediterranean restaurant to hint a bit that we're having
can you guess what they are all holding in their forks. And yeah recall that you can either pass
an image URL or the path to your image. So we're going to pass the path to the image and wait for
the results. Perfect. So let's see what the model has predicted. So it's say that we're all
eating meatballs. At least the model has predicted that it's something that is round but it was not
unfortunately able to detect that it was falafel despite the hint that we gave. But yeah I guess
that's a bit challenging maybe even for humans since it's round and a bit brown. So yeah but still
I think the description is quite accurate and it's quite funny how the model behaved with respect
to the prompt and the image. All right so to wrap up this lesson I invite you to explore hugging
face spaces in order to deploy your custom demos. You can also browse the spaces webpage to check
out the spaces of the week to get some inspiration and cool applications and demos that you can build
and easily share. So I invite you to have a look at all the things that we have covered during this
whole course and come up with some cool ideas that you can publish easily on hugging face spaces
and share it with your friends and colleagues. All right so there is just one more video where we'll
say thank you and wrap up the course. So let's go on to the next lesson.

## 16. Conclusion
Congratulations on making it to the end of this short course.
In this course, you've gained insights into the selection of open source models for text,
audio, and images, all available on the hugging phase hub.
You build multiple applications such as a chatbot, a speech transcriber,
and an object detector to name a few.
Now you know how to build a functional prototype for your application using Gradio
and how to quickly deploy it with hugging phase spaces.
Moving forward, you will be able to navigate the large landscape of open source models.
Consider the pipelines you've constructed as foundational building blocks
and explore the boundless opportunities for creativity they offer.
Combine them to create new, unique applications, skies to limit.
None of this would have been possible without the collaborative spirit of the open source community.
Thousands of people, researchers, engineers, educators, and thinkers
have contributed to open source by sharing models and data set on the hugging phase hub
by opening full requests to open source libraries and by engaging in discussions that lead to new ideas.
The progress in AI we witness today all merge to the open source community.
We extend our heartfelt appreciation to all those who contribute to its progress.
We encourage you to explore other machine learning tasks and models available on the hugging phase hub.
Build innovative applications and share your creations with the community
to inspire others with what you build.
If you find this course helpful, maybe you can even share it with your friends.
