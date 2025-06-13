# Quantization Fundamentals with Hugging Face

## 01. Introduction
Welcome to this short course, Quantization Fundamentals with Hoking Face,
built in partnership with Hoking Face.
Large GNIWII models, like large language models, can be so huge
that they're hard to run on consumer-grade hardware.
Quantization has emerged as a key tool for making this possible.
In this course, you learn about a variety of flavors of quantization
and the different options and data types,
like whether you should use int8 or float16,
or something called beeflow16, which stands for brainflow16,
to compress your models.
And you also learn about the technical theory
and the algorithmic details of how to compress and store
a 32-bit floating point number,
maybe from a model that you want to deploy
using, say, an 8-bit integer.
Um, delighted to introduce our instructors for this course,
Eunice Dalgoda, is a machine learning engineer at Hoking Face.
Eunice is involved in the open-source team
where he works at the intersection of many open-source tools
developed by Hoking Face, such as transformers,
peft, and TRL.
Mark Sun is also a machine learning engineer at Hoking Face.
Mark is part of the open-source team where he contributes to libraries
such as transformers or accelerate.
Mark and Eunice are also deeply involved in quantization
in order to make large models accessible to the AI community.
Thanks, Andrew.
We are excited to work with you and your team on this.
In this course, you will first learn about basic concept
around integer and floating point representation
and how to load AI models using different data types
using PyTorch and Hoking Face Transformers library.
You will also understand the pros and cons of each different data type
in order to make the right decision for your use case.
You will also dive deep into linear quantization
by understanding how it works in practice.
You will see how linear quantization works in simple terms.
The quantization scheme is used in most state-of-the-art quantization methods.
After reviewing how linear quantization works,
you'll directly apply it into a small text generation model
using the Quanto library from Hoking Face.
Quanto makes linear quantization easy to use for any PyTorch model.
We'll first load the model using transformers library
and then use Quanto to quantize the model.
In summary, in this course,
you see in detail the fundamental theory behind quantization
as well as the practical aspects of how to use quantization.
I hope you learned about these techniques
and combined these building blocks yourself
to create some unique applications.
Many people have worked to create this course.
I'd like to thank on the Hoking Face site,
the entire Hoking Face team for the review of the course content,
as well as the Hoking Face community for the contributions
to open source models.
From deeplearning.ai,
Adishu has also contributed to this course.
This is a short course that covers a lot.
So I'm excited about what you'll be able to learn
in a compressed way.
I hope you enjoyed the course.

## 02. Handling Big Models
AI models have been getting bigger and bigger.
So quantization has been really exciting for the AI community
because it enables us to shrink models to a small size
so that anyone can run it with their own computer
with little to no performance degradation.
Let's get an overview of why we need quantization
and also what is quantization.
Let's get started.
Nowadays quantization is an exciting topic
as it enables us to shrink models to a small size
so that anyone can run it on their own computer
with no performance degradation.
So let's check everything together.
Nowadays deep learning architectures
tend to become larger and larger.
Specifically for large language models,
in just a few years,
average model sizes grew by another order of magnitude.
This clearly widens the gap between largest GPU hardwares
which at the time of speaking,
are around 80 gigabytes at most and largest models.
As you can see here from this graph,
this graph stops in 2022.
Although it seems to increase until that year.
So from 2020 to 2023,
the largest most used state-of-the-art LLMs
seem to have an average number of parameters around 70 billion.
This still creates a gap between the largest hardware
and largest models,
as a 7B model would need
approximately 280 gigabytes
just to make the model fit on the hardware.
Note also,
consumer type hardware,
such as Nvidia T4 GPUs,
have only 16 gigabytes RAM.
Therefore, running this state-of-the-art models
is still a challenge for the community.
How to run these models efficiently,
without having the need for accessing memory heavy hardware?
So the entire challenge now for the community
is to make these models more accessible
through model compression.
So we'll start quickly reviewing
some of the current state-of-the-art methods
for model compression,
such as pruning and knowledge distillation
before spending more time on quantization.
So first of all pruning,
simply consists of removing layers in a model
that do not have much importance
on the model's decisions.
It simply consists of removing some layers
based on some metrics,
such as the magnitudes of the weights
and possibly other metrics as well.
There is also another method called
knowledge distillation.
In this protocol,
you train a student model,
which is the target compressed model
using the output from the teacher model
in addition to the main loss term.
The challenge here
is that you need to make sure you have enough compute
to fit the original model
and to get the predictions from the original model
so that you can send them
to the teacher model while computing the loss.
And this can be quite costly
if you have to distill very large models.
So before diving into quantization,
recall that for a neural network,
you can represent the weights
and the activations as follows.
When quantizing a neural network,
you can either quantize the model weights,
which can be represented by the matrix W,
but you can also, sometimes,
if you want, also quantize the activations of the model,
which corresponds to the output
of the computation that you can see on the right.
Let's now go over quantization.
Quantization simply consists
of representing model weights
in a lower precision.
Let's start by considering this small matrix
on the left,
which stores some parameters of a small model.
Since the matrix is stored in float32,
which is the default storing data type for most models,
it has to allocate four bytes per parameters,
four times eight-bit precision.
Therefore, the total memory footprint
of that matrix is going to be 36 bytes.
If we quantize the weight matrix in eight-bit precision,
so int eight,
we end up allocating only one byte per parameter.
Hence, we'll need in total only nine bytes
to store the entire weight matrix.
However, this comes with a price,
which is the quantization error.
The whole challenge behind state-of-the-art quantization methods
is to lower this error as much as possible
to avoid any performance degradation.
To sum up,
let's summarize what we are going to cover in this course.
In order to give you a good basic understanding
of underlying notions around quantization,
we're first going to visit most commonly used data types
in machine learning,
such as integrals,
mainly int eight precision,
as well as understanding how the floating point representations work,
such as float16,
B float16, or float32,
and which precision to use,
depending on your use case,
meaning depending on whether you're doing model training
or model inference.
Next, we're going to deep dive into linear quantization
and see how it works.
After that, we're going to use the quantum library
from HuggingFace ecosystem
in order to quantize a transformer's model
and test results out using different configurations.
Finally, we will go over the recent advances
in quantization techniques applied to LLMs.
In the next lesson,
you will learn how to explore data types,
which are the core building blocks
of these machine learning models.
So let's go into the next lesson.

## 03. Data Types and Sizes
In this lesson, you will learn about the common data types used to store the parameters of machine learning models.
This lesson is essential to better understand quantization, since quantization is achieved by converting numerical values to a different data type.
Let's get started.
Let's start with the integer data type.
An unsigned integer data type is used to represent a positive integer.
The range of an n-bit unsigned integer is 0 to 2 to the power of n minus 1.
For example, the minimum value of an 8-bit unsigned integer is 0, and the maximum value is 255.
The computer allocates a sequence of 8 bits to store the 8-bit integer.
For an unsigned integer, the decoding process is at follows.
If the bit is equal to 0, its value is 0.
If the bit is equal to 1, the decoded value is a power of 2.
For the first bit, it is equal to 2 to the power of 0.
For the second one, it is equal to 2 to the power of 1, and so on.
In this example, as you can see, in the first position of the sequence, we have bit equals to 1.
So the decoded value is 2 to the power of 0.
For the second one, and the third one, both bits are equals to 0, so we have 0 and 0.
For the first position bit, we have 1, so the decoded value is 2 to the power of 3, and so on.
And at the end, we add those values, so we have 128 plus 8 plus 1, which equals to 137.
For a design integer data type, it is used to represent a negative or positive integer.
There are multiple representations, but the one we will look into is the 2's complement 1, since it is the most common one.
The range is minus 2 to the power of n minus 1, and 2 to the power of n minus 1 minus 1.
So for an 8 bit sign integer, the minimum value is minus 128, and the maximum value is 127.
The difference with the unsigned integer is that for the bit in the last position, as you can see here, the value is negative.
So if we process the same sequence as earlier, we need to add a minus here, and the result will be minus 128 plus 8 plus 1, which will be equal to minus 119.
This way of processing the sequence brings some questions.
Thus, the addition between two sign integers works.
Let's have a look at a quick example with 4 bits to convince ourselves.
As you can see, the first sequence represents 2, and the second represents minus 2.
The addition of these two sequences should give 0.
So let's do that. We have 0 and 0 here, so it gives 0.
We have 1 and 1, so we have 0, but 1 is carried on the left.
So we have 1, 0 and 1, so we get 0, and 1 is carried to the left.
We have 1, 0, 1, so we get 0, and 1 is carried to the left.
However, since we only store 4 bits in total, we don't save the last bit that was carried on the left.
And at the end, we end up with 0, as you can see here.
Creating data with integer data types is very easy in PyTorch.
You just need to set the correct TorchDType.
As you can see in this table, to create an 8-bit sign integer, you just need to pass the following TorchDType.
Torch.int8.
For an 8-bit unsigned integer, you just need to pass Torch.u.int8.u stands for unsigned.
In PyTorch, you can also create 16-bit sign integer, 32-bit sign integer, and 64-bit sign integer.
For example, let's check the information about the 8-bit unsigned integer.
To do that, we will use Torch.info, and we need to pass the TorchDType we want to check.
For this classroom, the libraries have already been installed for you.
If you are running this on your own machine, you can install the Torch library by running the following.
Pipe install Torch.
Since the library have already been installed in this classroom, we won't be running this command,
and I'll just comment it out.
And we can just type the command we talked earlier.
So Torch.info.
And we will pass, for example, Torch.u.int8 to check the 8-bit unsigned TorchDType.
And as you can see, we get that the minimum value is 0, and the maximum value is 255,
which makes sense with what we saw earlier.
Let's do the same for the 8-bit sign integer.
As you can see here, the command is the same, but we just need to pass this time Torch.info.
And as you can see, we get that the minimum is minus 128, and the maximum is 127, just as expected.
Great, let's move on.
Now is a good time to pause the video, and you can try other data types on your own.
For example, you can try the following data types Torch.in64, Torch.in32, and Torch.in16.
And you can check if the results you get make sense with what we saw in the theory report.
Now, let's move on to floating point representation.
There are three components in floating point representation.
We have the sign, only one bit is needed since a number can be either positive or negative.
We have the exponent which determines the range of the number, how big in magnitude it can be in both the positive and negative direction.
Lastly, we have the fraction, the fraction determines the precision of the number.
I mean, can you define a number as 0.4999 or only as 0.5?
Floating point 32, B flow 16, floating point 16, and floating point 8 are all floating point data types with a specific number of bits for the exponent and fraction.
Let's have a look at floating point 32.
Floating point 32 or FP32 in short is composed of one bit for the sign, 8 bit for the exponent, and 23 bits for the fraction.
And if you add them, we will end up with 32 bits.
Here's the range for floating point 32 for positive values.
We can represent a very small number as small as 10 to the power of minus 45, and as big as 10 to the power of 38.
As for negative values, this is the same range that with a minus in front of each value.
So the minimum value is minus 3.4 times 10 to the power of 38.
For floating points, we have two formulas to decode the sequence.
One to represent very small values, which are also called subnormal values, and the other one to represent very big values, called normal values.
If we compare it too much about these formulas down here, the point here is to see how big and how small a number you can store using floating point 32.
As you will see how this differs with other data types.
This data type is very important in machine learning since most models store their weight in floating point 32.
For floating point 16, we only have six bits for the exponent and 10 bits for the fraction.
The smallest positive value you can represent is 10 to the power of minus 8, and the biggest is 10 to the power of 4.
Compared to floating point 16, beef load 16 locates 8 bits for the exponent and 7 bits for the fraction.
As you can see in the range, you can represent very small values and very big values.
However, the downside is the precision, which is worse than floating point 16.
To sum up, FP32 has the best precision, and the range is also very big.
FP16 has a better precision than beef load 16, but the range is smaller.
Lastly, beef load 16 range is close to the range of floating point 32, but it has a worse precision.
The nice thing about floating point 16 and beef load 16 is that they take up half of the space of floating point 32.
Let's see how you can use them in PyTorch.
Here's the table with floating data types in PyTorch.
For example, as you can see, to create a 16-bit floating point, you need to set the torch d type to be equal to torch dot float 16 to create a 16-bit brain floating point.
You just need to set the torch d type to be equal to torch dot beef load 16.
In PyTorch, you can also create the 32-bit floating point and 64-bit floating point.
Now, let's see what happens when you convert a Python value to a PyTorch tensor with a specific data type.
First, we create the value 1-ser.
In PyTorch, the value we created is stored in floating point 64.
If we create a tensor with a torch d type to be equal to torch dot float 64, we shouldn't see a difference. Let's do that.
So, we first create the value 1-3rd.
Then, let's check the value.
And as you can see, the value we passed do not correspond exactly as it is.
It is converted to floating point 64.
And the value that is stored inside the computer is an approximation.
Now, let's create a tensor with a d type equals to floating point 64.
To do that, we need to call torch dot tensor.
We put the value and we need to specify the d type argument to be equal to torch dot float 64.
Let's have a look at the value.
So, as you can see, we have the same results.
Now, let's do that for other data types.
We will do that for floating point 32, floating point 16 and below 16.
And let's check the results.
So, to create the floating point 32 d type tensor, we just need to change the d type to be equal to torch dot float 32.
Then, let's do that also for floating point 16 and below 16.
Then, let's print all the results together so that we have a good comparison.
From these results, we can make the following observation.
The less bits we have, the less precise the approximation will be.
And for Bflow 16, as we said before, the precision is worse than floating point 16.
So, this is why, as you can see here, the approximation is worse than the floating point 16.
But Bflow 16 has a bigger range than floating point 16.
You can check this information directly from PyTorch using the function torch dot f info.
Let's do that for Bflow 16, for example.
Let's compare this information with floating point 32.
So, we just need to change the torch d type.
And as you can see from these results, the minimum value and the maximum value are quite closed.
But you can see that the resolution of the floating point 32 is way smaller than the Bflow 16.
Now, it's a good time to pause the video and try it by yourself by changing the torch d type to be equal to torch float 16 or torch dot float 64.
Now that we know how integer and floating point works, we can have a look at downcasting.
Downcasting happens when we convert a higher data type to a lower data type.
The value will be converted to the nearest value in the lower data type.
A floating point 32 value, for example 0.1, downcast it to 8-bit integer will be converted to 0.
You see that we have a loss of data?
Let's check the impact on matrix multiplication.
First, let's create a random tensor with torch d type float 32 of size 1000.
To create the random tensor, we will use the run function from torch, so torch dot run.
The first argument is the size of the tensor, so we will put 1000.
And we need to specify the d type of that tensor, which will be torch dot float 32.
And that's it.
Let's have a look at the tensor we just created.
Since the tensor we created is very big, we'll just look at the five first elements.
And you can see that we indeed have random values in torch dot float 32.
Now let's downcast this tensor to be float 16.
To do that, we will use the two methods.
And we just need to specify the d type to be equal to torch dot be float 16.
And now let's have a look at the first five elements.
And as you can see, we managed to downcast our tensor.
We see that the d type is now equals to torch dot be float 16.
And since we downcast the tensor, we do not have the same values as the original one, but they are very close.
Now, let's check the impact of downcasting on multiplication.
First, let's do the multiplication with the original tensor in flowing point 32.
To do that, we will use the dot methods from torch.
So multiplication with float 32 equals to torch dot dot.
And we put as arguments the two tensors in floating point 32.
Let's check the results.
And the results we get is this one.
But the result you will get will be different since we initialize random tensors.
Now let's check the multiplication on the be float tensors.
And as you can see, the result is quite close to the original one, but still we do have a loss of precision.
The advantages of downcasting is reduced memory footprint.
We have a more efficient use of GPU memories.
It enables the training of larger models and also enables larger batch sizes.
But also we have an increased compute and speed.
Computation using low precision, for example, floating point 16 and be float 16 can be faster than floating point 32.
Since it requires less memory and it also depends on the hardware.
Whether you're using Google GPU or NVIDIA 100.
The disadvantage is that it is less precise.
We are using less memory, hence the computation is less precise.
One of the use case of downcasting is mixed precision training.
We do the computation of these models in smaller precision, for example, floating point 16, be float 16.
But we store and update the weights in higher precision.
Usually it is floating point 32.
Now let's move on to the next lesson.
Eunice will present you how to load models with different data types.

## 04. Loading Models by data type
In this lesson, you will load your machine learning models using different data types such as
float 16 or B float 16 and study their impact on the model's performance. Let's get started.
Welcome to this new lap session. In this lap, you will put into practice things that you have
learned in the first lap session. Specifically, you will see how to load some ML models in different
data types such as float 32, float 16 or B float 16. You will also learn how to load popular
generative AI models in different decisions and study their impact of floating these models in
health precision on their performance. And you will also learn how to load any model inside your
workflow with your desired health precision data type out of the box. So at the beginning of our
lesson, we will try to inspect the data type of a model. So what do we mean exactly by the data
type of a model itself? So recall, for a ML model, each model's layer contains some weights that
are going to be used for inference, meaning when you get the model's prediction. And each weight
is stored usually as a matrix of learnable parameters, which can be represented in different
precision. So for example, in this dummy model, we have let's say 12 layers and each weight has
end parameters that each of them are stored in 32-bit precision. Therefore, inspecting the
model's data type is equivalent to inspecting the data type of a model's weights. So let's get
started. So we'll first try to inspect the model's data type using a dummy model. So for that,
we've prepared a dummy architecture. So we'll just import it from the helper methods.
And then we'll load the model as follows and print it.
As you can see here, this is a small model that has a token embedding layer, a linear layer,
a layer norm layer, another linear layer, and a last layer norm layer, and a language model head.
So this is equivalent of having a very small and dummy language model. So we said we wanted to check
the data type of the model. Okay, so for that, we'll just use an utility method from PyTorch called
named parameters that you can call on a module to loop into each module's parameter and its name.
So we'll just loop into the name modules as follows.
And simply print the name of the module together with the d-type of the module.
Print
and to access to a data type of a parameter, you just have to call param.dtype.
All right, so let's test the method out right now.
Perfect. So as you can see, we're able to print the model weights name together with the d-type
of each weights. And as you can see, all model weights are loaded in Flow32, which is the
default for PyTorch. So let's see now how we can cast the model into different precision,
such as Flow16 or B Flow16. So yeah, so in order to cast any PyTorch module into,
let's say, Flow16 or B Flow16, the API is pretty much straightforward. So let's say your
target d-type is Flow16. You simply have to call one of these two methods. So either model.2,
your d-type, or model.half for Flow16, or model.b Flow16 for B Flow16. So let's try that
away. And let's print also the d-types of each model's parameter using the method we've defined
before. So we're going to call print param d-type on model FP16. And as you can see, all the
models weights have been converted successfully to Flow16. All right, great. Let's say now I
want to use the model and just perform a simple inference on the model. So recall, I'm using a CPU
instance here. So we're going to see if this is possible to use it out of the box. So
let's define a dummy input. So here we're using, so a long tensor corresponds to the IDs of the
tokens that you're going to pass to your transformers model. Because recall the architecture of the
model, which is a transformers-like model. And it has an embedding layer as the first layer of
the model. So the embedding layer expects to have a long tensor as input. And the embedding
layer will output hidden states in floating point precision. So either FP32, if your model is in
Flow32, or in Flow16, if your model is loaded in Flow16. All right, so we have our input.
Let's first try to do an inference with the Flow32 model and see that it works.
Perfect. We can also print the final logits. All right. And then we'll try to perform an inference
with the FP16 model and see how it fails. So the reason it fails with this error, so ADDMM
not implemented for half, means that some CPU kernels are not implemented for FP16.
So one of the disadvantages with PyTorch and FP16 and CPU is that for most of transformer-based
models, you are not able to use these models out of the box in Flow16. So one way to overcome
this issue is instead of loading your model in Flow16, you can also load it in B Flow16
and perform inference. Let's create a new dummy model, cast it in B Flow16 and get the logits of the
model. So we're going to create a copy of the model using deep copy,
so that we have the same weights across the BF16 and the FP32 model and we're going to cast
the model in B Flow16. All right, let's print the parameters as a sanity check.
Perfect, so your model is now in BF16 and let's get the logits of your model using the same input,
perfect, and then we can compute the mean difference the errors between those two logits to compare
if there is any huge gap, any gap between those two computed logits. So we're going to run this cell
just to have an idea. Perfect. So we can see that there are very small differences that can be
observed between the full precision model and the BF16 model. But in practice, when you switch from
FP32 to BF16, this doesn't really lead to a huge performance degradation in practice even for
large models. So casting the FP32 models into BF16 are most of the time, if not all the time,
performance cost-free. In the second section of this lesson, let's see how to load some popular
generative models in different data types and see kind of study their impact on their performance.
So we're going to load a multi-model model, meaning a model that can take many modalities
as input. So we're going to load a blip model that can take a text and image and predict some text.
So we're going to use a model called blip image captioning that can perform image captioning.
So you pass an image, you can pass an optional text, and the model will try to describe what's
in the image given the context that you have passed to the model. And if you're interested in
knowing more about this model and also other models into the hiking physical system,
you can also have a look at our short course called open source models with hiking face,
where we show you how to load these models and how to build fun and cool demos around these
models. Yeah, so let's get started. So we just have to import this class of blip for conditional
generation from transformers. And as I said, we're going to use this model. So blip image captioning
base and to load the model, nothing simpler than just calling from pre-trained on the model name.
So transformers by default loads the model in full precision, so flow 32, which is the default
for PyTorch. We can confirm that using the method we have designed. So if you print the
detail of each model's parameter, obviously we have a lot of parameters now because the model
is larger than our dummy model, but that's how you print each parameter's detail. And as you can
see, all of them should be in flow 32, which is the expected value for us. Perfect. We can also
learn more about the so-called the memory footprint of the model, meaning how much in terms of
memory, so megabytes, gigabytes, does the model takes in memory. So for that, we can just call
model that get the memory footprint to get the memory footprint of the model.
And we can print the values in bytes, but also in megabytes.
Yeah, so the flow 32 model takes approximately 990 megabytes. And let's see how to load the model
in different precision, such as float 16 or B float 16, as we've seen before. So the canonical way
in transformers to load models in different precision is to pass the parameter
torch d-type equal your target d-type directly in front pre-trained. So we're going to do that and
load our B float 16 model, because float 16 doesn't work on CPU for us.
All right, so once we have loaded the model, we can directly check the memory footprint of the model.
And we can also print the relative difference between the two memory footprints and see how much
did we gain in terms of memory. So yeah, as you can see, the B float 16 model is half of the size
of the FP32 model. So we just helped the size of the model by just passing torch d-type equal
torch B float 16. So you may be wondering now, how does this affect the model's predictions or
the model's generations? Is this reduction for free? So we're going to see that now by just
getting some qualitative comparison between the two models. So according to the model's
model card, this should be the way to load the model, load also the model's processor and image
and get some generation. So we're going to do that and try to get some generations with both models.
So we're going to load the processor first. And obviously we can use the same processor for
both models. So there is no difference. And we're going to load an image from the internet
and display it for you. All right, so it just, you know, simple image on the beach with a dog
and a woman. And we wrapped the whole generation pipeline in a small gets generation helper method
for you so that it's easier to get the models generation. So we're just going to call that method
by passing the processor the image and also the d-type of the model. So we can get the results of
the full precision model as follows. So let's print models prediction. Perfect. So a woman sitting
on the beach with her dog. Nice. Yeah, let's just try out with the BF16 model and qualitatively
compare both results. All right, so we got pretty similar results between the two models. So the only
difference is that the FP32 model predicted with her dog, whereas the BF16 model predicted with a dog.
But in both cases, the results seem pretty consistent with the image, pretty accurate.
The reason it affected the generated token here is that all the errors between, you know,
the FP32 logits and the BF16 logits gets accumulated across layers and layers. And since the model
is an autoregressive model, meaning it uses the results of the previous iteration to get the result
of the new iteration, all these errors gets accumulated until at some point impacting the
model's prediction. But overall, it doesn't really affect the overall performance of the model
and you can expect to use out of the box BF16 if you are on CPU or flute 16 if you are using a GPU.
All right, so before wrapping up the lesson, so I wanted to give a quick heads up on how those
torch D-type argument works under the hood in short summers and how you can adapt it in your workflow.
By that, I mean, in the current workflow, there is a small issue. So we need to first load
the model in flute 32 and then cast a model in let's say flute 16 or BF16. That can be an issue
in practice, for example, in production because you have to load the bigger model first and then cast
in flute 16 or BF16. You might want to directly be able to load out of the box the model
in your desired precision without having to first load it in full precision to save a memory.
So under the hood in short summers, we call a new TT method in PyTorch called set default D-type
where you pass the desired D-type. And then that way, when you initialize your model,
it's better if it gets initialized in your target D-type. So we'll see how to do that. Let's say I want
initialize my model directly in BF16. So you just have to call the torch.default set default D-type
torch BF16. And then I can initialize my model and it should be automatically casted
in BF16. Perfect. And once you have done that, don't forget to reset the default D-type
in Flo32 so that you may avoid some unexpected behaviors. If let's say you want to keep
some initializations of other, I don't know, tensors or inputs in Flo32, then you should revert
back to the default D-type, but this shouldn't affect the D-type of your model that you have already
loaded. So that's it for this lesson. I invite you to try out these approaches on other models.
You can also try out to load different models from Hanging Face Hub in different precision.
You can also try out different modalities. You can let's say try out audio models, vision models,
and you know, I'll load them in different precision and study a bit their impact and play with them.
If you find this trick also useful, don't hesitate to try it out in your workflow as well. So in
this lesson you have learned how to load models in half precision, so either in FB16 or BF16.
In the next lesson you will learn how to use Hanging Face's Quanto library in order to load
your models in int8 precision by quantizing them. So yeah, let's move on to the next lesson.

## 05. Quantization Theory
In this lesson, you will implement a technique called linear quantization.
This is the most popular quantization scheme, and it is used in most state-of-the-art quantization methods.
Then, you will apply linear quantization to real models using quantum, a python quantization toolkit from hugging face.
Let's get started.
Quantization is the process of mapping a large set to a small set of values.
There are many quantization techniques.
In this course, you will focus on linear quantization.
Let's have a look at an example of how to perform it be quantization on a simple tensor.
We will go from float32 values to 8 in values.
This will give you the intuition on how linear quantization works.
Let's take a look at this matrix of random numbers.
The values are in float32.
How do you convert the float32 weights to int8 weight without losing too much information?
Well, let's try this.
You can map the most positive number in this matrix, which is 728.6 in this case,
to the maximum value that the int8 can store, which is 127.
Similarly, you can map the most negative number of these matrix,
negative 184 in this case, to the minimum value that the int8 can store, which is negative 128.
You can map the rest of the values following a linear mapping.
You will see a bit more of that math later in an optional section at the end of the lesson.
But just assume that it is a little bit of multiplication and addition.
That's it. You manage to quantize the tensor.
Next, you can delete the original tensor to free up space.
You end up with the quantized tensors plus the parameters S and Z that you use to perform the linear mapping.
S stands for scale and Z for zero point.
Looks like you save a lot of space, but one question remains.
How do you go the other way?
From the quantized tensor back to the original tensor in FP32.
You can't get exactly the same as the original tensor,
but you can perform the quantization following the linear relationship that you use to quantize the original tensor.
Now, let's take a look at how you can perform that linear mapping to perform linear quantization.
We follow the linear mapping we defined previously to decontize the tensor.
Again, you can see the details at the end of this lesson, but it's going to be some math.
So the minimum and maximum decontized will get you these values.
And if you apply the same linear mapping to the other numbers, you can decontize the whole tensor.
As you can see, quantization results in a loss of information.
Let's compare the original tensor and the decontized tensor.
The result is that the decontized tensor is pretty accurate.
The quantization error are not zero, but they are not too bad either.
Even if linear quantization looks very simple, it is used in many states of the odd quantization methods.
Now you will use quantum, a Python quantization toolkit library from HuygingFace to quantize any PyTorch model using linear quantization.
In this classroom, the libraries have already been installed for you.
If you're running this on your own machine, you can install the Transformers library by running the following.
PeepingStore Transformers.
Similarly for quantum library, you need to type peepingStoreQuanto.
You also need to install Torch by typing peepingStoreTorch.
Since in this classroom, the libraries have already been installed, we don't need to run this cell, so I will just comment them out.
Now let's load the model using this specific class from Transformers library.
So we import from Transformers.
The auto model for COZLLM class.
We define the name of the model that we are going to load, which is this one.
IlusaAI is a non-profit research lab focused on interpretability, alignment, and ethics in artificial intelligence.
Then we will use the form pre-trained methods to load the model.
The first argument is the checkpoint, and this is optional, but you can also set the low CPU memory usage argument to be true, so that it loads the model more efficiently.
The model we just loaded is the PTA model from IlusaAI.
After loading the model, you will load the tokenizer as well.
To do that, we need to import from Transformers library the auto tokenizer class.
Then we will use the form pre-trained methods from the auto tokenizer to load the tokenizer.
And you just need to pass the model name.
The tokenizer you use to transform the text into a list of tokens that the model is able to understand.
Now, let's check if the model is able to generate text.
To do that, we will use the generate methods.
First, we need to define the text. We will choose a very simple text, such as Hello, my name is.
Then we need to pass this text into the tokenizer.
Then we also need to define the return tensor to be PT, which stands for PyTorch.
This way, we will get PyTorch tensors at the end.
Finally, to get the outputs, we just need to call the generate methods.
We pass the input inside the generate function, and we need to put the double-store in front since inputs is a dictionary of arguments.
We can also define the max new token argument.
We will set it to 10. These arguments control the number of new tokens that the model generates.
So, with these settings, the model can only generate a maximum of 10 new tokens.
Right now, as you can see, the output is just a list of tokens.
To decode this list of integers, we need to use the decode methods from the tokenizer.
This is optional, but we can also define skip special tokens to be true, to not have any special tokens in our output.
And as you can see, the model generated the following text, and I am a newbie to decide.
Now, let's check the size of the model.
Pitcher is a 400 million parameters model, so since you loaded this model in floating point 32, each parameter takes 32 bits, which is 4 bytes.
So, the model should take around 400 million times 4 bytes, which is equal to 1.6 gigabyte.
Let's check that using the compute module sizes that we already coded in the helper.py file.
To import that function, we just do from helper import compute module sizes.
Then, we just call it on our model.
And let's see what are the results.
And as you can see, it says that the model size is around 1.6 gigabyte, just as we expected.
Let's also have a look at the weights of one of the linear layers.
As you can see, the weights are in FP32.
Now, let's quantize the model.
To do that, you need to import two functions from the quantum library, quantize and freeze.
We also need to import torch.
Then, let's have a look at the architecture of the model.
As you can see, the model has many layers, but the one we are going to focus on are the linear layers.
These are the layers that we are going to quantize.
To quantize the model, you just call quantize, you pass the model.
You also need to specify the weights.
We want them to be quantized to the d-type torch.int8.
And if you remember, in the model, you can quantize the weights, but also the activation.
In this lesson, we will only quantize the weights.
So, this is why we set activation equals to none.
Let's check what happened to the model.
As you can see, the linear layers were replaced by q-linear quantize linear.
And if we look at one of the weights of these linear layers,
we see that the weights are still in floating point 32.
The model is not fully quantized.
For what you will do in this course, you don't need this intermediate state,
but for more advanced topic, the intermediate state is quite useful.
If your curious about when the intermediate state is used, please stick around for the optional section at the end of this lesson.
Next, to get the quantized model, we just need to call frees.
Now, if you look at these weights, you can see that they are quantized in torch.int8.
And we have also the linear quantization parameters scales right here.
In this case, you don't see the zero point because the zero point is set to zero.
Now, let's check the size of the model.
As you can see, it's now only a fourth of its original size.
It's good that we manage to decrease the size of the model,
but let's also have a look at the performance of the model.
If there is any performance degradation or not, let's do the same thing as we did earlier.
So output equals to model of the generate.
Then let's print the decoded output.
And as you can see, we get the same output.
This is not an extensive weight of testing if there is any performance degradation,
but still, it's good that we manage to get the same results.
That's it for the required part of this lesson.
All that's left are some optional discussion about the math of the linear mapping
and the explanation for the intermediate state for the quantum library.
If you're ready to move on to the next lesson,
this will give you another view of how quantization methods are applied to large language models.
The theory of linear quantization is very simple.
It is based on a simple idea linear mapping.
But first, let's have a look at the figure.
Here is the visual of the normal line of the original tensor,
which can be in floating point 32 on the top,
which goes from ormin to ormax.
The formula for linear quantization is as follows.
We have OR, which is equals to S times Q minus Z,
OR is the original value and Q is the quantized value.
S is the scale and Z is the zero point.
You can use this formula to quantize the original value or decontize the quantized value.
But one question remains, how do you get the scale and the zero point?
To get these parameters, you need to look at the extreme values
and you should get the following formulas.
And after solving these two equations,
you should get that the scale is equal to that,
and the zero point is equal to that.
Feel free to pause the video and take out a pencil on the paper
to derive the scale and the zero point yourself.
But don't worry, remember this is optional
and this is not required for you to successfully complete the course.
Also, recall that quantum library creates
an intermediate state after you call quantize.
Then you call frees to get the quantized weights.
This intermediate state can be useful for two things.
If you decide to quantize the model activation,
when you run the inference on a model by passing an input
such as an image, a text,
the activation of the model with vary depending on the input.
You get good linear parameters to perform the linear mapping
for the linear quantization of the activation.
It will really help to know what is the minimum
and maximum range of this activation.
To do that, you can get some simple data that is similar to the data
you would expect and run inference on the model.
This process is called calibration.
This is optional, but if you do the calibration,
you will get better quantized activation.
This intermediate state is also useful
when performing quantization or well-training.
Quantization or well-training means that when you train the model,
you can keep it in its intermediate state,
which means that for the forward pass,
you will use the quantized version of the weights,
but the model will still update its original
unquantized weights during that propagation.
The goal of quantization or well-training
is to better control how the model will perform
once you quantize it by calling Freeze.
Thanks for staying for the optional section.
Next, Eunice will give you an overview
of how quantization methods are applied to large language models.
Let's go on to the next video.

## 06. Quantization of LLMs
In this lesson, we'll take a look at what you get when you apply state-of-the-art
quantization to large language models. For example, is it possible for quantization to help you
with fine-tuning an LLM? Spoiler alert, the answer is yes. Let's see how.
As seen in the previous lessons, quantization is about compressing model weights in a certain manner.
Quantization applied to large language models brought a lot of interest in the open source AI
community as effectively quantizing those models with a minimal performance degradation can
open up a lot of cool opportunities for anyone. Many groundbreaking papers came out in a
short period of time and were naming just a few of them here. Starting from summer 2022,
LLM.int8 proposed a no-performance degradation 8-bit quantization method by decomposing the
underlying matrix multiplication in two stages. To mitigate emergent features from LLM's at scale,
the authors proposed to decompose the metamule in two stages, the outlier part in flow 16 and the
non-outlier part in int8. QLora proposed to make LLM's much more accessible by quantizing them
in four-bit precision and being able to fine-tune what we call low-rank adapters on top of the model.
Don't worry, we'll explain that a bit later. Therefore, making fine-tuning LLM's much more
accessible to anyone. On the other hand, AWQ, GPTQ, and also smooth quant proposed to pre-calibrate
the model so that the quantized model does not get affected by large activations caused by large
models. Later on, came out more methods that proved promising results for two-bit precision,
such as Q-sharp, HQQ, and more recently, HQLM. All these amazing work, which all aim at focusing
on making LLM smaller and faster, are open-source, meaning that you can directly get your hands
on the official implementation and try them out by your own. We cited just a few papers,
but you can easily find many other papers that work on this specific topic. At the same time,
you may also be wondering, are these methods generalizable for all models? Among these methods,
some of them require a calibration procedure, meaning you need to first pre-calibrate the model
by iterating over a data set and by minimizing an error to quantization error to get the best
quantization parameters. As the original work for these methods are adapted on large language
models, you might need to tweak these methods on your own projects and use cases. However,
some other methods do not require this step, meaning they can be used out of the box,
regardless of the modality, usually by replacing all instances of linear layers with a new
quantized modules, as we've been seeing in our lessons for the linear quantization.
Quantization makes also LLMs easier to distribute, since they are smaller. For example,
a 70 billion parameter model would need 280 gigabyte storage in full precision, whereas this
can be further reduced to 40 gigabytes if stored in 4-bit precision, leading to a 7x reduction.
This makes loading these models much more affordable and opens up opportunities of loading
these LLMs in local computers, using, for example, the GGUF format with lama.c++.
In the hugging phase ecosystem, you can also find some powerful quantized model distributors,
such as the block that you can see here. That distributes to the community quantized weights
that most of the time require some pre-calibration that might be quite costly for anyone to run it,
such as AWQ or GPTQ. You may be also wondering how all these quantization methods affect the
model performance with respect to its original version. One way to evaluate this is to check
the performance of the LLM on different well-known benchmarks, specifically crafted for large
language models. For that, you could check the open LLM leaderboard from hugging phase,
and the leaderboard now supports running evaluation on state-of-the-art quantized models,
such as LLM.int8, Qlora, and GPTQ. So, for the model you are interested in,
you could check its performance on the leaderboard and check if you are happy with it.
So, to wrap up, I would like to quickly cover another topic which is fine-tuning LLMs.
You may be wondering if it's possible to fine-tune a quantized model. Well, there are two cases where
you might be interested in this scenario. The first case, being where you would like to fine-tune a
model while being quantized to get the best quantized model possible, and the second case would
be useful for people who would like to adapt their model for specific use cases and applications,
such as fine-tuning an LLM on their own data set. The first scenario would be doable through
quantization our training. In this case, we train the model to be more accurate once we quantize it.
Note this method is not compatible with all the methods that we shared before, which belong to
the category of post-training quantization techniques. So, for the second use case, we would leverage
peft methods or parameter-efficient fine-tuning methods. So, peft methods aim at drastically reducing
the number of trainable parameters of a model while trying to keep the same performance as full
fine-tuning. So, we will specifically deep dive into peft plus q-dora, which leverages both
quantization and peft methods, and you can also check this example on how to train a LM7B model
on a free-tier Google Collab instance. So, this is an animated diagram that shows you how
low-rank adapters or lora work. When doing lora, you simply have to attach extra trainable
parameters, the blue weights that you can see on the left, to a frozen weight that you can see
on the right. Since the R parameter is usually extremely small compared to input hidden states
dimension, the final optimizer states end up being extremely small thus making the training
protocol much more accessible. Q-lora leverages this by quantizing the base weights,
so the blue weights that you can see on the left, in 4-bit precision and making sure the data type
of the activation of the quantized weight matches the data type of the lora weights. That way,
we could perform the sum that you can see here, easily, and we also get the best from both words,
quantization and parameter-efficient fine-tuning methods, and unlock many cool opportunities,
such as the ability to fine-tune llms on a free-tier Google Collab instance. So, that's it for
this lesson. I hope this gave you some good insights on the current stages of state-of-the-art
quantization with large language models and gave you a good overview of what you can achieve
with these methods and how you can apply them depending on your use case. So, in the next lesson,
we will wrap up this course together and review what you have done during this course.

## 07. Conclusion
Congratulations on making to the end of this short course.
You'll learn about some common data types
with hesitation in machine learning,
such as integer and floating point,
and how to load AI model using different data types.
You also learn about the underlying concepts
behind model quantization and how linear quantization works.
You use the quantum library to quantize
any PyTorch model in 8-bit precision.
Then you learn about some application of quantization
on LLMs, such as the recent state-of-the-art methods
for quantizing LLMs.
With this knowledge in hand,
you'll be able to better understand
the challenges of a model compression
and select the best quantization techniques for your use case.
If you find this course helpful,
maybe you can even share it with your friends.
