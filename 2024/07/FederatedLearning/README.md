# Federated Learning


![Promo banner for the two-part course series](https://ci3.googleusercontent.com/meips/ADKq_NYBC8ChWrY36YVZaQUPN1WJ5GtsuTdMDxgBFi1DwuIFBFSB9ltMa67DBt9ALmTYlhtzqmNJ5jHPii9LhFPC2Av9ZvnG9t3Lqmogv5RgRCq18FV8PHQVMpYRQ_zbtGOkt96ZWpqIri0Il1jCLxqj_FEwapZpinwl3Jui8WL__cevF2F8fiz65rraZRXy5XUYwzd6XphQHcMJ1PUT5t9EKzsNBx1etOMwDYPz6nj_OUGEroZb1A=s0-d-e1-ft#https://info.deeplearning.ai/hs-fs/hubfs/V2_DeepLearning_Flower_Banner_2070x1080.png?width=1120&upscale=true&name=V2_DeepLearning_Flower_Banner_2070x1080.png)

Dear learner, 

 

Addressing security and privacy in applications is vital. Applications built on LLMs pose special challenges, especially regarding private data.

 

Introducing Federated Learning, a two-part course series built in collaboration with Flower Labs and instructed by its Co-Founder & CEO, Daniel Beutel, and Co-founder and Chief Scientific Officer Nic Lane.

 

This series is designed to help you learn how to use Flower, a popular open source framework, to build a federated learning system, and implement federated fine-tuning of LLMs with private data. Federated learning allows models to be trained across multiple devices or organizations without sharing data, improving privacy and security. 

 

In the first course, called [Intro to Federated Learning](https://learn.deeplearning.ai/courses/intro-to-federated-learning), you'll learn about the federated training process, how to tune and customize it, how to increase data privacy, and how to manage bandwidth usage in federated learning.

 

In the second course, [Federated Fine-tuning of LLMs with Private Data](https://learn.deeplearning.ai/courses/intro-to-federated-learning-c2), you’ll learn to apply federated learning to LLMs. You’ll explore challenges like data memorization and the computational resources required by LLMs, and explore techniques for efficiency and privacy enhancement, such as Parameter-Efficient Fine-Tuning (PEFT) and Differential Privacy (DP).

 

This two-part course series is self-contained. If you already know what federated learning is, you can start directly with part two of the course.


## Details
- Explore the components of federated learning systems and learn to customize, tune, and orchestrate them for better model training.

- Leverage federated learning to enhance LLMs by effectively managing key privacy and efficiency challenges.

- Learn how techniques like parameter-efficient fine-tuning and differential privacy are crucial for making federated learning secure and efficient.

### [Part One: Intro to Federated Learning](https://learn.deeplearning.ai/courses/intro-to-federated-learning/)

In detail, here’s what you’ll do in part one: 

  -  Learn how federated learning is used to train a variety of models, ranging from those for processing speech and vision all the way to the large language models, across distributed data while offering key data privacy options to users and organizations.
  -  Learn how to train AI on distributed data by building, customizing, and tuning a federated learning project using Flower and PyTorch.
  -  Gain intuition on how to think about Private Enhancing Technologies (PETs) in the context of federated learning, and work through an example using Differential Privacy, which protects individual data points from being traced back to their source. 
  -  Learn about two types of differential privacy - central and local - along with the dual approach of clipping and noising to protect private data.
  -  Explore the bandwidth requirements for federated learning and how you can optimize it by reducing the update size and communication frequency.

[Enroll in Part One](https://learn.deeplearning.ai/courses/intro-to-federated-learning)

|Lesson|Video|Code|
|-|-|-|
|Introduction|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/L0/sc-Flower-C1-L0-master.m3u8)||
|Why Federated Learning|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/L1/sc-Flower-C1-L1-master.m3u8)|[code](./Part01/L1/)|
|Federated Training Process|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/L2/sc-Flower-C1-L2-master.m3u8)|[code](./Part01/L2/)|
|Tuning|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/L3/sc-Flower-C1-L3-master.m3u8)|[code](./Part01/L3/)|
|Data Privacy|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/L4/sc-Flower-C1-L4-master.m3u8)|[code](./Part01/L4/)|
|Bandwidth|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/L5/sc-Flower-C1-L5-master.m3u8)|[code](./Part01/L5/)|
|Conclusion|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C1/Conclusion/sc-Flower-C1-Conclusion-master.m3u8)||

### [Part Two: Federated Fine-tuning of LLMs with Private Data](https://learn.deeplearning.ai/courses/intro-to-federated-learning-c2/)

In the second part, you’ll learn how to train powerful models with your own data in a federated way, called federated LLM fine-tuning: 

  -  Understand the importance of safely training LLMs using private data.
  -  Learn about the limitations of current training data and how Federated LLM Fine-tuning can help overcome these challenges.
  -  Build an LLM that is fine-tuned with private medical data to answer complex questions, where you’ll see the benefits of federated methods when using private data.
  -  Learn how federated LLM fine-tuning works and how it simplifies access to private data, reduces bandwidth with Parameter-Efficient Fine-Tuning (PEFT), and increases privacy to training data with Differential Privacy. 
  -  Understand how LLMs can leak training data, how federated LLMs can lower this risk.

[Enroll in Part Two](https://learn.deeplearning.ai/courses/intro-to-federated-learning-c2)


|Lesson|Video|Code|
|-|-|-|
|Introduction|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C2/L0/sc-Flower-C2-L0-master.m3u8)||
|Smarter LLMs with Private Data|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C2/L1/sc-Flower-C2-L1-master.m3u8)||
|Centralized LLM Fine-tuning|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C2/L2/sc-Flower-C2-L2-master.m3u8)|[code](./Part02/L2/)|
|Federated Fine-tuning for LLMs|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C2/L3/sc-Flower-C2-L3-master.m3u8)|[code](./Part02/L3/)|
|Keeping LLMs Private|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C2/L4/sc-Flower-C2-L4-master.m3u8)|[code](./Part02/L4/)|
|Conclusion|[video](https://dyckms5inbsqq.cloudfront.net/Flower/C2/Conclusion/sc-Flower-C2-Conclusion-master.m3u8)||
