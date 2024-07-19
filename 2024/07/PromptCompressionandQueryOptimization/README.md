# [Prompt Compression and Query Optimization](https://learn.deeplearning.ai/courses/prompt-compression-and-query-optimization)

![Promotional banner for](https://ci3.googleusercontent.com/meips/ADKq_NZyUpSNJIT-Y6ZhO4S_r6uXGpXelo7qcwexE_q0ntfELmXX9kbWP2IFMEP6uP314vJzCCwRoxvGj_8YkrI7Vrfpbp_s5hDy02o3EDtLzfOqwucSjRPnAlD0fN_VpGRRnHOtWxMDd964dE9w_psdwJgSsYd9xS52SjJPh6T8qPS-_zX-WBcWwk6aL-LbIAYxw-8ediEa6RPxUB2kVxAUtV9y5QTzzKmbD8O2S16CUWeyHMgPUes6PBUbWahgA58=s0-d-e1-ft#https://info.deeplearning.ai/hs-fs/hubfs/V2_DeepLearning_MongoDB_Banner_2070x1080%20_.png?width=1120&upscale=true&name=V2_DeepLearning_MongoDB_Banner_2070x1080%20_.png)

Dear learner, 

 

Introducing [Prompt Compression and Query Optimization](https://learn.deeplearning.ai/courses/prompt-compression-and-query-optimization), a short course made in collaboration with MongoDB and taught by Richmond Alake, Developer Advocate at MongoDB.

 

In this course, you’ll learn to integrate traditional database features with vector search capabilities to optimize the performance and cost-efficiency of large-scale Retrieval Augmented Generation (RAG) applications.

 

You'll learn how to apply these key techniques: 

  - **Prefiltering and Postfiltering:** Techniques for filtering results based on specific conditions. Prefiltering is done at the database index creation stage, while postfiltering is applied after the vector search is performed.
  - **Projection:** This technique involves selecting a subset of the fields returned from a query to minimize the size of the output.
  - **Reranking:** This involves reordering the results of a search based on other data fields to move the more desired results higher up the list.
  - **Prompt Compression:** To reduce the length of prompts, which can be expensive to process in large-scale applications.

![Launch email GIFs (25)](https://ci3.googleusercontent.com/meips/ADKq_Nb3MfciAyj6cMkoo-IlzkBPg_VZ2WVpnZ6SBZwnMTtP_gx-WbDAfh3wj6aWFvxxb-P1oykvYAtTJKFy1RBM5vEvAGosvME6yXWAcrLjbGpKDZGDrENYFFnGldpfjc576x8DEpXYQIQYicQva_X4VxFMq_1q4T5p4VFlcpw2AMDudgGYy4asj8Tw8nUKqxLw380cJDkv2CnAXsrWYQOu=s0-d-e1-ft#https://info.deeplearning.ai/hs-fs/hubfs/Launch%20email%20GIFs%20(25).gif?width=1120&upscale=true&name=Launch%20email%20GIFs%20(25).gif)

You’ll also learn with hands-on exercises how to: 

  - Implement vector search for RAG using MongoDB.
  - Develop a multi-stage MongoDB aggregation pipeline. 
  - Use metadata to refine and limit the search results returned from database operations, enhancing efficiency and relevancy.
  - Streamline the outputs from database operations by incorporating a projection stage into the MongoDB aggregation pipeline, reducing the amount of data returned and optimizing performance, memory usage, and security.
  - Rerank documents to improve information retrieval relevance and quality, and use metadata values to determine reordering position.
  - Implement prompt compression and gain an intuition of how to use it and the operational advantages it brings to LLM applications.

Start optimizing the efficiency, security, query processing speed, and cost of your RAG applications with prompt compression and query optimization techniques.

## Details

- Combine vector search capabilities with traditional database operations to build efficient and cost-effective RAG applications.

- Learn how to use pre-filtering, post-filtering, and projection techniques for faster query processing and optimized query output.

- Use prompt compression techniques to reduce the length of prompts that are expensive to process in large-scale applications.


|Lesson|Video|Code|
|-|-|-|
|Introduction|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/L0/sc-MongoDB-C1-L0-master.m3u8)||
|Vanilla Vector Search|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/L1/sc-MongoDB-C1-L1-master.m3u8)|[code](./L1/)|
|Filtering With Metadata|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/L2/sc-MongoDB-C1-L2-master.m3u8)|[code](./L2/)|
|Projections|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/L3/sc-MongoDB-C1-L3-master.m3u8)|[code](./L3/)|
|Boosting|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/L4/sc-MongoDB-C1-L4-master.m3u8)|[code](./L4/)|
|Prompt Compression|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/L5_v2/sc-MongoDB-C1-L5-master.m3u8)|[code](./L5/)|
|Conclusion|[video](https://dyckms5inbsqq.cloudfront.net/MongoDB/C1/Conclusion_v2/sc-MongoDB-C1-Conclusion-master.m3u8)||
|Appendix-Tips and Help||[code](./Appendix/)|