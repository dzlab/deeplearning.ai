def print_result(result):
    """ Print results with colorful formatting """
    for i,item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()


def search_wikipedia_subset(client, query, num_results = 3, results_lang='en',
                   properties = ["text", "title", "url", "views", "lang", "_additional {distance}"]):

    nearText = {"concepts": [query]}
    
    # To filter by language
    if results_lang:
        where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
        }
        response = (
            client.query
            .get("Articles", properties)
            .with_where(where_filter)
            .with_near_text(nearText)
            .with_limit(5)
            .do()
        )

    # Search all languages
    else:
        response = (
            client.query
            .get("Articles", properties)
            .with_near_text(nearText)
            .with_limit(5)
            .do()
        )


    result = response['data']['Get']['Articles']

    return result



def generate_given_context(query, weav_client, co_client ):
    
    results = search(client, query, results_lang='en' )

    title = results[0]['title']
    context = results[0]['text']



    prompt = f"""
    You are a useful AI trained to answer questions based on the context your are provided.
    Use the Context Information provided below to answer the questions "{query}". If the answer to 
    the question is in the context, extract it and print it. If it's not contained in the provided 
    information, say "I do not know". 
    ---
    Context information about {title}:
    Context: {context}
    End of Context Information
    ---
    Question: {query}
    """

    # to answer from the Context Information
    prediction = co_client.generate(
        prompt=prompt,
        max_tokens=50,
        # model='command-light',
        # temperature=0.3,
        num_generations=5)


    return prediction, context_title, context_text
