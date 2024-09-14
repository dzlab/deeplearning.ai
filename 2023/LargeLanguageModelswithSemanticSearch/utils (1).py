def print_result(result):
    """ Print results with colorful formatting """
    for i,item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()


### Revised version
def keyword_search(query, 
                   client,
                   results_lang='en', 
                   properties = ["title","url","text"],
                   num_results=3):

    where_filter = {
    "path": ["lang"],
    "operator": "Equal",
    "valueString": results_lang
    }

    response = (
        client.query.get("Articles", properties)
        .with_bm25(
          query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
        )
    
    result = response['data']['Get']['Articles']
    return result