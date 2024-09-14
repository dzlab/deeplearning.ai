import random

import jsonlines

from util.make_llama_3_prompt import make_llama_3_prompt


def load_training_data(args, make_question):
    path = f"data/training_data/{args.training_file_name}"

    limit = 1000

    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reversed(list(reader))):
            if index >= limit:
                break

            yield {
                "input": make_llama_3_prompt(**make_question(obj)),
                "output": obj["sql"] + "<|eot_id|>",
            }


def get_dataset(args, make_question):
    if args.training_file_name == "archive/generated_queries.jsonl":
        return "407d05ea9d8f119f214ada0bde018225dbae16b589f5680f745ea12098f1fd4f"
    elif (
        args.training_file_name
        == "archive/generated_queries_v2_large_filtered_cleaned.jsonl"
    ):
        return "72e8d7d3a94f4d180b1b95f9b0ac5c9cf4476c132e02f301ed7f50164e09c961"
    dataset = list(load_training_data(args, make_question))

    return dataset
