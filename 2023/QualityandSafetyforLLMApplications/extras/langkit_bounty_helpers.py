import datetime
import logging
import sys
import pandas as pd

import bounty

from whylogs import log
from whylogs.core.relations import Predicate
from whylogs.core.schema import DatasetSchema
from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.viz import NotebookProfileVisualizer
from whylogs.api.logger.transient import TransientLogger


def show_metrics(result_set):
    paths = []
    for col_name, col in result_set.profile().view().get_columns().items():
        paths.extend([col_name + ":" + path for path in col.get_metric_component_paths()])
    for path in paths:
        print(path)

def llm_schema():
    _stderr = sys.stderr
    _stdout = sys.stdout
    sys.stderr = sys.stdout = None
    from langkit import llm_metrics
    schema = llm_metrics.init()
    sys.stderr = _stderr
    sys.stdout = _stdout
    return schema

def base_clean_schema(metric_name):
    schema = udf_schema()

    final_udf_spec = []
    for udf_spec in schema.multicolumn_udfs:
        if metric_name in udf_spec.udfs:
            final_udf_spec.append(udf_spec)
    schema.multicolumn_udfs = final_udf_spec
    return schema

def base_show_queries(annotated_dataset, metric_name, n, ascending):
    if ascending == None and metric_name in ["response.relevance_to_prompt"]:
        sorted_annotated_dataset = annotated_dataset.sort_values(by=[metric_name], ascending=True)
    elif ascending == None and metric_name in ["prompt.toxicity", "response.toxicity"]:
        sorted_annotated_dataset = annotated_dataset.sort_values(by=[metric_name], ascending=False)
    else:
        if ascending == None: 
            ascending = False
        sorted_annotated_dataset = annotated_dataset.sort_values(by=[metric_name], ascending=ascending)
        
    return sorted_annotated_dataset[:n][["prompt", "response", metric_name]]


def show_langkit_critical_queries(dataset, metric_name, n=3, ascending=None):
    annotated_dataset, _ = base_clean_schema(metric_name).apply_udfs(dataset)
    return base_show_queries(annotated_dataset, metric_name, n, ascending)


def show_custom_critical_queries(dataset, metric_name, n=3, ascending=None):
    return base_show_queries(dataset, metric_name, n, ascending)


def base_visualize_metric(dataset_or_profile, metric_name, schema, numeric):
    logging.getLogger("whylogs.viz.notebook_profile_viz").setLevel(logging.ERROR)
    if type(dataset_or_profile) == pd.DataFrame:
        prof_view = TransientLogger().log(dataset_or_profile, schema=schema).profile().view()
    else:
        prof_view = dataset_or_profile.view()

    viz = NotebookProfileVisualizer()
    viz.set_profiles(prof_view)
    if metric_name in ["prompt.has_patterns", "response.has_patterns"]:
        return viz.distribution_chart(metric_name)
    else:
        return viz.double_histogram(metric_name)
    
def visualize_langkit_metric(dataset_or_profile, metric_name, numeric=None):
    schema=base_clean_schema(metric_name)
    if numeric == None:
        if metric_name in ["prompt.has_patterns", "response.has_patterns"]:
            numeric = False
        else:
            numeric = True
    return base_visualize_metric(dataset_or_profile, metric_name, schema, numeric)

def visualize_custom_metric(dataset_or_profile, metric_name, numeric):
    return base_visualize_metric(dataset_or_profile, metric_name, DatasetSchema(), numeric)

def evaluate_examples(dataset=pd.DataFrame(), scope="all"):
    return bounty.evaluate_bounty_entries(dataset, scope)

def evaluate_threshold(dataset, metric_name, threshold, scope="all"):
    schema=base_clean_schema(metric_name)
    annotated_data = schema.apply_udfs(dataset)
    annotated_data = annotated_data[threshold(annotated_data[metric_name])]
    return bounty.evaluate_bounty_entries(annotated_data, scope)

def load_monitoring_data():
    data = []

    #base_url = "https://whylabs-public.s3.us-west-2.amazonaws.com/langkit-examples/behavior-llm"
    base_path = "../extras/monitoring_data"
    starting_date = datetime.datetime(2023, 3, 17).replace(tzinfo=datetime.timezone.utc)
    ending_date = datetime.datetime(2023, 3, 27).replace(tzinfo=datetime.timezone.utc)

    for day in pd.date_range(starting_date, ending_date):
        date_str = day.strftime("%Y-%m-%d")
        try:
            url = f"{base_path}/daily_{date_str}.csv"
            df = pd.read_csv(url)
            df = df.drop(columns=["chat_date", "ref_0", "ref_1", "ref_2"])
            data.append({'date_str': date_str, 'datetime': day, 'data': df})
        except:
            print(f"Could not load data for {date_str}. Continuing...")
    
    return data

def get_openai_key():
    import os
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

def get_openai_base_url():
    import os
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_BASE")
