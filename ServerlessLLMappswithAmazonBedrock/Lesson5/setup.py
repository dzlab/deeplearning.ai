import boto3, os

from helpers.Lambda_Helper import Lambda_Helper
from helpers.S3_Helper import S3_Helper

lambda_helper_pre = Lambda_Helper()
s3_helper = S3_Helper()
bucket_name = os.environ['LEARNERS3BUCKETNAMETEXT']
lambda_helper_pre.filter_rules_suffix = "json"
lambda_helper_pre.deploy_function(["pre_lambda_function.py", "prompt_template.txt"], function_name="LambdaFunctionSummarize")

lambda_helper_pre.add_lambda_trigger(bucket_name)
