import boto3
import json
import os

class S3_Helper: 

    def __init__(self):
        
        # Get the account ID being used
        sts_client = boto3.client('sts')
        response = sts_client.get_caller_identity()
        account_id = response['Account']
        
        # Create a Boto3 client for the S3 service      
        self.s3_client = boto3.client('s3', region_name='us-west-2')

    def list_objects(self, bucket_name):
        try:
            # List objects within the bucket
            response = self.s3_client.list_objects_v2(Bucket=bucket_name)

            # Check if the bucket has any objects
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    creation_time = obj['LastModified']
                    print(f"Object: {key}, Created on: {creation_time}")
            else:
                print(f"No objects found in the bucket: {bucket_name}")
    
        except Exception as e:
            print(f"Error: {str(e)}")

    def upload_file(self, bucket_name, file_name):
        try:
            # Upload file to an S3 object from the specified local path
            self.s3_client.upload_file(file_name, bucket_name, file_name)
            print(f"Object '{file_name}' uploaded to bucket '{bucket_name}'")
        except Exception as e:
            print(f"Error: {str(e)}")
            
    def download_object(self, bucket_name, object_key):
        try:
            # Download the object from S3 to the specified local path
            self.s3_client.download_file(bucket_name, object_key, f"./{object_key}")
            print(f"Object '{object_key}' from bucket '{bucket_name}' to './{object_key}'")
        except Exception as e:
            print(f"Error: {str(e)}")
