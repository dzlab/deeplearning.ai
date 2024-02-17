import boto3
import json
import os
import datetime
from botocore.exceptions import ClientError


class CloudWatch_Helper: 

    def __init__(self):
        # Create a Boto3 client for the CloudWatch Logs service     
        self.cloudwatch_logs_client = boto3.client('logs', region_name="us-west-2")

    def create_log_group(self, log_group_name):
        try:
            response = self.cloudwatch_logs_client.create_log_group(logGroupName=log_group_name)
            print(f"Log group '{log_group_name}' created successfully.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                print(f"Log group '{log_group_name}' already exists.")
            else:
                print(f"Failed to create log group '{log_group_name}'. Error: {e}")

    def print_recent_logs(self, log_group_name, minutes=5):
        try:
            # Calculate the time range
            end_time = int(datetime.datetime.now().timestamp() * 1000)  # Current time in milliseconds
            start_time = end_time - (minutes * 60 * 1000)  # 5 minutes ago in milliseconds

            # Fetch log streams (assumes logs are stored in streams within the log group)
            streams = self.cloudwatch_logs_client.describe_log_streams(
                logGroupName=log_group_name,
                orderBy='LastEventTime',
                descending=True
            )

            for stream in streams.get('logStreams', []):
                # Fetch log events from each stream
                events = self.cloudwatch_logs_client.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=stream['logStreamName'],
                    startTime=start_time,
                    endTime=end_time
                )

                for event in events.get('events', []):
                    try:
                        # Try to load the string as JSON
                        json_data = json.loads(event['message'])
                        # Pretty print the JSON data
                        print(json.dumps(json_data, indent=4))
                    except json.JSONDecodeError:
                        # If it's not valid JSON, print the original string
                        print(event['message'])
                    print(f'{"-"*25}\n')

        except ClientError as e:
            print(f"Error fetching logs: {e}")

