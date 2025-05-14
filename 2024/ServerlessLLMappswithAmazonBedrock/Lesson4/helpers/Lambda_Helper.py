import boto3
import zipfile
import json
import os




class Lambda_Helper: 

    def __init__(self):
        
        # Get the account ID being used
        sts_client = boto3.client('sts')
        response = sts_client.get_caller_identity()
        account_id = response['Account']
        
        # Create a Boto3 client for the Lambda service
        self.lambda_client = boto3.client('lambda', region_name='us-west-2')  
        self.function_name = 'LambdaFunctionDLAICourse'
        self.role_arn = f"arn:aws:iam::{account_id}:role/LambdaRoleDLAICourse"
        self.function_description = "Lambda function uploaded by a notebook in a DLAI course."
        self.lambda_arn = ""
        self.lambda_environ_variables = {}
        self.filter_rules_suffix = ""
        
        self.s3_client = boto3.client('s3', region_name='us-west-2')
            
    def deploy_function(self, code_file_names, function_name=""):

        if function_name:
            self.function_name = function_name
        else:
            print(f"Using function name: {self.function_name}")
        
        print('Zipping function...')
        zip_file_path = 'lambda_function.zip'

        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for code_file_name in code_file_names:
                zipf.write(code_file_name, arcname=code_file_name)

        try:
            print('Looking for existing function...')
            # Try to get the details of the Lambda function
            self.lambda_client.get_function(FunctionName=self.function_name)

            # If the function exists, update its code
            print(f"Function {self.function_name} exists. Updating code...")
            response = self.lambda_client.update_function_code(
                FunctionName=self.function_name,
                ZipFile=open(zip_file_path, 'rb').read()  # Read the ZIP file and provide its content
            )
            print(f"Function {self.function_name} code updated: {response['LastModified']}")
            self.lambda_arn = response['FunctionArn']
            print("Done.")

        except self.lambda_client.exceptions.ResourceNotFoundException:
            # If the function does not exist, create a new one
            print(f"Function {self.function_name} does not exist. Creating...")
            response = self.lambda_client.create_function(
                FunctionName=self.function_name,
                Runtime='python3.11',
                Role=self.role_arn,
                Handler='lambda_function.lambda_handler',
                Description=self.function_description,
                Layers=[os.environ['LAMBDALAYERVERSIONARN']],
                Timeout=120,
                Code={
                    'ZipFile': open(zip_file_path, 'rb').read()
                },
                Environment= {'Variables': self.lambda_environ_variables }
            )
            print(f"Function {self.function_name} created: {response['FunctionArn']}")
            self.lambda_arn = response['FunctionArn']
            print("Done.")

        except Exception as e:
            # Handle other potential exceptions
            print(f"An error occurred: {e}")
            self.lambda_arn = ""
            print("Done, with error.")
            

    def add_lambda_trigger(self, bucket_name, function_name=""):

        if function_name:
            self.function_name = function_name
        else:
            print(f"Using function name of deployed function: {self.function_name}")
        
        # Check and remove existing permissions for the specific source (S3 bucket)
        try:
            policy = self.lambda_client.get_policy(FunctionName=self.function_name)['Policy']
            policy_dict = json.loads(policy)
        
            for statement in policy_dict['Statement']:
                if statement['Action'] == 'lambda:InvokeFunction' and self.lambda_arn in statement['Resource']:
                    self.lambda_client.remove_permission(
                        FunctionName=self.function_name,
                        StatementId=statement['Sid']
                    )
                    print(f"Removed existing permission: {statement['Sid']}")

        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Handle if the policy is not found - might mean function has no permissions set
            pass

        except Exception as e:
            # Handle other exceptions
            print(f"Error checking for existing permissions: {e}")
            return

        # Grant permission to S3 to invoke the Lambda function
        try:
            response = self.lambda_client.add_permission(
                FunctionName=self.function_name,
                Action='lambda:InvokeFunction',
                Principal='s3.amazonaws.com',
                StatementId='s3-trigger-permission',  # A unique statement ID
                SourceArn=f"arn:aws:s3:::{bucket_name}"
            )
            print_out = json.dumps(json.loads(response['Statement']), indent=4)
            print(f"Permission added with Statement: {print_out}")

        except Exception as e:
            print(f"Error adding Lambda permission: {e}")
            return

        # Add bucket notification to trigger the Lambda function
        lambda_arn = self.lambda_client.get_function(
            FunctionName=self.function_name
        )['Configuration']['FunctionArn']
                
        notification_configuration = {
            'LambdaFunctionConfigurations': [
                {
                    'LambdaFunctionArn': lambda_arn,
                    'Events': ['s3:ObjectCreated:*'],
                    'Filter': {
                        'Key': {
                            'FilterRules': [
                                {
                                    'Name': 'suffix',
                                    'Value': self.filter_rules_suffix
                                }
                            ]
                        }
                    }
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_notification_configuration(
                Bucket=bucket_name,
                NotificationConfiguration=notification_configuration
            )
            print(f"Trigger added for {bucket_name} -> {self.function_name}")

        except Exception as e:
            print(f"Error setting S3 notification: {e}")
