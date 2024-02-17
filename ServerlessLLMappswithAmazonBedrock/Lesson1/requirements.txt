# Note: To run the code in these files in your own environment you will need to install 
# some Python dependencies, and also have access to an AWS environment.  

# AWS Development Environment Setup:
# ----------------------------------
# 1. Create AWS Account and an IAM user.  Follow this guide for more details: https://docs.aws.amazon.com/SetUp/latest/UserGuide/setup-overview.html
# 2. Generate some access keys for your IAM user. For more information see here: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html#Using_CreateAccessKey
# 3. Install the AWS CLI. Follow this guide for more details: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
# 4. Configure AWS CLI: Run 'aws configure' with your access keys, set default region to 'us-west-2' for this code. For more information see here: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-methods 
# 5. Use Boto3 for AWS SDK in Python. Your locally run code should interact with your AWS account. https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

# Note: If using a company AWS account, consult your AWS administrator before proceeding.


# requirements file
# This code was developed and tested on Python 3.10


boto3==1.28.68
