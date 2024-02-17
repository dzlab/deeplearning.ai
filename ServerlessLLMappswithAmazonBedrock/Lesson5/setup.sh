#!/bin/bash

max_iteration=18
for i in $(seq 1 $max_iteration)
do
# Get the most recent CloudFormation stack that is in a 'CREATE_COMPLETE' or 'UPDATE_COMPLETE' state
  recent_stack=$(aws cloudformation describe-stacks \
                --query 'Stacks[?StackStatus==`CREATE_COMPLETE` || StackStatus==`UPDATE_COMPLETE`].[StackName,CreationTime]' \
                --output text | sort -k2,2r | head -n1 | cut -f1)

  if [ -z "$recent_stack" ]; then
    echo "cloudformation not ready"
    sleep 5 
  else
    echo "result_stack=$recent_stack"
    break
  fi
done

# Check if a stack was found
if [ -z "$recent_stack" ]; then
    echo "No running CloudFormation stack found."
    exit 1
fi

echo "Most recent running stack: $recent_stack"

# Retrieve outputs from the most recent stack
outputs=$(aws cloudformation describe-stacks --stack-name "$recent_stack" \
          --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' \
          --output text)

echo "outputs=$outputs"

# Check if outputs are available
if [ -z "$outputs" ]; then
    echo "No outputs found for stack $recent_stack."
    exit 1
fi

# Set outputs as environment variables
while read -r key value; do
    export "$key=$value"
#    echo "export $key=$value" >> /etc/bash.bashrc
    echo "Exported $key=$value"
done <<< "$outputs"

echo "All outputs from stack $recent_stack have been set as environment variables."


# Variables
LAYER_NAME="dlai-bedrock-jinja-layer"
ZIP_FILE_PATH="/home/jovyan/dlai-bedrock-jinja-layer.zip"

# Deploy the Lambda Layer
OUTPUT=$(aws lambda publish-layer-version --layer-name "$LAYER_NAME" --zip-file "fileb://$ZIP_FILE_PATH")

echo "output=$OUTPUT"

# Check if the deployment was successful
if [ $? -eq 0 ]; then
    echo "Lambda Layer deployed successfully."

    # Extract the Layer ARN with version
    LAYER_ARN=$(echo $OUTPUT | jq -r '.LayerVersionArn')
    
    # Check if jq command succeeded
    if [ $? -eq 0 ]; then
        echo "Layer ARN: $LAYER_ARN"
        
        # Set environment variable with the Layer ARN
        export LAMBDALAYERVERSIONARN=$LAYER_ARN
        echo "Environment variable LAMBDALAYERVERSIONARN set to $LAMBDALAYERVERSIONARN"
    else
        echo "Failed to extract Layer ARN using jq."
    fi
else
    echo "Failed to deploy Lambda Layer."
fi
