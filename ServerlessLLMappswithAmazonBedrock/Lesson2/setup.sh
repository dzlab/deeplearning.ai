#!/bin/bash

# Get the most recent CloudFormation stack that is in a 'CREATE_COMPLETE' or 'UPDATE_COMPLETE' state
recent_stack=$(aws cloudformation describe-stacks \
                --query 'Stacks[?StackStatus==`CREATE_COMPLETE` || StackStatus==`UPDATE_COMPLETE`].[StackName,CreationTime]' \
                --output text | sort -k2,2r | head -n1 | cut -f1)

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

# Check if outputs are available
if [ -z "$outputs" ]; then
    echo "No outputs found for stack $recent_stack."
    exit 1
fi

# Set outputs as environment variables
while read -r key value; do
    export "$key=$value"
    echo "Exported $key=$value"
done <<< "$outputs"

echo "All outputs from stack $recent_stack have been set as environment variables."
