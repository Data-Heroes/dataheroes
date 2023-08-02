import boto3
from botocore.utils import IMDSFetcher

def shutdown_current_aws_instance():
    boto_client = boto3.client('ec2', region_name='us-east-1')
    # Get the current instance's id
    my_id = IMDSFetcher()._get_request("/latest/meta-data/instance-id", None).text
    # Issue an instance shutdown
    boto_client.stop_instances(InstanceIds=[my_id])