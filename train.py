import lutorpy as lua
import boto3
import os

print os.environ['S3_BUCKET_NAME']

train = require("trainer")

train()

s3 = boto3.resource('s3')
s3.meta.client.upload_file('data/model.t7', os.environ['S3_BUCKET_NAME'], 'model.t7')