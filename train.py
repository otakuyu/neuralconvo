import lutorpy as lua
import boto3
import os
import json


print os.environ['S3_BUCKET_NAME']

train = require("trainer")

train()

s3 = boto3.resource('s3')
s3.meta.client.upload_file('data/model.t7', os.environ['S3_BUCKET_NAME'], 'model.t7')

with open('progress.json', 'w') as outfile:
    json.dump({
        "progress": 1.0
    }, outfile)