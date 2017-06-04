import lutorpy as lua
import boto3

train = require("trainer")

train()

s3 = boto3.resource('s3')
s3.meta.client.upload_file('data/model.t7', 'chatbot-deploy-bucket', 'model.t7')