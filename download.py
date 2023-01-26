import boto3
import os

# fetch credentials from env variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

# setup a AWS S3 client/resource
s3 = boto3.resource(
    's3', 
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    )

# point the resource at the existing bucket
bucket = s3.Bucket('anyoneai-datasets')

# download the training dataset
with open('training_image_set.tgz', 'wb') as data:
    print('Downloading training set')
    bucket.download_fileobj('cars196/car_ims.tgz', data)
# download the dataset labels
with open('car_dataset_labels.csv', 'wb') as data:
    print('Downloading CSV file..')
    bucket.download_fileobj('cars196/car_dataset_labels.csv', data)
