import snowflake.connector
import boto3
import json

def get_secret(secret_name):
    secret_name = secret_name
    region_name = "us-east-1"

    session = boto3.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    secret_value_response = client.get_secret_value(SecretId=secret_name)
        
    return json.loads(secret_value_response['SecretString'])


def get_snowflake_connection(secret_name, warehouse, database, schema):
    snowflake_secret = get_secret(secret_name)
    
    ctx = snowflake.connector.connect(
        user=snowflake_secret['username'],
        password=snowflake_secret['password'],
        account=snowflake_secret['accountname'],
        ocsp_response_cache_filename='/tmp/ocsp_response_cache'
        )
    
    ctx.execute_string("USE WAREHOUSE {};".format(warehouse), return_cursors=False)
    ctx.execute_string("USE DATABASE {};".format(database), return_cursors=False)
    ctx.execute_string("USE SCHEMA {};".format(schema), return_cursors=False)
    
    return ctx