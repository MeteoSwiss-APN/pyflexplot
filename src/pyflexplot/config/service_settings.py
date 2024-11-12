# Third party
from pydantic import BaseModel

# Local
from .base_settings import BaseServiceSettings

class Bucket(BaseModel):
    region: str
    name: str
    endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    retries: int

class S3(BaseModel):
    input: Bucket
    output: Bucket

class AWS(BaseModel):
    s3: S3

class Paths(BaseModel):
    input: str
    output: str

class LocalSettings(BaseModel):
    paths: Paths

class AppSettings(BaseModel):
    app_name: str
    aws: AWS
    local: LocalSettings

class ServiceSettings(BaseServiceSettings):
    main: AppSettings
