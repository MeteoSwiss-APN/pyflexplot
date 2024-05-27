import tempfile
from pathlib import Path
import os

import boto3
from moto import mock_aws
import pytest

from pyflexplot.s3.s3 import download_keys_from_bucket, upload_directory
from pyflexplot import CONFIG
from pyflexplot.config.service_settings import Bucket


@pytest.fixture(scope="function")
def aws_credentials():
    """
    Mocked AWS credentials for moto.
    Strictly not necessary, but it's a good safety net if somehow the regular connection is invoked.
    """
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    if os.getenv("AWS_PROFILE"):
        del os.environ["AWS_PROFILE"]


@pytest.fixture(scope="function")
def s3(aws_credentials):
    """
    This fixture replaces the regular boto3 client with the mocking library moto.
    The replacement is done by mocking the constant which is used to make sure that a S3 client is reused for calls.
    This replacement is a bit different from the usual way of introducing moto described in
    https://docs.getmoto.org/en/latest/docs/getting_started.html. Doing it this way was caused by problems with the
    MinIO server url which could not be easily overwritten.
    """
    with mock_aws():
        session = boto3.Session()
        s3 = session.client('s3')
        s3.create_bucket(Bucket=CONFIG.main.aws.s3.input.name)
        s3.create_bucket(Bucket=CONFIG.main.aws.s3.output.name)
        yield s3


@pytest.fixture(scope="session")
def resource_dir() -> Path:
    resource: Path = Path(os.path.dirname(os.path.realpath(__file__))) / 'resource'
    return resource


def test_download_keys_from_bucket(s3, model_data):
    
    bucket = Bucket(
        region = CONFIG.main.aws.s3.model_data.region, 
        name = CONFIG.main.aws.s3.model_data.name
        )

    some_files = list(model_data.iterdir())[4:7]
    _add_files_to_bucket(bucket, some_files, s3)

    expected_objs = {file.name for file in some_files}
    expected_objs.pop()

    with tempfile.TemporaryDirectory() as tmpdirname:
        actual_objs = download_keys_from_bucket(list(expected_objs), Path(tmpdirname), bucket) 

    for obj in actual_objs:
        assert obj.name in expected_objs
    assert len(expected_objs) == len(actual_objs)
    assert len(actual_objs) < len(some_files)


def test_upload_directory(s3, landuse_data):

    # given
    bucket = Bucket(
        region = CONFIG.main.aws.s3.output.region, 
        name = CONFIG.main.aws.s3.output.name
    )

    # when
    upload_directory(landuse_data, bucket)

    # then
    assert 'Contents' in s3.list_objects(Bucket = bucket.name)

    for path in landuse_data.iterdir():

        # check the files were uploaded as expected
        actual = s3.get_object(Bucket = bucket.name, Key = path.name)["Body"].read()
        with open(path, mode='rb') as f:
            assert actual == f.read()

def _add_files_to_bucket(bucket: Bucket, files: list[Path], s3) -> None:

    for path in files:
        s3.upload_file(path, bucket.name, path.name)
