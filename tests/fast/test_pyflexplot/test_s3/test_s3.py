import tempfile
from pathlib import Path
import os

import boto3
from moto import mock_aws
import pytest

from pyflexplot.s3 import (
    download_key_from_bucket, 
    upload_outpaths_to_s3, 
    expand_key, 
    split_s3_uri)
from pyflexplot import CONFIG
from pyflexplot.config.service_settings import Bucket
from pyflexplot.setups.model_setup import ModelSetup



@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", 'testing')
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", 'testing')
    monkeypatch.setenv("AWS_SECURITY_TOKEN", 'testing')
    monkeypatch.setenv("AWS_SESSION_TOKEN", 'testing')
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    if "AWS_PROFILE" in os.environ:
        monkeypatch.delenv("AWS_PROFILE")


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


def test_expand_key_valid():
    result = expand_key("data_{ens_member:03}.nc", (1, 2, 3))
    expected = ["data_001.nc", "data_002.nc", "data_003.nc"]
    assert result == expected

    result = expand_key("data_{ens_member:03d}.nc", (1, 2, 3))
    expected = ["data_001.nc", "data_002.nc", "data_003.nc"]
    assert result == expected

def test_expand_key_no_ensemble_members():
    with pytest.raises(ValueError) as exc_info:
        expand_key("data_{ens_member:03}.nc", ())
    assert str(exc_info.value) == "Must provide list of ensemble members as argument to expand key data_{ens_member:03}.nc."

def test_expand_key_invalid_pattern():
    with pytest.raises(RuntimeError) as exc_info:
        expand_key("data.nc", (1, 2, 3))
    assert str(exc_info.value) == "Cannot expand key, key must contain pattern {ens_member:03} or {ens_member:03d}"


def test_split_s3_uri():
    result = split_s3_uri("s3://my_bucket/path/to/my_file.txt")
    expected = ("my_bucket", "path/to/my_file.txt", "my_file.txt")
    assert result == expected

def test_split_s3_uri_root_file():
    result = split_s3_uri("s3://my_bucket/my_file.txt")
    expected = ("my_bucket", "my_file.txt", "my_file.txt")
    assert result == expected

def test_split_s3_uri_with_slashes():
    result = split_s3_uri("s3:///my_bucket///path///to///my_file.txt")
    expected = ("my_bucket", "path/to/my_file.txt", "my_file.txt")
    assert result == expected

def test_split_s3_uri_invalid():
    with pytest.raises(ValueError) as exc_info:
        split_s3_uri("invalid_s3_uri")
    assert str(exc_info.value) == "invalid_s3_uri must be an S3 URI."


def test_download_key_from_bucket(s3):
    
    bucket = CONFIG.main.aws.s3.input
    
    test_files: list[Path] = []

    try:
        for i in range(5):
            temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=f"test_file_{i}_", suffix=".txt")
            test_files.append(Path(str(temp_file.name)))
            
            with open(Path(str(temp_file.name)), 'w') as f:
                f.write(f"Dummy data for test file {i}\n")
        
        _add_files_to_bucket(bucket, test_files, s3)

        expected_objs = {file.name for file in test_files}
        expected_objs.pop()

        actual_objs = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for key in expected_objs:
                actual_objs.append(download_key_from_bucket(key, Path(tmpdirname), bucket))

        for obj in actual_objs:
            assert obj.name in expected_objs
        assert len(expected_objs) == len(actual_objs)
        assert len(actual_objs) < len(test_files)

    finally:
        # Cleanup: Delete the created files
        for file in test_files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")


def test_upload_outpaths_to_s3(s3):

    # given
    bucket = CONFIG.main.aws.s3.output

    model = ModelSetup(name = 'COSMO-1E', base_time='1234')

    test_files = []

    try:
        for i in range(5):
            temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=f"test_file_{i}", suffix=".txt")
            test_files.append(Path(str(temp_file.name)))
            
            with open(Path(str(temp_file.name)), 'w') as f:
                f.write(f"Dummy data for test file {i}\n")
        
        # when
        upload_outpaths_to_s3(test_files, model, bucket=bucket)

        # then
        assert 'Contents' in s3.list_objects(Bucket = bucket.name)

        for path in test_files :

            # check the files were uploaded as expected
            actual = s3.get_object(Bucket = bucket.name, Key = f"{model.name}/{model.base_time}/{path.name}")["Body"].read()
            with open(path, mode='rb') as f:
                assert actual == f.read() 
            
    finally:
        # Cleanup: Delete the created files
        for file in test_files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")



def _add_files_to_bucket(bucket: Bucket, files: list[Path], s3) -> None:

    for path in files:
        s3.upload_file(path, bucket.name, path.name)
