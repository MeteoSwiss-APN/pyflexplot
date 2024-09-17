"""
Module for interacting with AWS S3 for downloading and uploading files.

This module provides utility functions to handle S3 URIs,
download files from an S3 bucket, and upload files to an S3 bucket. 
It also includes functions for expanding S3 keys with
ensemble member identifiers and retrying operations with exponential backoff.
"""

import os
import logging
import random
import time
from pathlib import Path
from typing import Callable


import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pyflexplot.config.service_settings import Bucket
from pyflexplot.setups.model_setup import ModelSetup
from pyflexplot import CONFIG

_LOGGER = logging.getLogger(__name__)

def expand_key(key: str, ens_member_id: int | None) -> str:
    """Expand a key pattern into S3 object
      key using ensemble member identifier."""

    if ens_member_id is None:
        raise ValueError(
            f"Must provide ensemble member id as argument to expand key {key}.")

    search_patterns = ["{ens_member:03}", "{ens_member:03d}"]

    validation_msg = f"Cannot expand key, key must contain pattern {' or '.join(search_patterns)}"

    if not any(pattern in key for pattern in search_patterns):
        raise RuntimeError(validation_msg)

    for pattern in search_patterns:
        if pattern in key:
            return key.replace(pattern, f"{ens_member_id:03}")

    return key

def split_s3_uri(infile: str) -> tuple[str, str, str]:
    """Split an S3 URI into bucket name, key, and filename."""

    if not infile.startswith("s3:/"):
        raise ValueError(f'{infile} must be an S3 URI.')

    _, bucket_name, *key_prefix = [s for s in infile.split("/") if s]
    key = "/".join(key_prefix)
    if key_prefix:
        filename = key_prefix[-1]
    else:
        filename = ''
    return bucket_name, key, filename

def download_key_from_bucket(key: str,
                             dst_dir: Path,
                             bucket: Bucket = CONFIG.main.aws.s3.input) -> Path:
    """ 
    Download object from S3 bucket.
    Filename of resulting local file is formatted value of the key.

    Args:
        key: S3 object key
        dst_dir: Parent directory to download file to.
        bucket: S3 bucket from where data will be fetched.
    """

    client = boto3.Session().client('s3', config=Config(
                                            region_name=bucket.region,
                                            retries={
                                                'max_attempts': int(bucket.retries),
                                                'mode': 'standard'
                                            })
                                        )
    # Make directory if not existing
    if not os.path.exists( dst_dir ):
        os.makedirs( dst_dir )

    path = dst_dir / key.replace('/', '-')

    # Download object
    with open(path, 'wb') as data:
        client.download_fileobj(bucket.name, key, data)

    _LOGGER.info('Finished downloading %s from bucket %s to %s', key, bucket.name, path)

    return path


def upload_outpaths_to_s3(upload_outpaths: list[str],
                    model: ModelSetup,
                    bucket: Bucket = CONFIG.main.aws.s3.output,
                    ) -> None:
    """Upload a list of local file paths to an S3 bucket \
    using the model for key formatting."""

    if not model:
        raise ValueError("Model object must be provided to upload to S3, \
                         model name and base time are used in the object key.")
    try:
        client = boto3.Session().client('s3', config=Config(
                                            region_name=bucket.region,
                                            retries={
                                                'max_attempts': int(bucket.retries),
                                                'mode': 'standard'
                                            })
                                        )

        for outpath in upload_outpaths:
            key = f"{model.name}/{model.base_time}/{Path(outpath).name}"
            try:
                _LOGGER.info("Uploading file: %s \
                             to bucket: %s \
                             with key: %s", outpath, bucket.name, key)
                _retry_with_backoff(
                    client.upload_file,
                    args=[outpath, bucket.name, key],
                    retries=int(bucket.retries)
                    )
            except ClientError as e:
                _LOGGER.error(e)
    except Exception as err:
        _LOGGER.error('Error uploading paths to S3.')
        raise err


def _retry_with_backoff(fn: Callable,
                        args: list | None = None,
                        kwargs: dict | None = None,
                        retries: int = 5,
                        backoff_in_seconds: int = 1) -> None:
    """Retry a function with exponential backoff."""
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    x = 0
    while True:
        try:
            fn(*args, **kwargs)
            return
        except Exception as e:
            if x == retries:
                raise RuntimeError(f"retried {fn} {retries} times.") from e
            sleep: float = backoff_in_seconds * 2 ** x + random.uniform(0, 1)
            logging.info("Sleep: %.2f seconds", sleep)
            time.sleep(sleep)
            x += 1
