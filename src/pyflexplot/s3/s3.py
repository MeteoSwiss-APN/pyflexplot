import os
import logging
import glob
from pathlib import Path
from typing import Tuple

import boto3
from botocore.exceptions import ClientError
from pyflexplot.config.service_settings import Bucket
from pyflexplot import CONFIG

client = boto3.Session().client('s3', region_name=CONFIG.main.aws.s3.input.region)

_LOGGER = logging.getLogger(__name__)

def expand_key(key: str, ens_member_ids: Tuple[int]) -> list[str]:

    if not ens_member_ids:
        raise ValueError(f"Must provide list of ensemble members as argument to expand key {key}.")

    search_patterns = ["{ens_member:03}", "{ens_member:03d}"]

    validation_msg = f"Cannot expand key, key must contain pattern {' or '.join(search_patterns)}"

    if not any(pattern in key for pattern in search_patterns):
        raise RuntimeError(validation_msg)
    
    for pattern in search_patterns:
        if pattern in key:
            return [key.replace(pattern, f"{i:03}") for i in ens_member_ids]

    return []

def split_s3_uri(infile: str) -> Tuple[str, str, str]:

    _, bucket_name, *key_prefix = [s for s in infile.split("/") if s]
    key = "/".join(key_prefix)
    filename = key_prefix[-1]
    return bucket_name, key, filename

def download_key_from_bucket(key: str, bucket: Bucket, dst_dir: Path, filename: str | None = None) -> list[Path]:
    """ 
    Download objects with key from S3 bucket. Filename of resulting local file is the value of the key.

    Args:
        dst_dir: Parent directory to download file to.

        bucket: S3 bucket from where data will be fetched.
    """

    if not os.path.exists( dst_dir ):
        os.makedirs( dst_dir )

    if not filename:
        filename = key.split('/')[-1]

    path = dst_dir / filename
    if not os.path.exists( path.parent ):
        os.makedirs( path.parent )
    _LOGGER.info('Downloading %s from bucket %s to %s', key, bucket.name, path)
    with open(path, 'wb') as data:
        client.download_fileobj(bucket.name, key, data)

    return path


def upload_directory(directory: Path,
                    bucket: Bucket = Bucket(
                        region = CONFIG.main.aws.s3.output.region, 
                        name = CONFIG.main.aws.s3.output.name,
                        retries = CONFIG.main.aws.s3.output.retries), 
                    parent: str | None = None) -> None:

    # Verify directory exists
    if not directory.is_dir():
        _LOGGER.error("Directory is empty, cannot upload: %s ", directory)
        raise RuntimeError("Directory provided to upload does not exist.")

    try:
        client = boto3.Session().client('s3', region_name=bucket.region)

        path_list = [Path(f) for f in glob.iglob(str(directory)+'/**', recursive=True) if os.path.isfile(f)]
        if parent:
            path_list = [f for f in path_list if f.parent.name == parent]

        for path in path_list:
            key = path.name
            try:
                _LOGGER.info("Uploading file: %s to bucket: %s with key: %s", path, bucket.name, key)
                client.upload_file(path, bucket.name, key)
            except ClientError as e:
                _LOGGER.error(e)
    except Exception as err:
        _LOGGER.error('Error uploading directory to S3.')
        raise err
