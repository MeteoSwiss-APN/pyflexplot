import os
import logging
from pathlib import Path
from typing import Tuple

import boto3
from botocore.exceptions import ClientError
from pyflexplot.config.service_settings import Bucket
from pyflexplot.setups.model_setup import ModelSetup
from pyflexplot import CONFIG

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

    if not infile.startswith("s3:/"):
        raise ValueError(f'{infile} must be an S3 URI.')

    _, bucket_name, *key_prefix = [s for s in infile.split("/") if s]
    key = "/".join(key_prefix)
    filename = key_prefix[-1]
    return bucket_name, key, filename

def download_key_from_bucket(key: str, 
                             dst_dir: Path, 
                             bucket: Bucket = CONFIG.main.aws.s3.input,
                             filename: str | None = None) -> Path:
    """ 
    Download objects with key from S3 bucket. Filename of resulting local file is the value of the key.

    Args:
        dst_dir: Parent directory to download file to.

        bucket: S3 bucket from where data will be fetched.
    """

    client = boto3.Session().client('s3', region_name=bucket.region)


    if not os.path.exists( dst_dir ):
        os.makedirs( dst_dir )

    if not filename:
        filename = key.split('/')[-1]

    path = dst_dir / filename
    if not os.path.exists( path.parent ):
        os.makedirs( path.parent )
    with open(path, 'wb') as data:
        client.download_fileobj(bucket.name, key, data)
    _LOGGER.info('Finished downloading %s from bucket %s to %s', key, bucket.name, path)

    return path


def upload_outpaths_to_s3(upload_outpaths_to_s3: list[str],
                    model: ModelSetup,
                    bucket: Bucket = CONFIG.main.aws.s3.output, 
                    ) -> None:
    
    if not model:
        raise ValueError("Model object must be provided to upload to S3, \
                         model name and base time are used in the object key.")
    try:
        client = boto3.Session().client('s3', region_name=bucket.region)

        for outpath in upload_outpaths_to_s3:
            key = f"{model.name}/{model.base_time}/{Path(outpath).name}"
            try:
                _LOGGER.info("Uploading file: %s to bucket: %s with key: %s", outpath, bucket.name, key)
                client.upload_file(outpath, bucket.name, key)
            except ClientError as e:
                _LOGGER.error(e)
    except Exception as err:
        _LOGGER.error('Error uploading directory to S3.')
        raise err
