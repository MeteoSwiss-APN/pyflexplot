"""Files setup."""
# Standard library
import dataclasses as dc
from typing import Tuple
from typing import Union

# Local
from .base_setup import BaseSetup


# SR_TMP <<< TODO cleaner solution
def is_files_setup_param(param: str) -> bool:
    return param.replace("files.", "") in FilesSetup.get_params()


# SR_TMP TODO pull common base class out of LayoutSetup, ModelSetup etc.
@dc.dataclass
class FilesSetup(BaseSetup):
    input: str
    output: Union[str, Tuple[str, ...]]
    # output_time_format: str = "%Y%m%d%H%M"
    outfile_time_format: str = "%Y%m%d%H%M"
