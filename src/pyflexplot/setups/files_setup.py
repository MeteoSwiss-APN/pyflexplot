"""Files setup."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Union

# First-party
from srutils.dataclasses import cast_field_value
from srutils.dict import merge_dicts


# SR_TMP <<< TODO cleaner solution
def is_files_setup_param(param: str) -> bool:
    return param.replace("files.", "") in FilesSetup.get_params()


# SR_TMP TODO pull common base class out of LayoutSetup, ModelSetup etc.
@dc.dataclass
class FilesSetup:
    input: str
    # output: Union[str, Tuple[str, ...]]
    # output_time_format: str = "%Y%m%d%H%M"
    outfile_time_format: str = "%Y%m%d%H%M"

    # SR_TMP copy-pasted from PlotPanelSetup
    def derive(self, params: Dict[str, Any]) -> "FilesSetup":
        return type(self).create(merge_dicts(self.dict(), params, overwrite_seqs=True))

    def dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "FilesSetup":
        params = cls.cast_many(params)
        return cls(**params)

    # SR_TMP Identical to CoreDimensions.cast etc.
    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        return cast_field_value(
            cls,
            param,
            value,
            auto_wrap=True,
            bool_mode="intuitive",
            timedelta_unit="hours",
            unpack_str=False,
        )

    # SR_TMP Identical to ModelSetup.cast_many etc.
    @classmethod
    def cast_many(
        cls, params: Union[Collection[Tuple[str, Any]], Mapping[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            params_dct: Dict[str, Any] = {}
            for param, value in params:
                if param in params_dct:
                    raise ValueError("duplicate parameter", param)
                params_dct[param] = value
            return cls.cast_many(params_dct)
        params_cast = {}
        for param, value in params.items():
            params_cast[param] = cls.cast(param, value)
        return params_cast

    # SR_TMP Identical to CoreSetup.get_params and CoreDimensions.get_params
    @classmethod
    def get_params(cls) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101
