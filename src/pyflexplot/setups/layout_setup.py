"""Layout setup."""
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
def is_layout_setup_param(param: str) -> bool:
    return param.replace("layout.", "") in LayoutSetup.get_params()


# SR_TMP TODO pull common base class out of LayoutSetup, ModelSetup etc.
@dc.dataclass
class LayoutSetup:
    # plot_type: str = "auto"
    type: str = "auto"

    def __post_init__(self) -> None:
        layouts = [
            "auto",
            "vintage",
            "post_vintage",
            "post_vintage_ens",
        ]
        if self.type not in layouts:
            raise ValueError(
                f"invalid type '{self.type}'; choices: "
                + ", ".join(map("'{}'".format, layouts))
            )

    # SR_TMP copy-pasted from PlotPanelSetup
    def derive(self, params: Dict[str, Any]) -> "LayoutSetup":
        return type(self).create(merge_dicts(self.dict(), params, overwrite_seqs=True))

    def dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "LayoutSetup":
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
