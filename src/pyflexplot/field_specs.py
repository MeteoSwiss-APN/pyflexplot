# -*- coding: utf-8 -*-
"""
Field specifications.
"""
# Local
from .setup import Setup
from .utils import summarizable
from .var_specs import MultiVarSpecs


@summarizable
class FieldSpecs:
    """FLEXPART field specifications."""

    def __init__(
        self, setup: Setup, multi_var_specs: MultiVarSpecs,
    ):
        """Create an instance of ``FieldSpecs``."""
        self.setup = setup
        self.multi_var_specs = multi_var_specs

    # SR_TMP <<<
    @classmethod
    def create(cls, setup_or_setups):
        multi_var_specs_lst = MultiVarSpecs.create(setup_or_setups)
        assert isinstance(multi_var_specs_lst, list)  # SR_TMP
        assert isinstance(multi_var_specs_lst[0], MultiVarSpecs)  # SR_TMP
        return [
            cls(multi_var_specs.setup, multi_var_specs)
            for multi_var_specs in multi_var_specs_lst
        ]

    def __repr__(self):
        s = f"{type(self).__name__}(\n"

        # Variables specifications
        s += f"  var_specs: {len(self.multi_var_specs)}x\n"
        for line in repr(self.multi_var_specs).split("\n"):
            s += f"    {line}\n"
        s += f")"
        return s
