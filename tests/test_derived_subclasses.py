"""Subclasses with their own constructor and state survive every derived-block operation."""

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pytest

from NDLib import DataBlock
from NDLib.AxesStructures import LinearAxis


class Tagged(DataBlock):
    """A block with extra constructor arguments and extra state, like eels_base's NDImage."""

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        data: Any,
        axes: Any,
        quantity: str = "q",
        unit: str = "-",
        tag: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
        original_metadata: Mapping[str, Any] | None = None,
        provenance: Any = None,
        squeeze: bool = True,
    ) -> None:
        super().__init__(
            data,
            axes,
            quantity,
            unit,
            metadata=metadata,
            original_metadata=original_metadata,
            provenance=provenance,
            squeeze=squeeze,
        )
        self.tag = tag
        self.notes: list[str] = []

    def _construct(
        self, data: Any, axes: Any, quantity: str, unit: str, *, squeeze: bool = True
    ) -> Self:
        return type(self)(data, axes, quantity, unit, self.tag, squeeze=squeeze)

    def _inherit(self, new: Self, step: str | None = None) -> Self:
        new = super()._inherit(new, step)
        new.notes = [*self.notes]
        return new


def make() -> Tagged:
    rng = np.random.default_rng(0)
    axes = [
        LinearAxis(0.0, 1.0, 6, "x", 0, "nm", navigate=True),
        LinearAxis(0.0, 1.0, 4, "y", 1, "nm", navigate=True),
        LinearAxis(0.0, 2.0, 8, "E", 2, "eV"),
    ]
    block = Tagged(
        rng.random((6, 4, 8)),
        axes,
        "intensity",
        "counts",
        "sample-7",
        metadata={"k": 1},
    )
    block.notes.append("hello")
    return block


OPERATIONS = {
    "get": lambda b: b.get({"x": "x>=2"}),
    "crop_axis": lambda b: b.crop_axis("E", 2.0, 8.0),
    "pool": lambda b: b.pool({"E": 2}),
    "reduce_axis": lambda b: b.reduce_axis(
        "E", np.sum, out_quantity="sum", out_unit="counts"
    ),
    "reorder_axis": lambda b: b.reorder_axis("E", 0),
    "neg": lambda b: -b,
    "add": lambda b: b + b,
    "rename_quantity": lambda b: b.rename_quantity("other"),
    "with_navigation": lambda b: b.with_navigation("E"),
    "isel": lambda b: b.isel(x=slice(1, 3)),
    "sel": lambda b: b.sel(E=slice(2.0, 6.0), y=1.0),
    "mean_over": lambda b: b.mean_over("x", "y"),
    "sum_over": lambda b: b.sum_over("E"),
}


@pytest.mark.parametrize("name", OPERATIONS)
def test_operation_keeps_type_state_and_metadata(name: str) -> None:
    source = make()
    result = OPERATIONS[name](source)
    assert type(result) is Tagged
    assert result.tag == "sample-7"
    assert result.notes == ["hello"]
    assert result.metadata == {"k": 1}
    assert result.notes is not source.notes  # derived state is copied, not shared


def test_squeeze_false_reaches_the_subclass_constructor() -> None:
    result = make().isel(x=slice(2, 3))
    assert type(result) is Tagged
    assert result.shape == (1, 4, 8)


def test_split_on_axis_keeps_type() -> None:
    block = Tagged(
        np.arange(12.0).reshape(4, 3),
        [LinearAxis(0, 1, 4, "x", 0, "nm"), LinearAxis(0, 1, 3, "y", 1, "nm")],
        tag="t",
    )
    parts = block.split_on_axis("x")
    assert len(parts) == 4
    assert all(type(p) is Tagged and p.tag == "t" for p in parts)


def test_save_load_round_trip(tmp_path: Any) -> None:
    block = make()
    block.save(tmp_path / "tagged")
    loaded = Tagged.load(tmp_path / "tagged")
    assert type(loaded) is Tagged
    assert loaded.metadata == {"k": 1}
    np.testing.assert_allclose(loaded.compute(), block.compute())


def test_subclass_without_override_falls_back_for_squeeze_false() -> None:
    """The documented fallback: custom __init__ and no _construct override -> plain DataBlock."""

    class Bare(DataBlock):
        def __init__(
            self,
            data: Any,
            axes: Any,
            quantity: str = "q",
            unit: str = "-",
            extra: int = 0,
        ) -> None:
            super().__init__(data, axes, quantity, unit)
            self.extra = extra

    block = Bare(
        np.zeros((4, 3)),
        [LinearAxis(0, 1, 4, "x", 0, "nm"), LinearAxis(0, 1, 3, "y", 1, "nm")],
    )
    assert type(block.isel(x=slice(0, 2))) is DataBlock
    assert type(-block) is Bare  # the squeeze=True path still instantiates the subclass
