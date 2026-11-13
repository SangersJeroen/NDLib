"""LinearAxis, navigation/signal roles, metadata and calibrated selection."""

from copy import deepcopy

import dask.array as da
import numpy as np
import pytest

from NDLib import DataBlock, SignalAxis
from NDLib.AxesStructures import CategoricalAxis, LinearAxis, UnorderedSignalAxis


def spectrum_image(lazy: bool = True) -> tuple[DataBlock, np.ndarray]:
    """I(y, x, E): 4 x 5 scan of 50-point spectra, E from 100 eV in 2 eV steps."""
    raw = np.random.default_rng(0).random((4, 5, 50))
    data = da.from_array(raw, chunks=(2, 5, 50)) if lazy else raw
    axes = [
        LinearAxis(0.0, 0.5, 4, "y", 0, "nm", navigate=True),
        LinearAxis(0.0, 0.5, 5, "x", 1, "nm", navigate=True),
        LinearAxis(100.0, 2.0, 50, "E", 2, "eV"),
    ]
    block = DataBlock(
        data,
        axes,
        "intensity",
        "counts",
        metadata={"beam_kev": 200.0},
        original_metadata={"raw": {"tag": 1}},
    )
    return block, raw


class TestLinearAxis:
    def test_points_are_computed_from_offset_and_scale(self) -> None:
        axis = LinearAxis(10.0, 0.25, 5, "E", 0, "eV")
        np.testing.assert_allclose(axis.points, [10.0, 10.25, 10.5, 10.75, 11.0])
        assert (axis.min, axis.max, axis.scale, axis.size) == (10.0, 11.0, 0.25, 5)
        assert isinstance(axis, SignalAxis)

    def test_points_are_lazy(self) -> None:
        axis = LinearAxis(0.0, 1.0, 10**6, "q", 0, "1/nm")
        assert "points" not in vars(axis)
        assert len(axis) == 10**6

    @pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_bad_scale(self, scale: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            LinearAxis(0.0, scale, 3, "E", 0, "eV")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LinearAxis(0.0, 1.0, 0, "E", 0, "eV")

    def test_index_of_is_nearest_and_clipped(self) -> None:
        axis = LinearAxis(100.0, 2.0, 50, "E", 0, "eV")
        assert axis.index_of(105.1) == 3
        assert axis.index_of(-1e9) == 0
        assert axis.index_of(1e9) == 49

    def test_equality_and_hash_follow_points(self) -> None:
        a = LinearAxis(0.0, 1.0, 4, "x", 0, "nm")
        b = deepcopy(a)
        assert a == b
        assert hash(a) == hash(b)
        assert a != LinearAxis(0.0, 2.0, 4, "x", 0, "nm")

    def test_save_load_round_trip_keeps_linear(self, tmp_path) -> None:
        block, raw = spectrum_image()
        block.save(tmp_path / "si")
        loaded = DataBlock.load(tmp_path / "si")
        assert all(isinstance(a, LinearAxis) for a in loaded.axes)
        assert loaded.axis_obj("E").offset == pytest.approx(100.0)
        np.testing.assert_array_equal(loaded.compute(), raw)

    def test_existing_operations_accept_linear_axes(self) -> None:
        block, raw = spectrum_image()
        cropped = block.crop_axis("E", 110.0, 130.0)
        assert cropped.axis_obj("E").size == 11
        pooled = block.pool({"E": 5})
        assert pooled.axis_obj("E").size == 10
        np.testing.assert_allclose(
            pooled.compute(), raw.reshape(4, 5, 10, 5).mean(axis=-1)
        )


class TestRoles:
    def test_nav_and_signal_axes(self) -> None:
        block, _ = spectrum_image()
        assert [a.name for a in block.nav_axes] == ["y", "x"]
        assert [a.name for a in block.signal_axes] == ["E"]
        assert block.nav_shape == (4, 5)
        assert block.signal_shape == (50,)

    def test_with_navigation_replaces_roles(self) -> None:
        block, _ = spectrum_image()
        flipped = block.with_navigation("E")
        assert [a.name for a in flipped.nav_axes] == ["E"]
        assert [a.name for a in block.nav_axes] == ["y", "x"]  # original untouched

    def test_with_navigation_unknown_axis(self) -> None:
        block, _ = spectrum_image()
        with pytest.raises(KeyError):
            block.with_navigation("nope")

    def test_roles_survive_save_and_load(self, tmp_path) -> None:
        block, _ = spectrum_image()
        block.save(tmp_path / "si")
        loaded = DataBlock.load(tmp_path / "si")
        assert [a.name for a in loaded.nav_axes] == ["y", "x"]


class TestMetadata:
    def test_constructor_copies_metadata(self) -> None:
        source = {"a": [1]}
        block = DataBlock(
            np.zeros(3), [LinearAxis(0, 1, 3, "x", 0, "nm")], metadata=source
        )
        source["a"].append(2)
        assert block.metadata == {"a": [1]}

    def test_propagates_through_operations(self) -> None:
        block, _ = spectrum_image()
        for derived in (
            block.crop_axis("E", 110.0, 130.0),
            block.pool({"E": 5}),
            block.reorder_axis("E", 0),
            -block,
            block + block,
            block.rename_quantity("q2"),
        ):
            assert derived.metadata == {"beam_kev": 200.0}
            assert derived.original_metadata == {"raw": {"tag": 1}}

    def test_derived_metadata_is_independent(self) -> None:
        block, _ = spectrum_image()
        derived = block.crop_axis("E", 110.0, 130.0)
        derived.metadata["beam_kev"] = 80.0
        assert block.metadata["beam_kev"] == pytest.approx(200.0)

    def test_provenance_records_new_operations(self) -> None:
        block, _ = spectrum_image()
        result = block.sel(x=slice(0.0, 1.0)).mean_over("x")
        assert len(result.provenance) == 2
        assert result.provenance[0].startswith("sel(")
        assert result.provenance[1].startswith("mean_over")
        assert block.provenance == []

    def test_save_load_round_trip(self, tmp_path) -> None:
        block, _ = spectrum_image()
        block = block.isel(x=0)
        block.metadata["numpy_value"] = np.float64(1.5)
        block.save(tmp_path / "si")
        loaded = DataBlock.load(tmp_path / "si")
        assert loaded.metadata == {"beam_kev": 200.0, "numpy_value": 1.5}
        assert loaded.original_metadata == {"raw": {"tag": 1}}
        assert loaded.provenance == block.provenance

    def test_unserializable_metadata_is_rejected(self, tmp_path) -> None:
        block, _ = spectrum_image()
        block.metadata["bad"] = object()
        with pytest.raises(TypeError, match="not JSON serializable"):
            block.save(tmp_path / "si")


class TestSelection:
    def test_isel_int_drops_axis_and_keeps_calibration(self) -> None:
        block, raw = spectrum_image()
        out = block.isel(y=2, x=slice(1, 4))
        assert [a.name for a in out.axes] == ["x", "E"]
        assert [a.index_in_array for a in out.axes] == [0, 1]
        x = out.axis_obj("x")
        assert isinstance(x, LinearAxis)
        assert (x.offset, x.scale, x.size) == (0.5, 0.5, 3)
        np.testing.assert_allclose(out.compute(), raw[2, 1:4, :])

    def test_isel_step_and_negative(self) -> None:
        block, raw = spectrum_image()
        out = block.isel(E=slice(0, 50, 10), y=-1)
        e = out.axis_obj("E")
        assert (e.offset, e.scale, e.size) == (100.0, 20.0, 5)
        np.testing.assert_allclose(out.compute(), raw[-1, :, 0:50:10])

    def test_isel_is_lazy(self) -> None:
        block, _ = spectrum_image()
        assert isinstance(block.isel(x=1).data, da.Array)
        assert isinstance(block.sel(E=slice(110.0, 120.0)).data, da.Array)

    def test_isel_errors(self) -> None:
        block, _ = spectrum_image()
        with pytest.raises(IndexError):
            block.isel(x=5)
        with pytest.raises(KeyError):
            block.isel(z=0)
        with pytest.raises(TypeError):
            block.isel(x=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="empty"):
            block.isel(x=slice(3, 3))

    def test_sel_slice_is_inclusive_in_calibrated_units(self) -> None:
        block, raw = spectrum_image()
        out = block.sel(E=slice(110.0, 120.0), x=slice(0.5, 1.5))
        assert out.axis_obj("E").size == 6  # 110, 112, ..., 120
        assert out.axis_obj("x").size == 3  # 0.5, 1.0, 1.5
        np.testing.assert_allclose(out.compute(), raw[:, 1:4, 5:11])

    def test_sel_open_bounds(self) -> None:
        block, _ = spectrum_image()
        assert block.sel(E=slice(None, 104.0)).axis_obj("E").size == 3
        assert block.sel(E=slice(196.0, None)).axis_obj("E").size == 2  # 196, 198

    def test_sel_number_picks_nearest(self) -> None:
        block, raw = spectrum_image()
        out = block.sel(E=131.5, y=0.74)  # E index 15.75 -> 16, y index 1.48 -> 1
        assert [a.name for a in out.axes] == ["x"]
        np.testing.assert_allclose(out.compute(), raw[1, :, 16])

    def test_sel_number_halfway_takes_lower_point(self) -> None:
        block, raw = spectrum_image()
        linear = block.sel(E=131.0, y=0.0, x=0.0)  # between 130 and 132
        np.testing.assert_allclose(linear.compute(), raw[0, 0, 15])
        points = np.array([1.0, 2.0, 4.0])
        generic = DataBlock(np.arange(3.0), [SignalAxis(points, "q", 0, "-")])
        assert generic.sel(q=3.0).compute() == pytest.approx(
            1.0
        )  # between 2 and 4 -> lower

    def test_sel_outside_range_is_empty(self) -> None:
        block, _ = spectrum_image()
        with pytest.raises(ValueError, match="empty"):
            block.sel(E=slice(1000.0, 2000.0))

    def test_sel_on_signal_axis_with_arbitrary_points(self) -> None:
        points = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        block = DataBlock(np.arange(5.0), [SignalAxis(points, "q", 0, "1/nm")])
        out = block.sel(q=slice(2.0, 8.0))
        np.testing.assert_array_equal(out.axis_obj("q").points, [2.0, 4.0, 8.0])
        assert isinstance(out.axis_obj("q"), SignalAxis)
        assert not isinstance(out.axis_obj("q"), LinearAxis)
        assert out.sel(q=5.0).compute() == pytest.approx(2.0)  # nearest to 5 is 4

    def test_sel_on_unordered_axis(self) -> None:
        points = np.array([5.0, 1.0, 3.0])
        block = DataBlock(
            np.array([10.0, 20.0, 30.0]), [UnorderedSignalAxis(points, "t", 0, "s")]
        )
        out = block.sel(t=slice(2.0, None))
        np.testing.assert_array_equal(out.axis_obj("t").points, [5.0, 3.0])
        np.testing.assert_array_equal(out.compute(), [10.0, 30.0])

    def test_sel_on_categorical_axis(self) -> None:
        cats = CategoricalAxis(["center", "amplitude", "fwhm"], "param", 1, "-")
        x = LinearAxis(0.0, 1.0, 4, "x", 0, "nm")
        data = np.arange(12.0).reshape(4, 3)
        block = DataBlock(data, [x, cats])
        one = block.sel(param="amplitude")
        assert [a.name for a in one.axes] == ["x"]
        np.testing.assert_array_equal(one.compute(), data[:, 1])
        two = block.sel(param=["fwhm", "center"])
        assert list(two.axis_obj("param").points) == ["fwhm", "center"]
        np.testing.assert_array_equal(two.compute(), data[:, [2, 0]])
        with pytest.raises(KeyError):
            block.sel(param="missing")

    def test_size_one_axes_are_kept(self) -> None:
        block, raw = spectrum_image()
        out = block.isel(x=slice(2, 3))
        assert out.axis_obj("x").size == 1
        assert out.shape == (4, 1, 50)
        np.testing.assert_allclose(out.compute(), raw[:, 2:3, :])

    def test_selection_preserves_roles_and_does_not_alias_axes(self) -> None:
        block, _ = spectrum_image()
        out = block.sel(x=slice(0.0, 1.0))
        assert [a.name for a in out.nav_axes] == ["y", "x"]
        out.axis_obj("E").unit = "keV"
        assert block.axis_obj("E").unit == "eV"

    def test_numpy_backed_block_stays_numpy(self) -> None:
        block, raw = spectrum_image(lazy=False)
        out = block.isel(y=1)
        assert isinstance(out.data, np.ndarray)
        np.testing.assert_array_equal(out.data, raw[1])


class TestReductions:
    def test_mean_over_roi_matches_numpy(self) -> None:
        """The linked-view case: spectrum from a rectangular scan ROI."""
        block, raw = spectrum_image()
        spectrum = block.sel(x=slice(0.5, 1.5), y=slice(0.5, 1.0)).mean_over("x", "y")
        assert [a.name for a in spectrum.axes] == ["E"]
        assert spectrum.axis_obj("E") == block.axis_obj("E")
        np.testing.assert_allclose(
            spectrum.compute(), raw[1:3, 1:4, :].mean(axis=(0, 1))
        )
        assert spectrum.metadata == {"beam_kev": 200.0}

    def test_sum_over(self) -> None:
        block, raw = spectrum_image()
        np.testing.assert_allclose(block.sum_over("E").compute(), raw.sum(axis=2))

    def test_skipna(self) -> None:
        raw = np.array([[1.0, np.nan], [3.0, 5.0]])
        block = DataBlock(
            raw,
            [LinearAxis(0, 1, 2, "a", 0, "-"), LinearAxis(0, 1, 2, "b", 1, "-")],
        )
        assert np.isnan(block.mean_over("b").compute()[0])
        np.testing.assert_allclose(
            block.mean_over("b", skipna=True).compute(), [1.0, 4.0]
        )

    def test_reduce_everything_gives_scalar_block(self) -> None:
        block, raw = spectrum_image()
        total = block.sum_over("y", "x", "E")
        assert total.dims == 0
        np.testing.assert_allclose(total.compute(), raw.sum())

    def test_reduction_is_lazy_for_dask(self) -> None:
        block, _ = spectrum_image()
        assert isinstance(block.mean_over("x").data, da.Array)

    def test_reduction_argument_errors(self) -> None:
        block, _ = spectrum_image()
        with pytest.raises(ValueError, match="at least one"):
            block.mean_over()
        with pytest.raises(KeyError):
            block.mean_over("nope")


class TestSubclasses:
    def test_subclass_with_custom_init_selects_into_plain_datablock(self) -> None:
        class Custom(DataBlock):
            def __init__(self, data, axes, name: str = "q", extra: str = "") -> None:
                super().__init__(data, axes, name)
                self.extra = extra

        block, _ = spectrum_image()
        custom = Custom(block.data, block.axes, "q", extra="kept")
        out = custom.isel(x=slice(1, 2))
        assert type(out) is DataBlock
        assert out.shape == (4, 1, 50)

    def test_subclass_with_default_init_keeps_its_type(self) -> None:
        class Plain(DataBlock):
            pass

        block, _ = spectrum_image()
        plain = Plain(block.data, block.axes, "q", "counts")
        assert type(plain.isel(x=0)) is Plain
        assert type(plain.mean_over("x")) is Plain
