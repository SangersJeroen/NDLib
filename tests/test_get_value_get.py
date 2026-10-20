"""DataBlock/Ensemble get and value_get: selector kinds, axis types and error handling."""

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from NDLib.AxesStructures import CategoricalAxis, SignalAxis, UnorderedSignalAxis
from NDLib.BaseStructures import DataBlock, Ensemble

RAW = np.arange(4 * 6 * 3, dtype=float).reshape(4, 6, 3)
U_POINTS = np.array([5.0, 3.0, 9.0, 1.0, 2.0, 7.0])
CATS = ["amplitude", "center", "new york"]


def _block(*, computed=False) -> DataBlock:
    axes = [
        SignalAxis(np.arange(4) * 1.0, "x", 0, "nm", navigate=True),
        UnorderedSignalAxis(U_POINTS.copy(), "u", 1, "eV"),
        CategoricalAxis(CATS, "param", 2, "-"),
    ]
    data = RAW.copy() if computed else da.from_array(RAW, chunks=2)
    return DataBlock(data, axes, quantity="signal", unit="counts")


class TestDataBlockGet:
    def test_inequality_and_axis_types(self):
        result = _block().get({"x": "x>=2", "u": "u>2"})

        keep_u = U_POINTS > 2
        np.testing.assert_array_equal(result.compute(), RAW[2:, keep_u])
        assert isinstance(result.axes[0], SignalAxis)
        assert isinstance(result.axes[1], UnorderedSignalAxis)
        np.testing.assert_array_equal(result.axes[1].points, U_POINTS[keep_u])
        assert (result.quantity, result.unit) == ("signal", "counts")
        assert [ax.index_in_array for ax in result.axes] == [0, 1, 2]

    def test_categorical_expression(self):
        result = _block().get({"param": "param!=center"})

        np.testing.assert_array_equal(result.compute(), RAW[:, :, [0, 2]])
        assert list(result.axes[2].points) == ["amplitude", "new york"]

    def test_quoted_categorical_value_with_space(self):
        result = _block().get({"param": "param=='new york'"})

        # one category left: the categorical axis is squeezed and names the quantity
        assert result.shape == (4, 6)
        assert result.quantity == "param"
        np.testing.assert_array_equal(result.compute(), RAW[:, :, 2])

    def test_shorthand_and_scientific_notation(self):
        result = _block().get({"x": ">1e0", "u": "<=.5e1"})

        np.testing.assert_array_equal(result.compute(), RAW[2:, U_POINTS <= 5])

    def test_numeric_equality_and_inequality(self):
        eq = _block().get({"u": "u==3"})
        ne = _block().get({"u": "u!=3"})

        assert eq.shape == (4, 3)  # a single point left: u is squeezed away
        assert ne.shape == (4, 5, 3)
        np.testing.assert_array_equal(eq.compute(), RAW[:, 1])

    @pytest.mark.parametrize(
        "make_mask",
        [
            lambda: np.array([True, False, True, True, False, False]),
            lambda: [True, False, True, True, False, False],
            lambda: (True, False, True, True, False, False),
            lambda: da.from_array(np.array([True, False, True, True, False, False])),
        ],
        ids=["ndarray", "list", "tuple", "dask"],
    )
    def test_boolean_masks(self, make_mask):
        result = _block().get({"u": make_mask()})

        keep = np.array([True, False, True, True, False, False])
        np.testing.assert_array_equal(result.compute(), RAW[:, keep])
        np.testing.assert_array_equal(result.axes[1].points, U_POINTS[keep])

    def test_selecting_nothing_on_a_signal_axis(self):
        result = _block().get({"x": "x>100"})

        assert result.shape == (0, 6, 3)
        assert result.axes[0].size == 0

    @pytest.mark.parametrize("indexing", [{"u": "u>100"}, {"param": "param==nothing"}])
    def test_selecting_nothing_elsewhere_is_an_error(self, indexing):
        with pytest.raises(ValueError, match="selection on axis"):
            _block().get(indexing)

    def test_empty_indexing_returns_an_equal_copy(self):
        db = _block()

        result = db.get({})

        assert result is not db
        assert result.axes[0] is not db.axes[0]
        np.testing.assert_array_equal(result.compute(), RAW)

    def test_numpy_backed_block(self):
        result = _block(computed=True).get({"x": "x<2"})

        np.testing.assert_array_equal(result.compute(), RAW[:2])

    def test_source_is_untouched(self):
        db = _block()

        db.get({"x": "x>1", "u": "u<4"})

        assert db.shape == (4, 6, 3)
        assert [ax.size for ax in db.axes] == [4, 6, 3]

    @pytest.mark.parametrize(
        ("selector", "error", "match"),
        [
            ([0, 1, 1, 0, 1, 0], TypeError, "should be boolean"),
            (np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]), TypeError, "should be boolean"),
            ([True, False], ValueError, "doesn't match axis 'u' size"),
            (np.ones((6, 1), dtype=bool), ValueError, "should be 1-D"),
            (3, TypeError, "expression string or a boolean array"),
            (None, TypeError, "expression string or a boolean array"),
            ({"a": 1}, TypeError, "expression string or a boolean array"),
            ("u=3", ValueError, "use '=='"),
            ("y>3", ValueError, "doesn't match axis 'u'"),
            ("u>>3", ValueError, "Invalid inequality"),
        ],
    )
    def test_bad_selectors(self, selector, error, match):
        with pytest.raises(error, match=match):
            _block().get({"u": selector})

    def test_unknown_axis(self):
        with pytest.raises(ValueError, match="Axis 'q' not found"):
            _block().get({"q": "q>1"})

    def test_indexing_must_be_a_dict(self):
        with pytest.raises(TypeError, match="Expected dict"):
            _block().get([("x", "x>1")])


class TestDataBlockValueGet:
    def test_matches_get(self):
        db = _block()
        indexing = {"x": "x>0", "u": "u<8", "param": "param!=center"}

        data, axes = db.value_get(indexing)
        expected = db.get(indexing)

        np.testing.assert_array_equal(np.asarray(data), expected.compute())
        assert list(axes) == ["x", "u", "param"]
        for axis in expected.axes:
            np.testing.assert_array_equal(axes[axis.name], axis.points)

    def test_empty_indexing(self):
        db = _block()

        data, axes = db.value_get({})

        np.testing.assert_array_equal(np.asarray(data), RAW)
        np.testing.assert_array_equal(axes["u"], U_POINTS)

    def test_boolean_mask(self):
        data, axes = _block().value_get({"u": [True, False, False, False, False, True]})

        np.testing.assert_array_equal(np.asarray(data), RAW[:, [0, 5]])
        np.testing.assert_array_equal(axes["u"], U_POINTS[[0, 5]])

    def test_indexing_must_be_a_dict(self):
        with pytest.raises(TypeError, match="Expected dict"):
            _block().value_get([1])

    def test_unknown_axis(self):
        with pytest.raises(ValueError, match="Axis 'q' not found"):
            _block().value_get({"q": "q>1"})


def _ensemble(*, as_pandas=False) -> Ensemble:
    x, u, p = np.meshgrid(np.arange(4.0), U_POINTS, np.arange(3), indexing="ij")
    df = pd.DataFrame({
        "x": x.ravel(),
        "u": u.ravel(),
        "param": np.array(CATS)[p.ravel()],
        "v": np.arange(x.size, dtype=float),
    })
    # Shuffle the rows so the order of appearance differs from the axis order
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    axes = [
        SignalAxis(np.arange(4.0), "x", 0, "nm", navigate=True),
        UnorderedSignalAxis(U_POINTS.copy(), "u", 1, "eV"),
        CategoricalAxis(CATS, "param", 2, "-"),
    ]
    data = df if as_pandas else dd.from_pandas(df, npartitions=3)
    return Ensemble(data, axes, quantity="v", unit="counts")


def _frame(ensemble: Ensemble) -> pd.DataFrame:
    data = ensemble.data
    return data.compute() if isinstance(data, dd.DataFrame) else data


class TestEnsembleGet:
    @pytest.mark.parametrize("as_pandas", [False, True])
    def test_inequality_keeps_axis_order_and_types(self, as_pandas):
        result = _ensemble(as_pandas=as_pandas).get({"x": "x>=1", "u": "u>2"})

        frame = _frame(result)
        assert set(frame["x"]) == {1.0, 2.0, 3.0}
        assert frame["u"].min() > 2
        # points stay in the order of the original axis, not of appearance in the rows
        np.testing.assert_array_equal(result.axes[0].points, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.axes[1].points, [5.0, 3.0, 9.0, 7.0])
        assert isinstance(result.axes[0], SignalAxis)
        assert isinstance(result.axes[1], UnorderedSignalAxis)
        assert (result.quantity, result.unit) == ("v", "counts")

    def test_categorical_expression(self):
        result = _ensemble().get({"param": "param=='new york'"})

        assert set(_frame(result)["param"]) == {"new york"}
        assert list(result.axes[2].points) == ["new york"]

    def test_numeric_equality(self):
        result = _ensemble().get({"u": "u==3"})

        assert set(_frame(result)["u"]) == {3.0}

    def test_boolean_mask_selects_points_of_the_axis(self):
        mask = [True, False, False, True, False, False]

        result = _ensemble().get({"u": mask})

        assert set(_frame(result)["u"]) == {5.0, 1.0}
        np.testing.assert_array_equal(result.axes[1].points, [5.0, 1.0])

    def test_empty_indexing_returns_an_equal_copy(self):
        ensemble = _ensemble()

        result = ensemble.get({})

        assert result is not ensemble
        assert result.axes[0] is not ensemble.axes[0]
        assert len(_frame(result)) == len(_frame(ensemble))

    def test_selecting_nothing_on_a_signal_axis(self):
        result = _ensemble().get({"x": "x>100"})

        assert len(_frame(result)) == 0
        assert result.axes[0].size == 0

    def test_source_axes_are_untouched(self):
        ensemble = _ensemble()

        ensemble.get({"x": "x>1"})

        assert ensemble.axes[0].size == 4

    @pytest.mark.parametrize(
        ("selector", "error", "match"),
        [
            ([1, 0, 1, 0, 1, 0], TypeError, "should be boolean"),
            ([True, False], ValueError, "doesn't match axis 'u' size"),
            (3.5, TypeError, "expression string or a boolean array"),
            ("u=3", ValueError, "use '=='"),
            ("y>3", ValueError, "doesn't match axis 'u'"),
        ],
    )
    def test_bad_selectors(self, selector, error, match):
        with pytest.raises(error, match=match):
            _ensemble().get({"u": selector})

    def test_unknown_axis_and_non_dict(self):
        with pytest.raises(ValueError, match="Axis 'q' not found"):
            _ensemble().get({"q": "q>1"})
        with pytest.raises(TypeError, match="Expected dict"):
            _ensemble().get("x>1")


class TestEnsembleValueGet:
    @pytest.mark.parametrize("as_pandas", [False, True])
    def test_matches_get(self, as_pandas):
        ensemble = _ensemble(as_pandas=as_pandas)
        indexing = {"x": "x<3", "u": [True, False, True, False, True, False]}

        values, axes = ensemble.value_get(indexing)
        expected = _frame(ensemble.get(indexing))

        np.testing.assert_array_equal(np.asarray(values), expected["v"].to_numpy())
        assert list(axes) == ["x", "u", "param"]
        np.testing.assert_array_equal(np.asarray(axes["u"]), expected["u"].to_numpy())

    def test_boolean_mask_means_the_same_as_in_get(self):
        ensemble = _ensemble()
        mask = [False, True, False, False, False, True]

        values, axes = ensemble.value_get({"u": mask})

        assert set(np.asarray(axes["u"])) == {3.0, 7.0}
        assert len(np.asarray(values)) == 4 * 2 * 3

    def test_empty_indexing(self):
        values, axes = _ensemble().value_get({})

        assert len(np.asarray(values)) == 4 * 6 * 3
        assert list(axes) == ["x", "u", "param"]

    def test_bad_input(self):
        with pytest.raises(TypeError, match="Expected dict"):
            _ensemble().value_get(["x"])
        with pytest.raises(ValueError, match="Axis 'q' not found"):
            _ensemble().value_get({"q": "q>1"})
        with pytest.raises(ValueError, match="doesn't match axis 'u' size"):
            _ensemble().value_get({"u": [True, False]})
