"""Tests for utility functions."""

from datetime import datetime

import pytest

from goodseed.utils import (
    deserialize_value,
    flatten_dict,
    generate_run_name,
    normalize_path,
    serialize_value,
)


class TestGenerateRunName:
    def test_format(self):
        name = generate_run_name()
        parts = name.split("-")
        assert len(parts) == 2

    def test_unique(self):
        names = {generate_run_name() for _ in range(50)}
        # With 36 adjectives * 42 animals = 1512 combos,
        # 50 draws should almost never collide, but allow a few.
        assert len(names) >= 40


class TestSerializeValue:
    def test_none(self):
        assert serialize_value(None) == ("null", "")

    def test_bool_true(self):
        assert serialize_value(True) == ("bool", "true")

    def test_bool_false(self):
        assert serialize_value(False) == ("bool", "false")

    def test_int(self):
        assert serialize_value(42) == ("int", "42")

    def test_float(self):
        assert serialize_value(3.14) == ("float", "3.14")

    def test_string(self):
        assert serialize_value("hello") == ("str", "hello")

    def test_empty_string(self):
        assert serialize_value("") == ("str", "")

    def test_datetime(self):
        dt = datetime(2025, 1, 15, 12, 30, 0)
        tag, val = serialize_value(dt)
        assert tag == "datetime"
        assert "2025-01-15" in val

    def test_unsupported_becomes_str(self):
        assert serialize_value([1, 2, 3]) == ("str", "[1, 2, 3]")


class TestDeserializeValue:
    def test_null(self):
        assert deserialize_value("null", "") is None

    def test_null_with_none_raw(self):
        assert deserialize_value("null", None) is None

    def test_non_null_with_none_raw(self):
        # If the raw value is None but type_tag isn't null, treat as None
        assert deserialize_value("str", None) is None

    def test_bool_true(self):
        assert deserialize_value("bool", "true") is True

    def test_bool_false(self):
        assert deserialize_value("bool", "false") is False

    def test_bool_case_insensitive(self):
        assert deserialize_value("bool", "True") is True

    def test_int(self):
        assert deserialize_value("int", "42") == 42

    def test_float(self):
        assert deserialize_value("float", "3.14") == 3.14

    def test_string(self):
        assert deserialize_value("str", "hello") == "hello"

    def test_empty_string(self):
        """An empty string with type_tag 'str' should return '', not None."""
        assert deserialize_value("str", "") == ""

    def test_datetime(self):
        result = deserialize_value("datetime", "2025-01-15T12:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2025

    def test_unknown_tag_passthrough(self):
        assert deserialize_value("unknown", "data") == "data"


class TestSerializeRoundtrip:
    @pytest.mark.parametrize("value", [
        None, True, False, 0, 42, -1, 3.14, 0.0, "hello", "",
    ])
    def test_roundtrip(self, value):
        tag, raw = serialize_value(value)
        result = deserialize_value(tag, raw)
        assert result == value
        assert type(result) is type(value)


class TestFlattenDict:
    def test_flat_dict(self):
        result = flatten_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = flatten_dict({"model": {"hidden": 256, "layers": 4}})
        assert result == {"model/hidden": 256, "model/layers": 4}

    def test_deeply_nested(self):
        result = flatten_dict({"a": {"b": {"c": 1}}})
        assert result == {"a/b/c": 1}

    def test_list_values(self):
        result = flatten_dict({"tags": [10, 20]})
        assert result == {"tags/0": 10, "tags/1": 20}

    def test_list_of_dicts(self):
        result = flatten_dict({"layers": [{"size": 64}, {"size": 32}]})
        assert result == {"layers/0/size": 64, "layers/1/size": 32}

    def test_custom_separator(self):
        result = flatten_dict({"a": {"b": 1}}, sep=".")
        assert result == {"a.b": 1}

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            flatten_dict({"x": object()})

    def test_cast_unsupported(self):
        result = flatten_dict({"x": object()}, cast_unsupported=True)
        assert isinstance(result["x"], str)

    def test_none_value(self):
        result = flatten_dict({"x": None})
        assert result == {"x": None}

    def test_mixed_types(self):
        result = flatten_dict({
            "flag": True,
            "count": 42,
            "rate": 0.1,
            "name": "test",
            "empty": None,
        })
        assert len(result) == 5


class TestNormalizePath:
    def test_no_change(self):
        assert normalize_path("train/loss") == "train/loss"

    def test_strip_leading(self):
        assert normalize_path("/train/loss") == "train/loss"

    def test_strip_trailing(self):
        assert normalize_path("train/loss/") == "train/loss"

    def test_strip_both(self):
        assert normalize_path("/train/loss/") == "train/loss"

    def test_simple_key(self):
        assert normalize_path("loss") == "loss"
