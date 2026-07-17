"""Safe codec for GPU tool RPC messages.

Requests use JSON and responses use MessagePack so large arrays remain binary.
Only the data types needed by the GPU tools are supported. In particular,
decoding never imports a type named by the peer or invokes object hooks.
"""

import base64
import binascii
import io
import math
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import msgpack
import orjson
from PIL import Image

from spatial_agent.gpu_models import types as gpu_types


PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_NESTING = 64
_TYPE_KEY = "__spatialclaw_type__"


class RPCProtocolError(ValueError):
    """Raised when an RPC message does not match the wire schema."""


_DATACLASS_TYPES = {
    name: value
    for name, value in vars(gpu_types).items()
    if isinstance(value, type)
    and value.__module__ == gpu_types.__name__
    and is_dataclass(value)
}


def _encode_value(value: Any, depth: int = 0, binary_data: bool = False) -> Any:
    if depth > _MAX_NESTING:
        raise RPCProtocolError("RPC value exceeds the maximum nesting depth")

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        label = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {_TYPE_KEY: "float", "value": label}
    if isinstance(value, np.generic):
        return _encode_value(value.item(), depth + 1, binary_data)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise RPCProtocolError("Object arrays are not supported by the RPC codec")
        # memoryview.cast rejects zero-length multidimensional views. Empty
        # detections are valid tool results, so encode their payload explicitly.
        data = memoryview(array).cast("B") if array.nbytes else b""
        return {
            _TYPE_KEY: "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": data if binary_data else base64.b64encode(data).decode("ascii"),
        }
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        try:
            value.save(buffer, format="PNG")
        except OSError:
            # GPU vision models consume RGB pixels; normalize uncommon PIL
            # modes (for example CMYK and F) that PNG cannot represent.
            buffer.seek(0)
            buffer.truncate()
            value.convert("RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        return {
            _TYPE_KEY: "image",
            "data": data if binary_data else base64.b64encode(data).decode("ascii"),
        }
    if is_dataclass(value):
        type_name = type(value).__name__
        if _DATACLASS_TYPES.get(type_name) is not type(value):
            raise RPCProtocolError(f"Unsupported RPC dataclass: {type(value).__name__}")
        return {
            _TYPE_KEY: "dataclass",
            "name": type_name,
            "fields": {
                field.name: _encode_value(
                    getattr(value, field.name), depth + 1, binary_data
                )
                for field in fields(value)
            },
        }
    if isinstance(value, (list, tuple)):
        return [_encode_value(item, depth + 1, binary_data) for item in value]
    if isinstance(value, dict):
        if _TYPE_KEY in value:
            raise RPCProtocolError(f"{_TYPE_KEY!r} is reserved by the RPC codec")
        if not all(isinstance(key, str) for key in value):
            raise RPCProtocolError("RPC mappings must have string keys")
        return {
            key: _encode_value(item, depth + 1, binary_data)
            for key, item in value.items()
        }
    raise RPCProtocolError(f"Unsupported RPC value type: {type(value).__name__}")


def _decode_base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise RPCProtocolError("Encoded binary data must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RPCProtocolError("Invalid base64 data in RPC message") from exc


def _decode_value(value: Any, depth: int = 0, binary_data: bool = False) -> Any:
    if depth > _MAX_NESTING:
        raise RPCProtocolError("RPC value exceeds the maximum nesting depth")

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_decode_value(item, depth + 1, binary_data) for item in value]
    if not isinstance(value, dict):
        raise RPCProtocolError("Invalid value in RPC message")

    type_name = value.get(_TYPE_KEY)
    if type_name is None:
        if not all(isinstance(key, str) for key in value):
            raise RPCProtocolError("RPC mappings must have string keys")
        return {
            key: _decode_value(item, depth + 1, binary_data)
            for key, item in value.items()
        }

    if type_name == "float":
        if set(value) != {_TYPE_KEY, "value"}:
            raise RPCProtocolError("Invalid encoded float")
        floats = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
        try:
            return floats[value["value"]]
        except (KeyError, TypeError) as exc:
            raise RPCProtocolError("Invalid encoded float") from exc

    if type_name == "ndarray":
        if set(value) != {_TYPE_KEY, "dtype", "shape", "data"}:
            raise RPCProtocolError("Invalid encoded ndarray")
        shape = value["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) > 16
            or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
        ):
            raise RPCProtocolError("Invalid ndarray shape")
        if not isinstance(value["dtype"], str):
            raise RPCProtocolError("Invalid ndarray dtype")
        try:
            dtype = np.dtype(value["dtype"])
        except (TypeError, ValueError) as exc:
            raise RPCProtocolError("Invalid ndarray dtype") from exc
        if dtype.hasobject:
            raise RPCProtocolError("Object arrays are not supported by the RPC codec")
        if binary_data:
            data = value["data"]
            if not isinstance(data, bytes):
                raise RPCProtocolError("Encoded ndarray data must be bytes")
        else:
            data = _decode_base64(value["data"])
        expected_size = math.prod(shape) * dtype.itemsize
        if len(data) != expected_size:
            raise RPCProtocolError("Encoded ndarray size does not match its shape")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    if type_name == "image":
        if set(value) != {_TYPE_KEY, "data"}:
            raise RPCProtocolError("Invalid encoded image")
        try:
            if binary_data:
                data = value["data"]
                if not isinstance(data, bytes):
                    raise RPCProtocolError("Encoded image data must be bytes")
            else:
                data = _decode_base64(value["data"])
            image = Image.open(io.BytesIO(data))
            if image.format != "PNG":
                raise RPCProtocolError("RPC images must use PNG encoding")
            image.load()
            return image
        except (Image.DecompressionBombError, OSError, ValueError) as exc:
            raise RPCProtocolError("Invalid image data in RPC message") from exc

    if type_name == "dataclass":
        if set(value) != {_TYPE_KEY, "name", "fields"}:
            raise RPCProtocolError("Invalid encoded dataclass")
        cls = _DATACLASS_TYPES.get(value["name"])
        encoded_fields = value["fields"]
        if cls is None or not isinstance(encoded_fields, dict):
            raise RPCProtocolError("RPC dataclass type is not allowed")
        expected_fields = {field.name for field in fields(cls)}
        if set(encoded_fields) != expected_fields:
            raise RPCProtocolError("RPC dataclass fields do not match its schema")
        return cls(**{
            key: _decode_value(item, depth + 1, binary_data)
            for key, item in encoded_fields.items()
        })

    raise RPCProtocolError("Unknown encoded RPC value type")


def _dump_json(message: dict) -> bytes:
    try:
        return orjson.dumps(message)
    except (TypeError, orjson.JSONEncodeError) as exc:
        raise RPCProtocolError("RPC message is not JSON serializable") from exc


def encode_request(deployment: str, method: str, kwargs: dict) -> bytes:
    """Encode and validate one client request."""
    if not isinstance(deployment, str) or not deployment or len(deployment) > 128:
        raise RPCProtocolError("deployment must be a non-empty string")
    if not isinstance(method, str) or not method or len(method) > 128:
        raise RPCProtocolError("method must be a non-empty string")
    if not isinstance(kwargs, dict):
        raise RPCProtocolError("kwargs must be a mapping")
    return _dump_json({
        "version": PROTOCOL_VERSION,
        "deployment": deployment,
        "method": method,
        "kwargs": _encode_value(kwargs),
    })


def decode_request(body: bytes) -> dict:
    """Decode a request using a strict top-level schema."""
    try:
        message = orjson.loads(body)
    except orjson.JSONDecodeError as exc:
        raise RPCProtocolError("Request body must be valid JSON") from exc
    if not isinstance(message, dict) or set(message) != {
        "version", "deployment", "method", "kwargs"
    }:
        raise RPCProtocolError("Request does not match the RPC schema")
    if message["version"] != PROTOCOL_VERSION:
        raise RPCProtocolError("Unsupported RPC protocol version")
    deployment = message["deployment"]
    method = message["method"]
    if not isinstance(deployment, str) or not deployment or len(deployment) > 128:
        raise RPCProtocolError("deployment must be a non-empty string")
    if not isinstance(method, str) or not method or len(method) > 128:
        raise RPCProtocolError("method must be a non-empty string")
    kwargs = _decode_value(message["kwargs"])
    if not isinstance(kwargs, dict):
        raise RPCProtocolError("kwargs must be a mapping")
    return {"deployment": deployment, "method": method, "kwargs": kwargs}


def encode_success(result: Any) -> bytes:
    try:
        return msgpack.packb({
            "version": PROTOCOL_VERSION,
            "ok": True,
            "result": _encode_value(result, binary_data=True),
        }, use_bin_type=True)
    except (TypeError, ValueError) as exc:
        raise RPCProtocolError(f"RPC response is not serializable: {exc}") from exc


def encode_error(error: Exception) -> bytes:
    try:
        return msgpack.packb({
            "version": PROTOCOL_VERSION,
            "ok": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }, use_bin_type=True)
    except (TypeError, ValueError) as exc:
        raise RPCProtocolError(
            f"RPC error response is not serializable: {exc}"
        ) from exc


def decode_response(body: bytes) -> Any:
    """Decode a server response, raising a local exception for remote errors."""
    try:
        message = msgpack.unpackb(body, raw=False, strict_map_key=True)
    except (msgpack.UnpackException, ValueError) as exc:
        raise RPCProtocolError("Response body must be valid MessagePack") from exc
    if not isinstance(message, dict) or message.get("version") != PROTOCOL_VERSION:
        raise RPCProtocolError("Response does not match the RPC schema")
    if message.get("ok") is True and set(message) == {"version", "ok", "result"}:
        return _decode_value(message["result"], binary_data=True)
    if message.get("ok") is False and set(message) == {"version", "ok", "error"}:
        error = message["error"]
        if not isinstance(error, dict) or set(error) != {"type", "message"}:
            raise RPCProtocolError("Response error does not match the RPC schema")
        if not isinstance(error["type"], str) or not isinstance(error["message"], str):
            raise RPCProtocolError("Response error fields must be strings")
        error_types = {
            "AssertionError": AssertionError,
            "RPCProtocolError": ValueError,
            "ValueError": ValueError,
            "TypeError": TypeError,
        }
        raise error_types.get(error["type"], RuntimeError)(error["message"])
    raise RPCProtocolError("Response does not match the RPC schema")
