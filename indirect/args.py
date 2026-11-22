"""Indirect-draw argument packing exercise.

A command is a mapping with exactly these fields::

    {"count": int, "instance": int, "first": int, "base_vertex": int}

All four values are unsigned 32-bit integers. On the wire a command occupies
five little-endian ``uint32`` words (20 bytes): ``count``, ``instance``,
``first``, ``base_vertex``, and a zero reserved word.

Global byte contract:

* parameter buffers start at 4-byte-aligned addresses;
* one packed command is exactly ``DRAW_SIZE == 20`` bytes;
* a stride is a byte count, at least ``MIN_STRIDE == 20`` and a multiple of 4;
* ``first`` and ``base_vertex`` offsets are added while ``count`` is unchanged;
* computed byte lengths and per-block byte budgets are compared with at most
  1 byte of tolerance, but one packed command and an exact round-trip are
  never approximate.

Malformed commands, buffers, strides, alignments, and batches raise
``IndirectError``; no partial result is returned for a failed batch.
"""

import struct
from collections.abc import Mapping, Sequence


class IndirectError(Exception):
    """Raised when indirect-draw arguments or a batch violate the contract."""


DRAW_STRUCT = struct.Struct("<5I")
DRAW_SIZE = DRAW_STRUCT.size
MIN_STRIDE = DRAW_SIZE
UINT32_MAX = (1 << 32) - 1

_REQUIRED_FIELDS = ("count", "instance", "first", "base_vertex")


def _is_int(value) -> bool:
    """True for real integers; booleans are not valid argument values."""
    return isinstance(value, int) and not isinstance(value, bool)


def _checked_command(cmd: Mapping[str, int]) -> dict:
    """Validate one command mapping and return its four uint32 fields."""
    if not isinstance(cmd, Mapping):
        raise IndirectError(f"command must be a mapping, got {type(cmd).__name__}")
    keys = set(cmd)
    expected = set(_REQUIRED_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise IndirectError(f"command is missing fields: {sorted(missing)}")
    if extra:
        raise IndirectError(f"command has unexpected fields: {sorted(extra)}")
    values = {}
    for field in _REQUIRED_FIELDS:
        value = cmd[field]
        if not _is_int(value):
            raise IndirectError(
                f"{field} must be an integer, got {value!r}"
            )
        if not 0 <= value <= UINT32_MAX:
            raise IndirectError(
                f"{field}={value} is outside the uint32 range"
            )
        values[field] = value
    return values


def _checked_stride(stride: int) -> int:
    """Validate a byte stride: an integer >= MIN_STRIDE and a multiple of 4."""
    if not _is_int(stride):
        raise IndirectError(f"stride must be an integer, got {stride!r}")
    if stride < MIN_STRIDE:
        raise IndirectError(
            f"stride {stride} is smaller than one command ({MIN_STRIDE} bytes)"
        )
    if stride % 4 != 0:
        raise IndirectError(f"stride {stride} is not a multiple of 4")
    return stride


def pack_draw(cmd: Mapping[str, int]) -> bytes:
    """Pack one indirect-draw command into exactly 20 bytes.

    ``cmd`` must contain exactly the four contract fields ``count``,
    ``instance``, ``first``, and ``base_vertex``. Each value is an integer in
    ``0 .. 2**32 - 1``. The result is five little-endian uint32 words:

    ``count, instance, first, base_vertex, 0``

    The fifth word is reserved and is always zero, giving a 20-byte command
    suitable for a 4-byte-aligned GPU parameter buffer. Missing or extra
    fields, non-integer values, and values outside the uint32 range raise
    ``IndirectError``.
    """
    values = _checked_command(cmd)
    return DRAW_STRUCT.pack(
        values["count"],
        values["instance"],
        values["first"],
        values["base_vertex"],
        0,
    )


def unpack_draw(data: bytes) -> dict:
    """Parse exactly one 20-byte indirect-draw command.

    ``data`` is a bytes-like object of exactly ``DRAW_SIZE`` (20) bytes. The
    five little-endian uint32 words are decoded as ``count``, ``instance``,
    ``first``, ``base_vertex``, and a reserved word. The reserved word must
    be zero.

    Return exactly the four contract fields as a plain dict::

        {"count": int, "instance": int, "first": int, "base_vertex": int}

    A length shorter or longer than 20 bytes, or a non-zero reserved word,
    raises ``IndirectError``. The parser never reads into an adjacent command
    in a larger buffer.
    """
    data = bytes(data)
    if len(data) != DRAW_SIZE:
        raise IndirectError(
            f"packed command must be exactly {DRAW_SIZE} bytes, "
            f"got {len(data)}"
        )
    count, instance, first, base_vertex, reserved = DRAW_STRUCT.unpack(data)
    if reserved != 0:
        raise IndirectError("reserved word must be zero")
    return {
        "count": count,
        "instance": instance,
        "first": first,
        "base_vertex": base_vertex,
    }


def batch_writes(cmds: Sequence[Mapping[str, int]], stride: int) -> bytes:
    """Pack several commands into one buffer with a fixed byte stride.

    ``cmds`` is a sequence of command mappings as documented by
    :func:`pack_draw`. Commands are laid out in input order, command ``i``
    starting at byte offset ``i * stride``; each 20-byte command occupies the
    beginning of its slot and the remaining slot bytes are zero padding.
    Therefore every command starts on a 4-byte boundary when the buffer is
    4-byte-aligned.

    ``stride`` must be an integer >= ``MIN_STRIDE`` (20) and a multiple of 4.
    The returned buffer has exactly ``len(cmds) * stride`` bytes. An empty
    sequence returns ``b""``. An invalid stride or any invalid command raises
    ``IndirectError`` before any output is returned, so a short/overlapping
    stride can never write into the next command's slot.
    """
    stride = _checked_stride(stride)
    # Validate every command before building anything: all-or-nothing.
    packed = [pack_draw(cmd) for cmd in cmds]
    if not packed:
        return b""
    buffer = bytearray(len(packed) * stride)
    for index, blob in enumerate(packed):
        start = index * stride
        buffer[start : start + DRAW_SIZE] = blob
    return bytes(buffer)


def align_stride(size: int, alignment: int) -> int:
    """Return the smallest multiple of *alignment* that is >= *size*.

    ``size`` is a non-negative byte count and ``alignment`` is a positive
    power of two (1, 2, 4, 8, ...). The result is rounded **up**, never down:
    ``align_stride(20, 16) == 32`` (not 16), so a 20-byte command cannot be
    truncated into the previous 16-byte slot. ``align_stride(0, alignment)``
    is 0.

    A negative or non-integer size, and an alignment that is zero, negative,
    or not a power of two, raises ``IndirectError`` instead of dividing by
    zero or silently truncating.
    """
    if not _is_int(size) or size < 0:
        raise IndirectError(f"size must be a non-negative integer, got {size!r}")
    if not _is_int(alignment) or alignment <= 0 or (alignment & (alignment - 1)):
        raise IndirectError(
            f"alignment must be a positive power of two, got {alignment!r}"
        )
    return -(-size // alignment) * alignment


def apply_base_offset(
    cmds: Sequence[Mapping[str, int]], offset: int
) -> list:
    """Return copies of *cmds* with a base offset added to index fields.

    ``offset`` is a non-negative integer. For every command, ``offset`` is
    added to both ``first`` and ``base_vertex``; ``count`` and ``instance``
    are copied unchanged. The result is a new list in the original order,
    and the input commands are not mutated.

    If ``offset`` is negative, or if either addition would exceed
    ``UINT32_MAX``, raise ``IndirectError`` before returning any adjusted
    command. This prevents index wrap-around and keeps every draw inside the
    valid argument range.
    """
    if not _is_int(offset) or offset < 0:
        raise IndirectError(
            f"offset must be a non-negative integer, got {offset!r}"
        )
    # Validate everything before returning anything: all-or-nothing.
    values = [_checked_command(cmd) for cmd in cmds]
    adjusted = []
    for value in values:
        first = value["first"] + offset
        base_vertex = value["base_vertex"] + offset
        if first > UINT32_MAX or base_vertex > UINT32_MAX:
            raise IndirectError(
                "base offset would overflow the uint32 argument range"
            )
        adjusted.append(
            {
                "count": value["count"],
                "instance": value["instance"],
                "first": first,
                "base_vertex": base_vertex,
            }
        )
    return adjusted


def split_by_budget(
    cmds: Sequence[Mapping[str, int]], max_bytes: int
) -> list[list]:
    """Split *cmds* into ordered chunks that each fit a byte budget.

    ``max_bytes`` is the maximum packed size of one chunk in bytes and must be
    an integer >= ``MIN_STRIDE`` (20); otherwise raise ``IndirectError``.
    Each command costs 20 bytes, so a chunk may contain at most
    ``max_bytes // 20`` commands. Chunks are contiguous in input order, do
    not duplicate or drop commands, and every chunk's packed byte size
    (``20 * len(chunk)``) is <= ``max_bytes`` (with the stated 1-byte
    comparison tolerance). An empty command sequence returns an empty list.

    This is a byte-budget split, not a command-count split: 5 commands with
    ``max_bytes=45`` yield chunk sizes ``[2, 2, 1]`` rather than 5 or 3.
    """
    if not _is_int(max_bytes) or max_bytes < MIN_STRIDE:
        raise IndirectError(
            f"max_bytes must be an integer >= {MIN_STRIDE}, got {max_bytes!r}"
        )
    per_chunk = max_bytes // DRAW_SIZE
    chunks = []
    for start in range(0, len(cmds), per_chunk):
        chunks.append(list(cmds[start : start + per_chunk]))
    return chunks


def validate_args(cmds: Sequence[Mapping[str, int]]) -> list[str]:
    """Return all reported problems with *cmds*; empty means valid.

    Every command is checked for the four required contract fields and for:

    * ``count`` <= 0 (zero draws no geometry and is rejected);
    * any of ``count``, ``instance``, ``first``, ``base_vertex`` outside the
      uint32 range, including negative offsets;
    * an optional per-command ``stride`` below 20 or not a multiple of 4
      (the default is 20).

    The result is a list of human-readable strings in deterministic input
    order; one command may contribute more than one issue. Validation never
    mutates the commands and never raises ``IndirectError`` for bad data.
    """
    issues = []
    for index, cmd in enumerate(cmds):
        if not isinstance(cmd, Mapping):
            issues.append(f"command {index}: not a mapping")
            continue
        for field in _REQUIRED_FIELDS:
            if field not in cmd:
                issues.append(f"command {index}: missing field {field!r}")
        for field in _REQUIRED_FIELDS:
            if field not in cmd:
                continue
            value = cmd[field]
            if not _is_int(value):
                issues.append(
                    f"command {index}: {field} is not an integer: {value!r}"
                )
                continue
            if value < 0:
                issues.append(
                    f"command {index}: negative offset {field}={value}"
                )
            elif value > UINT32_MAX:
                issues.append(
                    f"command {index}: {field}={value} exceeds uint32 range"
                )
        count = cmd.get("count")
        if _is_int(count) and count <= 0:
            issues.append(
                f"command {index}: count must be positive, got {count}"
            )
        if "stride" in cmd:
            stride = cmd["stride"]
            if (
                not _is_int(stride)
                or stride < MIN_STRIDE
                or stride % 4 != 0
            ):
                issues.append(
                    f"command {index}: stride must be an integer >= "
                    f"{MIN_STRIDE} and a multiple of 4, got {stride!r}"
                )
    return issues


def dispatch_batch(
    buffers: Sequence[Sequence[Mapping[str, int]]], config: Mapping[str, int]
) -> dict:
    """Build several independent indirect parameter buffers as one batch.

    ``buffers`` is a sequence of command sequences; each inner sequence is one
    independent parameter buffer. ``config`` supplies ``stride`` (the byte
    distance between commands, >= 20 and a multiple of 4) and ``max_bytes``
    (the per-buffer byte budget, >= 20).

    Return::

        {
            "batches": [bytes, ...],  # one independent buffer per input buffer
            "data": bytes,            # batches concatenated in input order
            "count": int,             # total number of commands
            "stride": int,            # validated stride used by every batch
        }

    Every input buffer is validated before anything is built. If any buffer
    is invalid, a command is invalid, a buffer exceeds ``max_bytes``, or the
    configuration is invalid, raise ``IndirectError`` and return nothing:
    the batch is all-or-nothing and no buffer is shared or partially written
    between batches. Buffer and command order is preserved on success; empty
    input returns ``{"batches": [], "data": b"", "count": 0, ...}``.
    """
    if not isinstance(config, Mapping):
        raise IndirectError("config must be a mapping")
    for key in ("stride", "max_bytes"):
        if key not in config:
            raise IndirectError(f"config is missing {key!r}")
    stride = _checked_stride(config["stride"])
    max_bytes = config["max_bytes"]
    if not _is_int(max_bytes) or max_bytes < MIN_STRIDE:
        raise IndirectError(
            f"max_bytes must be an integer >= {MIN_STRIDE}, got {max_bytes!r}"
        )

    # Validate every buffer before building anything: all-or-nothing, and no
    # buffer is ever shared between batches.
    buffers = list(buffers)
    for buffer in buffers:
        if isinstance(buffer, (str, bytes)) or not isinstance(buffer, Sequence):
            raise IndirectError("each buffer must be a sequence of commands")
        for cmd in buffer:
            _checked_command(cmd)
        size = len(buffer) * stride
        if buffer and size > max_bytes + 1:
            raise IndirectError(
                f"buffer of {len(buffer)} commands needs {size} bytes, "
                f"exceeding the {max_bytes}-byte budget"
            )

    batches = [batch_writes(buffer, stride) for buffer in buffers]
    return {
        "batches": batches,
        "data": b"".join(batches),
        "count": sum(len(buffer) for buffer in buffers),
        "stride": stride,
    }
