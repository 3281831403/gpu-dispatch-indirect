"""Initial red test suite for the GPU indirect-draw exercise.

Each test calls at least one public stub in ``indirect.args`` and expresses
one user-visible contract. The suite must collect cleanly while all fifteen
tests fail with ``NotImplementedError``.
"""

import copy
import struct

import pytest

from indirect.args import (
    IndirectError,
    align_stride,
    apply_base_offset,
    batch_writes,
    dispatch_batch,
    pack_draw,
    split_by_budget,
    unpack_draw,
    validate_args,
)


def command_record(count, first, base_vertex=0, instance=1):
    """Build a command using exactly the documented four fields."""
    return {
        "count": count,
        "instance": instance,
        "first": first,
        "base_vertex": base_vertex,
    }


# ---------------------------------------------------------------------------
# One regression test per documented defect.
# ---------------------------------------------------------------------------


def test_pack_draw_uses_little_endian():
    # Defect 1: big-endian words make the GPU read every integer incorrectly.
    command = command_record(2, 4, base_vertex=5, instance=3)
    data = pack_draw(command)
    assert len(data) == 20
    assert struct.unpack("<5I", data) == (2, 3, 4, 5, 0)


def test_align_stride_never_truncates_the_twenty_byte_command():
    # Defect 2: truncating 20 bytes to a 16-byte boundary overlaps commands.
    assert align_stride(20, 16) == 32
    assert align_stride(20, 4) == 20


def test_batch_writes_rejects_stride_smaller_than_command():
    # Defect 3: a 16-byte stride would let the next command overwrite this one.
    with pytest.raises(IndirectError):
        batch_writes([command_record(1, 0)], 16)


def test_base_offset_updates_first_and_base_vertex_not_count():
    # Defect 4: adding the offset to count changes the draw size, not its range.
    command = command_record(6, 10, base_vertex=4, instance=2)
    original = copy.deepcopy(command)
    adjusted = apply_base_offset([command], 7)
    assert adjusted == [
        {"count": 6, "instance": 2, "first": 17, "base_vertex": 11}
    ]
    assert command == original


def test_split_by_budget_limits_each_chunk_in_bytes():
    # Defect 5: a count-only split puts three 20-byte commands in a 45-byte block.
    commands = [command_record(index + 1, index * 4) for index in range(5)]
    chunks = split_by_budget(commands, 45)
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [command["count"] for chunk in chunks for command in chunk] == [
        1,
        2,
        3,
        4,
        5,
    ]
    for chunk in chunks:
        assert len(batch_writes(chunk, 20)) <= 45


def test_zero_count_is_reported_as_an_issue():
    # Defect 6: count=0 is an empty draw and must not be silently accepted.
    issues = validate_args([command_record(0, 0)])
    assert issues
    assert any("count" in issue.lower() for issue in issues)


def test_negative_offsets_are_reported_and_rejected():
    # Defect 7: a negative index would wrap around instead of being rejected.
    bad = command_record(1, -1, base_vertex=-2)
    issues = validate_args([bad])
    assert issues
    assert any(
        "offset" in issue.lower() or "negative" in issue.lower()
        for issue in issues
    )
    with pytest.raises(IndirectError):
        apply_base_offset([bad], 4)


def test_unpack_draw_rejects_wrong_length():
    # Defect 8: parsing past a short command reads the next command's bytes.
    for size in (0, 19, 21):
        with pytest.raises(IndirectError):
            unpack_draw(b"\x00" * size)


def test_dispatch_batch_uses_independent_buffers_per_batch():
    # Defect 9: one shared parameter buffer lets later batches overwrite data.
    buffers = [
        [command_record(1, 0), command_record(2, 20)],
        [command_record(3, 40)],
    ]
    result = dispatch_batch(buffers, {"stride": 20, "max_bytes": 40})
    first, second = result["batches"]
    assert first != second
    assert unpack_draw(first[:20])["count"] == 1
    assert unpack_draw(first[20:40])["count"] == 2
    assert unpack_draw(second[:20])["count"] == 3
    assert result["data"] == first + second


def test_align_stride_rejects_zero_alignment():
    # Defect 10: alignment=0 must raise instead of dividing by zero.
    for alignment in (0, 3, -4):
        with pytest.raises(IndirectError):
            align_stride(20, alignment)


# ---------------------------------------------------------------------------
# Five additional contract tests.
# ---------------------------------------------------------------------------


def test_pack_draw_is_exactly_twenty_bytes():
    command = command_record(7, 8, base_vertex=9, instance=10)
    data = pack_draw(command)
    assert len(data) == 20
    assert unpack_draw(data) == command


def test_two_commands_are_contiguous_with_stride_twenty():
    first = command_record(1, 0)
    second = command_record(2, 3)
    data = batch_writes([first, second], 20)
    assert len(data) == 40
    assert unpack_draw(data[:20]) == first
    assert unpack_draw(data[20:]) == second


def test_stride_nineteen_raises_indirect_error():
    with pytest.raises(IndirectError):
        batch_writes([command_record(1, 0)], 19)


def test_empty_commands_return_empty_bytes():
    assert batch_writes([], 20) == b""
    assert split_by_budget([], 20) == []


def test_dispatch_batch_preserves_order_when_all_buffers_succeed():
    buffers = [
        [command_record(1, 0), command_record(2, 20)],
        [command_record(3, 40)],
    ]
    result = dispatch_batch(buffers, {"stride": 20, "max_bytes": 60})
    assert result["count"] == 3
    assert result["stride"] == 20
    assert len(result["data"]) == 60
    counts = [
        unpack_draw(result["data"][offset : offset + 20])["count"]
        for offset in (0, 20, 40)
    ]
    assert counts == [1, 2, 3]
