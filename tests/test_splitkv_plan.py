"""Hand-checkable capacity/shape fixtures for the pinned static partition plan."""
import pytest
from paged_decode.spec import DecodeSpec
from paged_decode.splitkv_plan import make_splitkv_plan


@pytest.mark.parametrize("batch,heads,kv,capacity,splits,tokens", [
    (8,8,2,1024,8,128), (8,16,4,1024,6,192),
    (1,16,4,2176,36,64), (1,8,2,2176,36,64), (32,32,8,8192,1,8192)])
def test_static_shape_examples(batch, heads, kv, capacity, splits, tokens):
    plan = make_splitkv_plan(DecodeSpec(batch,heads,kv,256,capacity),108)
    assert (plan.splits,plan.tokens_per_split)==(splits,tokens)
    assert plan.table_capacity_tokens == (capacity+255)//256*256


def test_explicit_count_partitions_capacity_in_tiles():
    plan = make_splitkv_plan(DecodeSpec(1,16,4,256,2176),108,num_splits=3)
    assert plan.splits==3 and plan.tokens_per_split==768
    assert plan.policy=="explicit"


@pytest.mark.parametrize("count", [-1,129,1.5])
def test_unsupported_explicit_counts(count):
    with pytest.raises(ValueError,match="num_splits"):
        make_splitkv_plan(DecodeSpec(1,8,2,256,1024),108,num_splits=count)


def test_unsupported_group():
    with pytest.raises(ValueError,match="16"):
        make_splitkv_plan(DecodeSpec(1,32,1,256,1024),108)
