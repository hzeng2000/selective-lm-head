from selective_lm_head.grammar.bitmask import bitmask_to_allowed_ids, bucket_k


def test_bool_mask_to_allowed_ids():
    assert bitmask_to_allowed_ids([False, True, False, True], vocab_size=4) == [1, 3]


def test_packed_int_mask_to_allowed_ids():
    # bits 0, 2, and 33 are set.
    assert bitmask_to_allowed_ids([0b101, 0b10], vocab_size=40) == [0, 2, 33]


def test_torch_packed_int_mask_to_allowed_ids_tensor_if_available():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from selective_lm_head.grammar.bitmask import bitmask_to_allowed_ids_tensor

    mask = torch.tensor([0b101, 0b10], dtype=torch.int32)
    ids = bitmask_to_allowed_ids_tensor(mask, vocab_size=40)
    assert ids.dtype == torch.long
    assert ids.tolist() == [0, 2, 33]


def test_torch_bool_mask_to_allowed_ids_tensor_if_available():
    try:
        import torch
    except ModuleNotFoundError:
        return

    from selective_lm_head.grammar.bitmask import bitmask_to_allowed_ids_tensor

    mask = torch.tensor([False, True, False, True])
    assert bitmask_to_allowed_ids_tensor(mask, vocab_size=4).tolist() == [1, 3]


def test_bucket_k():
    assert bucket_k(1) == "K=1"
    assert bucket_k(8) == "K<=8"
    assert bucket_k(2048) == "K<=2048"
    assert bucket_k(8193) == "K>8192"
