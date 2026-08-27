from simulator.memory import KVMemory


def make_memory(token_bytes=16, capacity_tokens=2):
    return KVMemory(kv_budget_bytes=token_bytes * capacity_tokens, kv_bytes_per_token=token_bytes)


def test_kv_stays_on_chip_while_complete_tokens_fit():
    memory = make_memory()
    first = memory.place_prompt_token()
    second = memory.place_decode_token(0)
    assert first.location == "on_chip"
    assert second.location == "on_chip"
    assert memory.on_chip_bytes == 32
    assert memory.off_chip_bytes == 0


def test_new_kv_spills_when_no_complete_token_space_remains():
    memory = make_memory()
    memory.place_prompt_token()
    memory.place_prompt_token()
    placement = memory.place_decode_token(0)
    assert placement.location == "off_chip"
    assert placement.bytes_count == 16
    assert memory.off_chip_write_bytes == 16
    assert memory.on_chip_bytes <= memory.kv_budget_bytes


def test_prompt_initialization_does_not_count_as_timed_write_traffic():
    memory = make_memory(capacity_tokens=0)
    memory.place_prompt_token()
    assert memory.off_chip_bytes == 16
    assert memory.off_chip_write_bytes == 0


def test_spilled_kv_read_grows_with_context():
    memory = make_memory()
    memory.place_prompt_token()
    memory.place_prompt_token()
    third = memory.place_decode_token(0)
    assert memory.off_chip_attention_bytes() == third.bytes_count
    fourth = memory.place_decode_token(1)
    assert memory.off_chip_attention_bytes() == third.bytes_count + fourth.bytes_count


def test_release_zeros_current_usage_but_retains_peaks():
    memory = make_memory(capacity_tokens=1)
    memory.place_prompt_token()
    memory.place_decode_token(0)
    memory.release()
    assert memory.on_chip_bytes == 0
    assert memory.off_chip_bytes == 0
    assert memory.on_chip_peak_bytes == 16
    assert memory.off_chip_peak_bytes == 16
