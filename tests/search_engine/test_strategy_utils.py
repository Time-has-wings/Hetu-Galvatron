import pytest
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Since the code lives at galvatron.utils.strategy_utils, we try to import
# from there first.  If the package isn't installed in the test environment
# we fall back to a local copy so the tests are still runnable standalone.
# ---------------------------------------------------------------------------
try:
    from galvatron.utils.strategy_utils import (
        ColorSet,
        DPType,
        StrategyBase,
        EmbeddingLMHeadStrategy,
        AttentionStrategy,
        FFNStrategy,
        LayerStrategy,
        MoEFFNStrategy,
        byte_to_MB,
        model_states_to_param_size_ratio,
        is_power_of_two,
        old_version_strategy_to_new_version_strategy,
        new_version_strategy_to_old_version_strategy,
        print_strategy_list,
        strategy_list2config,
    )
except ImportError:
    pytest.skip(
        "galvatron.utils.strategy_utils not importable – skipping module",
        allow_module_level=True,
    )


# ========================================================================= #
#                            DPType Tests                                    #
# ========================================================================= #
class TestDPType:
    def test_enum_values(self):
        assert DPType.DDP.value == "ddp"
        assert DPType.ZERO2.value == "zero2"
        assert DPType.ZERO3.value == "zero3"

    def test_values_returns_all_members(self):
        vals = DPType.values()
        assert set(vals) == {DPType.DDP, DPType.ZERO2, DPType.ZERO3}

    def test_contains_true(self):
        for dp in DPType:
            assert DPType.contains(dp) is True

    def test_contains_false(self):
        assert DPType.contains("not_a_dp_type") is False

    def test_lt_ordering(self):
        # string ordering: 'ddp' < 'zero2' < 'zero3'
        assert DPType.DDP < DPType.ZERO2
        assert DPType.ZERO2 < DPType.ZERO3
        assert not (DPType.ZERO3 < DPType.DDP)

    def test_lt_type_error(self):
        with pytest.raises(TypeError):
            _ = DPType.DDP < "ddp"


# ========================================================================= #
#                          ColorSet Tests                                    #
# ========================================================================= #
class TestColorSet:
    def test_ansi_codes_exist(self):
        assert ColorSet.YELLOW == "\033[33m"
        assert ColorSet.RED == "\033[31m"
        assert ColorSet.GREEN == "\033[32m"
        assert ColorSet.BLUE == "\033[34m"
        assert ColorSet.RESET == "\033[0m"


# ========================================================================= #
#                    EmbeddingLMHeadStrategy Tests                           #
# ========================================================================= #
class TestEmbeddingLMHeadStrategy:
    def test_default_values(self):
        s = EmbeddingLMHeadStrategy()
        assert s.pp_size == 1
        assert s.tp_size == 1
        assert s.sp_size == 1
        assert s.cp_size == 1
        assert s.dp_size == 1
        # dp_size==1 triggers auto-reset to DDP
        assert s.dp_type == DPType.DDP

    def test_auto_reset_dp_type_when_sdp_is_1(self):
        """When sdp_size == 1 and dp_type != DDP, it should be auto-corrected to DDP."""
        s = EmbeddingLMHeadStrategy(dp_size=1, dp_type=DPType.ZERO3)
        assert s.dp_type == DPType.DDP

    def test_dp_type_preserved_when_sdp_gt_1(self):
        s = EmbeddingLMHeadStrategy(dp_size=4, dp_type=DPType.ZERO2)
        assert s.dp_type == DPType.ZERO2

    def test_tp_and_sp_mutual_exclusion(self):
        with pytest.raises(AssertionError):
            EmbeddingLMHeadStrategy(tp_size=2, sp_size=2)

    def test_world_size(self):
        s = EmbeddingLMHeadStrategy(pp_size=2, tp_size=4, sp_size=1, cp_size=1, dp_size=8)
        assert s.world_size == 2 * 4 * 1 * 1 * 8

    def test_sdp_size(self):
        s = EmbeddingLMHeadStrategy(dp_size=4, sp_size=1, cp_size=2, dp_type=DPType.ZERO2)
        assert s.sdp_size == 4 * 1 * 2

    def test_tp_sp_size_with_tp(self):
        s = EmbeddingLMHeadStrategy(tp_size=4, sp_size=1)
        assert s.tp_sp_size == 4

    def test_tp_sp_size_with_sp(self):
        s = EmbeddingLMHeadStrategy(tp_size=1, sp_size=4, dp_size=4, dp_type=DPType.ZERO2)
        assert s.tp_sp_size == 4

    def test_equality_same(self):
        a = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        b = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        assert a == b

    def test_equality_different(self):
        a = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        b = EmbeddingLMHeadStrategy(pp_size=4, dp_size=4, dp_type=DPType.ZERO2)
        assert a != b

    def test_equality_different_type(self):
        a = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        assert a != "not_a_strategy"

    def test_hash_consistency(self):
        a = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        b = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        assert hash(a) == hash(b)

    def test_hash_usable_in_set(self):
        a = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        b = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        assert len({a, b}) == 1

    def test_lt(self):
        a = EmbeddingLMHeadStrategy(pp_size=1, dp_size=4, dp_type=DPType.ZERO2)
        b = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        assert a < b
        assert not (b < a)

    def test_lt_not_implemented_for_different_types(self):
        a = EmbeddingLMHeadStrategy()
        assert a.__lt__("string") is NotImplemented

    def test_to_string(self):
        s = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        result = s.to_string()
        assert "EmbeddingLMHeadStrategy" in result
        assert "pp_size=2" in result

    def test_str(self):
        s = EmbeddingLMHeadStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        result = str(s)
        assert "EmbeddingLMHeadStrategy" in result

    def test_to_simple_string_basic(self):
        s = EmbeddingLMHeadStrategy(pp_size=2, tp_size=1, sp_size=1, dp_size=4, dp_type=DPType.ZERO2)
        result = s.to_simple_string()
        assert result == "2-1-4"

    def test_to_simple_string_with_tp(self):
        s = EmbeddingLMHeadStrategy(pp_size=2, tp_size=4, sp_size=1, dp_size=2, dp_type=DPType.ZERO2)
        result = s.to_simple_string()
        assert result == "2-4*-2"

    def test_to_simple_string_zero3(self):
        s = EmbeddingLMHeadStrategy(pp_size=1, tp_size=1, sp_size=1, dp_size=8, dp_type=DPType.ZERO3)
        result = s.to_simple_string()
        assert result == "1-1-8f"

    def test_to_simple_string_with_sp(self):
        s = EmbeddingLMHeadStrategy(pp_size=1, tp_size=1, sp_size=4, dp_size=4, dp_type=DPType.ZERO2)
        result = s.to_simple_string()
        # sp_size > 1 → tp_sp_size=4 → '*', and suffix '-sp'
        assert result == "1-4*-4-sp"


# ========================================================================= #
#                       AttentionStrategy Tests                              #
# ========================================================================= #
class TestAttentionStrategy:
    def test_default_checkpoint_false(self):
        s = AttentionStrategy()
        assert s.checkpoint is False

    def test_inherits_embedding_fields(self):
        s = AttentionStrategy(pp_size=2, tp_size=4, dp_size=2, dp_type=DPType.ZERO2)
        assert s.pp_size == 2
        assert s.world_size == 2 * 4 * 1 * 1 * 2

    def test_to_embedding_lmhead_strategy(self):
        s = AttentionStrategy(pp_size=2, tp_size=4, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        emb = s.to_embedding_lmhead_strategy()
        assert isinstance(emb, EmbeddingLMHeadStrategy)
        assert not isinstance(emb, AttentionStrategy)
        assert emb.pp_size == 2
        assert emb.tp_size == 4

    def test_to_ffn_strategy(self):
        s = AttentionStrategy(pp_size=2, tp_size=4, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        ffn = s.to_ffn_strategy()
        assert isinstance(ffn, FFNStrategy)
        assert ffn.checkpoint is True
        assert ffn.pp_size == 2

    def test_to_layer_strategy(self):
        s = AttentionStrategy(pp_size=2, tp_size=4, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        layer = s.to_layer_strategy()
        assert isinstance(layer, LayerStrategy)
        assert layer.checkpoint is True

    def test_hash(self):
        a = AttentionStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True)
        b = AttentionStrategy(pp_size=2, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True)
        assert hash(a) == hash(b)

    def test_to_simple_string_with_checkpoint(self):
        s = AttentionStrategy(pp_size=1, tp_size=1, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True)
        result = s.to_simple_string()
        assert "-c" in result


# ========================================================================= #
#                          FFNStrategy Tests                                 #
# ========================================================================= #
class TestFFNStrategy:
    def test_default_checkpoint(self):
        s = FFNStrategy()
        assert s.checkpoint is False

    def test_to_embedding_lmhead_strategy(self):
        s = FFNStrategy(pp_size=2, tp_size=2, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True)
        emb = s.to_embedding_lmhead_strategy()
        assert isinstance(emb, EmbeddingLMHeadStrategy)
        assert not isinstance(emb, FFNStrategy)

    def test_hash(self):
        a = FFNStrategy(pp_size=1, dp_size=2, dp_type=DPType.ZERO2)
        b = FFNStrategy(pp_size=1, dp_size=2, dp_type=DPType.ZERO2)
        assert hash(a) == hash(b)


# ========================================================================= #
#                         LayerStrategy Tests                                #
# ========================================================================= #
class TestLayerStrategy:
    def test_default_checkpoint(self):
        s = LayerStrategy()
        assert s.checkpoint is False

    def test_to_embedding_lmhead_strategy(self):
        s = LayerStrategy(pp_size=4, tp_size=2, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        emb = s.to_embedding_lmhead_strategy()
        assert isinstance(emb, EmbeddingLMHeadStrategy)
        assert emb.pp_size == 4

    def test_hash(self):
        a = LayerStrategy(pp_size=1, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        b = LayerStrategy(pp_size=1, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


# ========================================================================= #
#                        MoEFFNStrategy Tests                                #
# ========================================================================= #
class TestMoEFFNStrategy:
    def test_default_values(self):
        s = MoEFFNStrategy()
        assert s.pp_size == 1
        assert s.ep_size == 1
        assert s.tp_size == 1
        assert s.dp_size == 1
        # dp_size==1 → auto-corrected to DDP
        assert s.dp_type == DPType.DDP
        assert s.checkpoint is False

    def test_auto_reset_dp_type_when_dp_is_1(self):
        s = MoEFFNStrategy(dp_size=1, dp_type=DPType.ZERO3)
        assert s.dp_type == DPType.DDP

    def test_dp_type_preserved_when_dp_gt_1(self):
        s = MoEFFNStrategy(dp_size=4, dp_type=DPType.ZERO2)
        assert s.dp_type == DPType.ZERO2

    def test_world_size(self):
        s = MoEFFNStrategy(pp_size=2, ep_size=4, tp_size=2, dp_size=2, dp_type=DPType.ZERO2)
        assert s.world_size == 2 * 2 * 2 * 4

    def test_sdp_size(self):
        s = MoEFFNStrategy(dp_size=8, dp_type=DPType.ZERO2)
        assert s.sdp_size == 8

    def test_equality(self):
        a = MoEFFNStrategy(ep_size=4, dp_size=2, dp_type=DPType.ZERO2)
        b = MoEFFNStrategy(ep_size=4, dp_size=2, dp_type=DPType.ZERO2)
        assert a == b

    def test_inequality(self):
        a = MoEFFNStrategy(ep_size=4, dp_size=2, dp_type=DPType.ZERO2)
        b = MoEFFNStrategy(ep_size=8, dp_size=2, dp_type=DPType.ZERO2)
        assert a != b

    def test_equality_different_type(self):
        a = MoEFFNStrategy()
        assert a != "not_a_strategy"

    def test_lt(self):
        a = MoEFFNStrategy(pp_size=1, ep_size=1, dp_size=2, dp_type=DPType.ZERO2)
        b = MoEFFNStrategy(pp_size=2, ep_size=1, dp_size=2, dp_type=DPType.ZERO2)
        assert a < b

    def test_lt_not_implemented(self):
        a = MoEFFNStrategy()
        assert a.__lt__(42) is NotImplemented

    def test_hash(self):
        a = MoEFFNStrategy(ep_size=4, dp_size=2, dp_type=DPType.ZERO2)
        b = MoEFFNStrategy(ep_size=4, dp_size=2, dp_type=DPType.ZERO2)
        assert hash(a) == hash(b)

    def test_str(self):
        s = MoEFFNStrategy(ep_size=4)
        result = str(s)
        assert "MoEFFNStrategy" in result


# ========================================================================= #
#                       Utility Function Tests                               #
# ========================================================================= #
class TestIsPowerOfTwo:
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 1024])
    def test_powers_of_two(self, n):
        assert is_power_of_two(n) is True

    @pytest.mark.parametrize("n", [0, -1, 3, 5, 6, 7, 9, 15, 100])
    def test_not_powers_of_two(self, n):
        assert is_power_of_two(n) is False


class TestConstants:
    def test_byte_to_MB(self):
        assert byte_to_MB == 1024 * 1024

    def test_model_states_ratio(self):
        assert model_states_to_param_size_ratio == 4


# ========================================================================= #
#                  Version Conversion Tests                                  #
# ========================================================================= #
class TestOldToNewVersionStrategy:
    def test_basic_ddp(self):
        # [pp, tp, dp, info]
        old = [2, 1, 4, {}]
        s = old_version_strategy_to_new_version_strategy(old, "ddp")
        assert isinstance(s, LayerStrategy)
        assert s.pp_size == 2
        assert s.tp_size == 1
        assert s.sp_size == 1
        assert s.cp_size == 1
        assert s.dp_size == 4
        assert s.dp_type == DPType.DDP
        assert s.checkpoint is False

    def test_with_fsdp(self):
        old = [1, 1, 8, {"fsdp": 1}]
        s = old_version_strategy_to_new_version_strategy(old, "ddp")
        assert s.dp_type == DPType.ZERO3
        assert s.dp_size == 8

    def test_with_checkpoint(self):
        old = [1, 1, 4, {"cpt": 1}]
        s = old_version_strategy_to_new_version_strategy(old, "ddp")
        assert s.checkpoint is True

    def test_with_sp(self):
        old = [1, 4, 2, {"sp": 1}]
        s = old_version_strategy_to_new_version_strategy(old, "zero2")
        assert s.tp_size == 1
        assert s.sp_size == 4

    def test_default_zero2(self):
        old = [1, 1, 4, {}]
        s = old_version_strategy_to_new_version_strategy(old, "zero2")
        assert s.dp_type == DPType.ZERO2

    def test_dp_size_1_forces_ddp(self):
        old = [2, 4, 1, {}]
        s = old_version_strategy_to_new_version_strategy(old, "zero2")
        assert s.dp_type == DPType.DDP


class TestNewToOldVersionStrategy:
    def test_basic_roundtrip_ddp(self):
        s = LayerStrategy(pp_size=2, tp_size=1, sp_size=1, cp_size=1, dp_size=4, dp_type=DPType.DDP, checkpoint=False)
        old = new_version_strategy_to_old_version_strategy(s)
        assert old[0] == 2  # pp
        assert old[1] == 1  # tp
        assert old[2] == 4  # dp
        assert "fsdp" not in old[3] or old[3].get("fsdp") == 0

    def test_fsdp_flag(self):
        s = LayerStrategy(pp_size=1, tp_size=1, sp_size=1, cp_size=1, dp_size=8, dp_type=DPType.ZERO3, checkpoint=False)
        old = new_version_strategy_to_old_version_strategy(s)
        assert old[3]["fsdp"] == 1

    def test_tp_flag(self):
        s = LayerStrategy(pp_size=1, tp_size=4, sp_size=1, cp_size=1, dp_size=2, dp_type=DPType.ZERO2, checkpoint=False)
        old = new_version_strategy_to_old_version_strategy(s)
        assert old[1] == 4
        assert old[3]["tp"] == 1
        assert old[3]["sp"] == 0

    def test_sp_flag(self):
        s = LayerStrategy(pp_size=1, tp_size=1, sp_size=4, cp_size=1, dp_size=4, dp_type=DPType.ZERO2, checkpoint=False)
        old = new_version_strategy_to_old_version_strategy(s)
        assert old[1] == 4
        assert old[3]["sp"] == 1

    def test_checkpoint_flag(self):
        s = LayerStrategy(pp_size=1, tp_size=1, sp_size=1, cp_size=1, dp_size=4, dp_type=DPType.DDP, checkpoint=True)
        old = new_version_strategy_to_old_version_strategy(s)
        assert old[3]["cpt"] == 1


# ========================================================================= #
#                     print_strategy_list Tests                              #
# ========================================================================= #
class TestPrintStrategyList:
    def test_none_input(self, capsys):
        # Should not raise
        print_strategy_list(None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_strategies(self, capsys):
        strategies = [
            LayerStrategy(pp_size=1, tp_size=1, dp_size=4, dp_type=DPType.ZERO2, checkpoint=False),
            LayerStrategy(pp_size=1, tp_size=1, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True),
        ]
        print_strategy_list(strategies)
        captured = capsys.readouterr()
        assert "1-1-4" in captured.out
        assert "-c" in captured.out

    def test_with_logger(self):
        class FakeLogger:
            def __init__(self):
                self.messages = []
            def info(self, msg):
                self.messages.append(msg)

        logger = FakeLogger()
        strategies = [
            LayerStrategy(pp_size=2, tp_size=1, dp_size=4, dp_type=DPType.DDP),
        ]
        print_strategy_list(strategies, logger=logger)
        assert len(logger.messages) == 1
        assert "2-1-4" in logger.messages[0]


# ========================================================================= #
#                     strategy_list2config Tests                             #
# ========================================================================= #
class TestStrategyList2Config:
    def test_empty_list(self):
        assert strategy_list2config([]) == {}

    def test_single_layer(self):
        strategies = [
            LayerStrategy(pp_size=2, tp_size=4, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True),
        ]
        config = strategy_list2config(strategies)
        assert config["pp_deg"] == 2
        assert config["tp_sizes_enc"] == "4"
        assert config["tp_consecutive_flags"] == "1"
        assert config["dp_types_enc"] == "0"  # ZERO2 → 0
        assert config["use_sp"] == "0"
        assert config["checkpoint"] == "1"

    def test_multiple_layers(self):
        strategies = [
            LayerStrategy(pp_size=2, tp_size=4, dp_size=2, dp_type=DPType.ZERO3, checkpoint=False),
            LayerStrategy(pp_size=2, tp_size=2, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True),
            LayerStrategy(pp_size=2, tp_size=1, sp_size=4, dp_size=4, dp_type=DPType.DDP, checkpoint=False),
        ]
        config = strategy_list2config(strategies)
        assert config["pp_deg"] == 2
        assert config["tp_sizes_enc"] == "4,2,4"
        assert config["tp_consecutive_flags"] == "1,1,1"
        assert config["dp_types_enc"] == "1,0,0"  # ZERO3, ZERO2, DDP
        assert config["use_sp"] == "0,0,1"
        assert config["checkpoint"] == "0,1,0"

    def test_all_zero3(self):
        strategies = [
            LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.ZERO3, checkpoint=True),
            LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.ZERO3, checkpoint=True),
        ]
        config = strategy_list2config(strategies)
        assert config["dp_types_enc"] == "1,1"
        assert config["checkpoint"] == "1,1"