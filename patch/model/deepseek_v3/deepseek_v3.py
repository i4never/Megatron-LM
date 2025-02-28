from typing import Literal, Optional

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from patch.model.deepseek_v3.configuration_deepseek import DeepseekV3Config


class DeepSeekV3Model(GPTModel):
    def __init__(
            self,
            config: DeepseekV3Config,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            rotary_percent = 1,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            rope_scaling_factor: float = 8.,
            scatter_embedding_sequence_parallel: bool = True,
            seq_len_interpolation_factor: Optional[float] = None,
            use_multi_token_prediction: bool = False,
        ):
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )

        self.use_multi_token_prediction = use_multi_token_prediction

        assert self.use_multi_token_prediction is False, "Multi-token prediction is not supported yet"
