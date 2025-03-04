TransformerBlockSubmodules(
    layer_specs=[
        ModuleSpec(
            module=<class 'megatron.core.transformer.transformer_layer.TransformerLayer'>,
            params={},
            submodules=TransformerLayerSubmodules(
                input_layernorm=<class 'megatron.core.extensions.transformer_engine.TENorm'>,
                self_attention=ModuleSpec(
                    module=<class 'megatron.core.transformer.multi_latent_attention.MLASelfAttention'>,
                    params={'attn_mask_type': <AttnMaskType.causal: 2>},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                        linear_q_down_proj=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                        linear_q_up_proj=<class 'megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear'>,
                        linear_kv_down_proj=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                        linear_kv_up_proj=<class 'megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear'>,
                        core_attention=<class 'megatron.core.extensions.transformer_engine.TEDotProductAttention'>,
                        linear_proj=<class 'megatron.core.extensions.transformer_engine.TERowParallelLinear'>,
                        q_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                        kv_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>
                    )
                ),
                self_attn_bda=<function get_bias_dropout_add at 0x7f37da80bd00>,
                pre_cross_attn_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                cross_attention=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                cross_attn_bda=<class 'megatron.core.transformer.identity_op.IdentityFuncOp'>,
                pre_mlp_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                mlp=ModuleSpec(
                    module=<class 'megatron.core.transformer.mlp.MLP'>,
                    params={},
                    submodules=MLPSubmodules(
                        linear_fc1=<class 'megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear'>,
                        linear_fc2=<class 'megatron.core.extensions.transformer_engine.TERowParallelLinear'>
                    )
                ),
                mlp_bda=<function get_bias_dropout_add at 0x7f37da80bd00>,
                sharded_state_dict_keys_map={}
            )
        ),
        ModuleSpec(
            module=<class 'megatron.core.transformer.transformer_layer.TransformerLayer'>,
            params={},
            submodules=TransformerLayerSubmodules(
                input_layernorm=<class 'megatron.core.extensions.transformer_engine.TENorm'>,
                self_attention=ModuleSpec(
                    module=<class 'megatron.core.transformer.multi_latent_attention.MLASelfAttention'>,
                    params={'attn_mask_type': <AttnMaskType.causal: 2>},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                        linear_q_down_proj=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                        linear_q_up_proj=<class 'megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear'>,
                        linear_kv_down_proj=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                        linear_kv_up_proj=<class 'megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear'>,
                        core_attention=<class 'megatron.core.extensions.transformer_engine.TEDotProductAttention'>,
                        linear_proj=<class 'megatron.core.extensions.transformer_engine.TERowParallelLinear'>,
                        q_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                        kv_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>
                    )
                ),
                self_attn_bda=<function get_bias_dropout_add at 0x7f37da80bd00>,
                pre_cross_attn_layernorm=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                cross_attention=<class 'megatron.core.transformer.identity_op.IdentityOp'>,
                cross_attn_bda=<class 'megatron.core.transformer.identity_op.IdentityFuncOp'>,
                pre_mlp_layernorm=<class 'megatron.core.extensions.transformer_engine.TENorm'>,
                mlp=ModuleSpec(
                    module=<class 'megatron.core.transformer.moe.moe_layer.MoELayer'>,
                    params={},
                    submodules=MoESubmodules(
                        experts=ModuleSpec(
                            module=<class 'megatron.core.transformer.moe.experts.TEGroupedMLP'>,
                            params={},
                            submodules=MLPSubmodules(
                                linear_fc1=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelGroupedLinear'>,
                                linear_fc2=<class 'megatron.core.extensions.transformer_engine.TERowParallelGroupedLinear'>
                            )
                        ),
                        shared_experts=ModuleSpec(
                            module=<class 'megatron.core.transformer.moe.shared_experts.SharedExpertMLP'>,
                            params={'gate': False},
                            submodules=MLPSubmodules(
                                linear_fc1=<class 'megatron.core.extensions.transformer_engine.TEColumnParallelLinear'>,
                                linear_fc2=<class 'megatron.core.extensions.transformer_engine.TERowParallelLinear'>
                            )
                        )
                    )
                ),
                mlp_bda=<function get_bias_dropout_add at 0x7f37da80bd00>,
                sharded_state_dict_keys_map={}
            )
        )
    ],
    layer_norm=<class 'megatron.core.extensions.transformer_engine.TENorm'>
)