from functools import reduce


class LayerData:
    def __init__(self, shape: tuple, dtype: str):
        assert len(shape) == 4 # (batch, head, M, K)
        self.shape = shape
        self.dtype = dtype
        self.data_num = reduce(lambda x, y: x * y, shape)
        self.bytes = self.data_num * self.get_bitwidth_from_dtype(dtype) / 8
        
    @staticmethod
    def get_bitwidth_from_dtype(data_format: str):
        # 根据数据格式字符串返回比特数，例如 "FP16" 或 "INT9"
        if data_format.startswith("INT"):
            return float(data_format.lstrip("INT"))
        elif data_format.startswith("FP"):
            return float(data_format.lstrip("FP"))
        else:
            raise ValueError
        
    @staticmethod
    def squeeze_shape(shape):
        # 移除形状中为 1 的维度，用于判断操作是否改变数据排布
        return tuple(d for d in shape if d != 1)


class ModelConfig:
    def __init__(self, model_name, W_dtype, A_dtype, KV_dtype, MISC_dtype="FP16"): # misc use FP16 by default
        self.model_name = model_name
        
        if self.model_name == "Llama-2-7B":
            self.BIAS_FLAG = False
            self.HIDDEN_SIZE = 4096
            self.INTERMEDIATE_SIZE = 11008
            self.NUM_ATTENTION_HEADS = 32
            self.NUM_HIDDEN_LAYERS = 32
            self.NUM_KEY_VALUE_HEADS = 32
            self.VOCAB_SIZE = 32000

        elif self.model_name in ("Llama-3-8B", "Llama-3.1-8B"):
            self.BIAS_FLAG = False
            self.HIDDEN_SIZE = 4096
            self.INTERMEDIATE_SIZE = 14336
            self.NUM_ATTENTION_HEADS = 32
            self.NUM_HIDDEN_LAYERS = 32
            self.NUM_KEY_VALUE_HEADS = 8
            self.VOCAB_SIZE = 128256

        elif self.model_name in ("Llama-3-70B", "Llama-3.1-70B"):
            self.BIAS_FLAG = False
            self.HIDDEN_SIZE = 8192
            self.INTERMEDIATE_SIZE = 28672
            self.NUM_ATTENTION_HEADS = 64
            self.NUM_HIDDEN_LAYERS = 80
            self.NUM_KEY_VALUE_HEADS = 8
            self.VOCAB_SIZE = 128256

        elif self.model_name in ("Llama-3-405B"):
            self.BIAS_FLAG = False
            self.HIDDEN_SIZE = 16384
            self.INTERMEDIATE_SIZE = 53248
            self.NUM_ATTENTION_HEADS = 128
            self.NUM_HIDDEN_LAYERS = 126
            self.NUM_KEY_VALUE_HEADS = 16
            self.VOCAB_SIZE = 128256

        else:
            raise ValueError("Model_name not found!")
        
        self.HEAD_DIM = self.HIDDEN_SIZE // self.NUM_ATTENTION_HEADS
        assert self.HIDDEN_SIZE % self.NUM_ATTENTION_HEADS == 0
        self.num_key_value_groups = self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS
        assert self.NUM_ATTENTION_HEADS % self.NUM_KEY_VALUE_HEADS == 0

        self.W_dtype = W_dtype
        self.A_dtype = A_dtype
        self.KV_dtype = KV_dtype
        self.MISC_dtype = MISC_dtype
        if self.BIAS_FLAG:
            self.bias_dtype = "FP16" # 默认FP16

        W_bits = LayerData.get_bitwidth_from_dtype(W_dtype)
        A_bits = LayerData.get_bitwidth_from_dtype(A_dtype)
        KV_bits = LayerData.get_bitwidth_from_dtype(KV_dtype)
        self.quant_name = f"[W: {self.W_dtype}, A: {self.A_dtype}, KV: {self.KV_dtype}]"
    
        self._profile_state = { # 记录所有profile结果
            "layer_name": list(),
            "layer_type": list(),
            "compute_amount": list(),
            "read_weight_bytes": list(),
            "read_activation_bytes": list(),
            "write_activation_bytes": list(),
        }

    def init_profile_state(self):
        for key in self._profile_state.keys():
            self._profile_state[key].clear()


class BaseLayer():
    def __init__(self, config: ModelConfig, layer_type, layer_name):
        self.config = config
        self.layer_type = layer_type
        self.layer_name = layer_name

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        raise NotImplementedError("Profile_with_input must be overload!")

    def update_profile_state(self, compute_amount, read_weight_bytes, read_activation_bytes, write_activation_bytes): # 自动把layer_name和layer_type传入
        assert len(self.config._profile_state["layer_name"]) == len(self.config._profile_state["layer_type"]) == \
            len(self.config._profile_state["compute_amount"]) == len(self.config._profile_state["read_weight_bytes"]) == \
            len(self.config._profile_state["read_activation_bytes"]) == len(self.config._profile_state["write_activation_bytes"])
        self.config._profile_state["layer_name"].append(self.layer_name)
        self.config._profile_state["layer_type"].append(self.layer_type)
        self.config._profile_state["compute_amount"].append(compute_amount)
        self.config._profile_state["read_weight_bytes"].append(read_weight_bytes)
        self.config._profile_state["read_activation_bytes"].append(read_activation_bytes)
        self.config._profile_state["write_activation_bytes"].append(write_activation_bytes)


class EmbeddingLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name, vocab_size, hidden_size):
        BaseLayer.__init__(self, config, "Embedding", layer_name)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = LayerData((1, 1, vocab_size, hidden_size,), config.MISC_dtype)

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        batch, head, input_token_num, input_token_dim = input_tensor.shape
        assert head == 1
        assert input_token_dim == 1
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData((batch, head, input_token_num, self.hidden_size), output_dtype)
        self.update_profile_state(
            compute_amount          = 0, 
            read_weight_bytes       = output_tensor.bytes,
            read_activation_bytes   = 0,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class LinearLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name, in_features, out_features, bias_flag):
        BaseLayer.__init__(self, config, "Linear", layer_name)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = LayerData((1, 1, in_features, out_features), config.W_dtype)
        self.bias_flag = bias_flag
        if bias_flag:
            self.bias = LayerData((out_features,), config.bias_dtype)

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        batch, head, input_M, input_K = input_tensor.shape
        assert head == 1
        assert input_K == self.in_features
        assert input_tensor.dtype == self.config.A_dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData((batch, head, input_M, self.out_features), output_dtype)
        self.update_profile_state(
            compute_amount          = batch * head * input_M * self.in_features * self.out_features * 2, # 2 for MAC
            read_weight_bytes       = self.weight.bytes,
            read_activation_bytes   = input_tensor.bytes,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class RMSNormLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name, hidden_size, bias_flag):
        BaseLayer.__init__(self, config, "RMSNorm", layer_name)
        self.hidden_size = hidden_size
        assert self.config.MISC_dtype == "FP16"
        self.weight = LayerData((1, 1, 1, hidden_size), self.config.MISC_dtype) # FP16 for RMSNorm weights
        assert not bias_flag # no bias for RMSNorm

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        batch, head, input_M, input_K = input_tensor.shape
        assert head == 1
        assert input_K == self.hidden_size
        assert input_tensor.dtype == self.config.MISC_dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData(input_tensor.shape, output_dtype)
        self.update_profile_state(
            compute_amount          = input_tensor.data_num * 2, # RMSNorm近似为过两遍数据
            read_weight_bytes       = self.weight.bytes,
            read_activation_bytes   = input_tensor.bytes,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class RotaryLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "RMSNorm", layer_name)
        self.head_dim = config.HEAD_DIM
        assert self.config.MISC_dtype == "FP16"
        self.weight = LayerData((1, 1, 2, self.head_dim,), self.config.MISC_dtype) # FP16 for RMSNorm weights, 2 means (sin, cos)

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        batch, head, input_M, input_K = input_tensor.shape
        assert input_K == self.head_dim
        assert input_tensor.dtype == self.config.MISC_dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData(input_tensor.shape, output_dtype)
        self.update_profile_state(
            compute_amount          = input_tensor.data_num * 2, # Rotary近似为过两遍数据
            read_weight_bytes       = input_M * self.weight.bytes, # Rotary需要根据token位置取出对应cos和sin进行计算
            read_activation_bytes   = input_tensor.bytes,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class SiLULayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "SiLU", layer_name)
        # SiLU has no weight

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        batch, head, input_M, input_K = input_tensor.shape
        assert head == 1
        assert input_tensor.dtype == self.config.MISC_dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData(input_tensor.shape, output_dtype)
        self.update_profile_state(
            compute_amount          = input_tensor.data_num,
            read_weight_bytes       = 0,
            read_activation_bytes   = input_tensor.bytes,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class EltwiseLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "Eltwise", layer_name)
        # Eltwise has no weight

    def profile_forward(self, input_tensor_a: LayerData, input_tensor_b: LayerData, eltwise_type: str, output_dtype: str):
        assert len(input_tensor_a.shape) == len(input_tensor_b.shape) == 4
        assert input_tensor_a.dtype == input_tensor_b.dtype == self.config.MISC_dtype
        assert input_tensor_a.shape == input_tensor_b.shape
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        assert eltwise_type in ("mul", "add")
        output_tensor = LayerData(input_tensor_a.shape, output_dtype)
        self.update_profile_state(
            compute_amount          = input_tensor_a.data_num,
            read_weight_bytes       = 0,
            read_activation_bytes   = input_tensor_a.bytes + input_tensor_b.bytes,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class SoftmaxLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "Softmax", layer_name)
        # Softmax has no weight

    def profile_forward(self, input_tensor: LayerData, output_dtype: str):
        assert input_tensor.dtype == self.config.MISC_dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData(input_tensor.shape, output_dtype)
        self.update_profile_state(
            compute_amount          = input_tensor.data_num * 2, # Softmax近似为过两遍数据
            read_weight_bytes       = 0,
            read_activation_bytes   = input_tensor.bytes,
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class MatmulLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name, num_key_value_groups, bias_flag):
        BaseLayer.__init__(self, config, "Matmul", layer_name)
        assert not bias_flag
        self.num_key_value_groups = num_key_value_groups
        assert num_key_value_groups == config.num_key_value_groups
        # Matmul has no weight

    def profile_forward(self, input_tensor_a: LayerData, input_tensor_b: LayerData, output_dtype: str):
        batch_a, head_a, input_M, input_K_a = input_tensor_a.shape
        batch_b, head_b, input_N, input_K_b = input_tensor_b.shape
        assert batch_a == batch_b
        assert head_a == head_b * self.num_key_value_groups
        assert input_K_a == input_K_b
        assert input_tensor_a.dtype == self.config.A_dtype
        assert input_tensor_b.dtype == self.config.KV_dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData((batch_a, head_a, input_M, input_N), output_dtype)
        self.update_profile_state(
            compute_amount          = batch_a * head_a * input_M * input_K_a * input_N * 2, # 2 for MAC
            read_weight_bytes       = 0,
            read_activation_bytes   = input_tensor_a.bytes + input_tensor_b.bytes, # GQA降低读取量
            write_activation_bytes  = output_tensor.bytes,
        )
        return output_tensor


class ConcatLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "Concat", layer_name)
        # Concat has no weight

    def profile_forward(self, input_tensor_a: LayerData, input_tensor_b: LayerData, dim: int): # Concat do not change dtype
        assert input_tensor_a.dtype == input_tensor_b.dtype == self.config.KV_dtype
        output_dtype = input_tensor_a.dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        for i in range(len(input_tensor_a.shape)):
            if i != dim:
                assert input_tensor_a.shape[i] == input_tensor_b.shape[i]
            else:
                assert input_tensor_b.shape[i] <= input_tensor_a.shape[i] # concat KV cache应该是新的往旧的（多）的上拼
        output_shape = list(input_tensor_a.shape)
        output_shape[dim] = input_tensor_a.shape[dim] + input_tensor_b.shape[dim]
        output_tensor = LayerData(tuple(output_shape), output_dtype)
        self.update_profile_state(
            compute_amount          = 0,
            read_weight_bytes       = 0,
            read_activation_bytes   = input_tensor_b.bytes,
            write_activation_bytes  = input_tensor_b.bytes,
        )
        return output_tensor


class TransposeLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "Transpose", layer_name)
        # Transpose has no weight

    def profile_forward(self, input_tensor: LayerData, dim: tuple): # Concat do not change dtype
        output_dtype = input_tensor.dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        assert len(dim) == 2 and dim[0] != dim[1]
        dim = dim
        output_shape = list(input_tensor.shape)
        output_shape[dim[0]], output_shape[dim[1]] = output_shape[dim[1]], output_shape[dim[0]]
        need_transpose = (LayerData.squeeze_shape(input_tensor.shape) != LayerData.squeeze_shape(output_shape)) # 不改变数据排布
        if not need_transpose:
            need_transpose = False 
            # print(f"Warning: transpose layer {self.layer_name} does not change data layout!")
        output_tensor = LayerData(tuple(output_shape), output_dtype)
        self.update_profile_state(
            compute_amount          = 0,
            read_weight_bytes       = 0,
            read_activation_bytes   = input_tensor.bytes if need_transpose else 0, # 不改变数据排布则无需额外读写
            write_activation_bytes  = output_tensor.bytes if need_transpose else 0,
        )
        return output_tensor


class ViewLayer(BaseLayer):
    def __init__(self, config: ModelConfig, layer_name):
        BaseLayer.__init__(self, config, "View", layer_name)
        # View has no weight

    def profile_forward(self, input_tensor: LayerData, output_shape: tuple): # View do not change dtype
        output_dtype = input_tensor.dtype
        assert output_dtype in (self.config.A_dtype, self.config.KV_dtype, self.config.MISC_dtype)
        output_tensor = LayerData(tuple(output_shape), output_dtype)
        assert input_tensor.data_num == output_tensor.data_num
        assert len(input_tensor.shape) == len(output_shape)
        changed_dims = [i for i, (a, b) in enumerate(zip(input_tensor.shape, output_shape)) if a != b] # 计算发生变化的维度数量
        is_reasonable = (len(changed_dims) == 2 and abs(changed_dims[0] - changed_dims[1]) == 1) or (len(changed_dims) == 0) # 判断是否只有两个维度发生变化且这两个维度是连续的，或者完全相同
        assert is_reasonable, "View layer is not reasonable!"
        self.update_profile_state(
            compute_amount          = 0,
            read_weight_bytes       = 0,
            read_activation_bytes   = 0,
            write_activation_bytes  = 0,
        )
        return output_tensor


class LlamaForCausalLM():
    def __init__(self, model_name, W_dtype, A_dtype, KV_dtype, MISC_dtype="FP16"):
        config = ModelConfig(model_name, W_dtype, A_dtype, KV_dtype, MISC_dtype)
        assert not config.BIAS_FLAG
        self.config = config
        
        self.embed_tokens = EmbeddingLayer(
            config          = config,
            layer_name      = f"embed_tokens",
            vocab_size      = config.VOCAB_SIZE,
            hidden_size     = config.HIDDEN_SIZE,
        )
        
        self.input_layernorm_list = list()
        self.q_proj_list = list()
        self.k_proj_list = list()
        self.v_proj_list = list()
        self.view_q_list = list()
        self.view_k_list = list()
        self.view_v_list = list()
        self.rotary_pos_emb_q = list()
        self.rotary_pos_emb_k = list()
        self.transpose_q_list = list()
        self.transpose_k_list = list()
        self.transpose_v_list = list()
        self.transpose_v_cache_list = list()
        self.concat_k_list = list()
        self.concat_v_list = list()
        self.attn_qkt_list = list()
        self.attn_softmax_list = list()
        self.attn_sv_list = list()
        self.transpose_attn_list = list()
        self.view_post_attn_list = list()
        self.o_proj_list = list()
        self.post_attn_add_list = list()
        self.up_proj_list = list()
        self.gate_proj_list = list()
        self.down_proj_list = list()
        self.act_fn_list = list()
        self.ffn_mul_list = list()
        self.ffn_add_list = list()
        self.post_attention_layernorm_list = list()

        for i in range(config.NUM_HIDDEN_LAYERS):
            self.input_layernorm_list.append(
                RMSNormLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_input_layernorm",
                    hidden_size     = config.HIDDEN_SIZE,
                    bias_flag       = False,
                )
            )
            self.q_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_q_proj",
                    in_features     = config.HIDDEN_SIZE,
                    out_features    = config.NUM_ATTENTION_HEADS * config.HEAD_DIM,
                    bias_flag       = False,
                )
            )
            self.k_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_k_proj",
                    in_features     = config.HIDDEN_SIZE,
                    out_features    = config.NUM_KEY_VALUE_HEADS * config.HEAD_DIM,
                    bias_flag       = False,
                )
            )
            self.v_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_v_proj",
                    in_features     = config.HIDDEN_SIZE,
                    out_features    = config.NUM_KEY_VALUE_HEADS * config.HEAD_DIM,
                    bias_flag       = False,
                )
            )
            self.view_q_list.append(
                ViewLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_view_q",
                )
            )
            self.view_k_list.append(
                ViewLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_view_k",
                )
            )
            self.view_v_list.append(
                ViewLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_view_v",
                )
            )
            self.rotary_pos_emb_q.append(
                RotaryLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_rotary_pos_emb_q",
                )
            )
            self.rotary_pos_emb_k.append(
                RotaryLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_rotary_pos_emb_k",
                )
            )
            self.transpose_q_list.append(
                TransposeLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_transpose_q",
                )
            )
            self.transpose_k_list.append(
                TransposeLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_transpose_k",
                )
            )
            self.transpose_v_list.append(
                TransposeLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_transpose_v",
                )
            )
            self.transpose_v_cache_list.append(
                TransposeLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_transpose_v_cache",
                )
            )
            self.concat_k_list.append(
                ConcatLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_concat_k_cache",
                )
            )
            self.concat_v_list.append(
                ConcatLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_concat_v_cache",
                )
            )
            self.attn_qkt_list.append(
                MatmulLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_attn_qkt",
                    num_key_value_groups = config.num_key_value_groups,
                    bias_flag       = False,
                )
            )
            self.attn_softmax_list.append(
                SoftmaxLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_attn_softmax",
                )
            )
            self.attn_sv_list.append(
                MatmulLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_attn_sv",
                    num_key_value_groups = config.num_key_value_groups,
                    bias_flag       = False,
                )
            )
            self.transpose_attn_list.append(
                TransposeLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_transpose_attn",
                )
            )
            self.view_post_attn_list.append(
                ViewLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_view_post_attn",
                )
            )
            self.o_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_o_proj",
                    in_features     = config.NUM_ATTENTION_HEADS * config.HEAD_DIM,
                    out_features    = config.HIDDEN_SIZE,
                    bias_flag       = False,
                )
            )
            self.post_attn_add_list.append(
                EltwiseLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_ffn_add",
                )
            )
            self.gate_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_gate_proj",
                    in_features     = config.HIDDEN_SIZE,
                    out_features    = config.INTERMEDIATE_SIZE,
                    bias_flag       = False,
                )
            )
            self.up_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_up_proj",
                    in_features     = config.HIDDEN_SIZE,
                    out_features    = config.INTERMEDIATE_SIZE,
                    bias_flag       = False,
                )
            )
            self.down_proj_list.append(
                LinearLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_down_proj",
                    in_features     = config.INTERMEDIATE_SIZE,
                    out_features    = config.HIDDEN_SIZE,
                    bias_flag       = False,
                )
            )
            self.act_fn_list.append(
                SiLULayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_act_fn",
                )
            )
            self.ffn_mul_list.append(
                EltwiseLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_ffn_mul",
                )
            )
            self.ffn_add_list.append(
                EltwiseLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_ffn_add",
                )
            )
            self.post_attention_layernorm_list.append(
                RMSNormLayer(
                    config          = config,
                    layer_name      = f"decoder_{i}_post_attention_layernorm",
                    hidden_size     = config.HIDDEN_SIZE,
                    bias_flag       = False,
                )
            )

        self.final_layernorm = RMSNormLayer(
            config          = config,
            layer_name      = "final_layernorm",
            hidden_size     = config.HIDDEN_SIZE,
            bias_flag       = False,
        )
        self.lm_head = LinearLayer(
            config          = config, 
            layer_name      = "lm_head",
            in_features     = config.HIDDEN_SIZE,
            out_features    = config.VOCAB_SIZE,
            bias_flag       = False,
        )


    def forward(self, batch, input_tokens, kv_cache_len):
        self.config.init_profile_state() # 清空profile记录，从而实现单次init model后可以多次forward

        if kv_cache_len == 0: # 没有KV cache
            stage_name = "prefill"
        else: # 存在KV cache
            assert input_tokens == 1 # decode
            stage_name = "decode"

        input_ids = LayerData((batch, 1, input_tokens, 1), self.config.MISC_dtype)
        # embedding
        hidden_states = self.embed_tokens.profile_forward(input_ids, self.config.MISC_dtype)
        # decoders
        for i in range(self.config.NUM_HIDDEN_LAYERS):
            # input layernrom
            hidden_states_layernorm = self.input_layernorm_list[i].profile_forward(hidden_states, self.config.A_dtype)
            # generate q, k, v
            query           = self.q_proj_list[i].profile_forward(hidden_states_layernorm, self.config.MISC_dtype)
            key             = self.k_proj_list[i].profile_forward(hidden_states_layernorm, self.config.MISC_dtype)
            value           = self.v_proj_list[i].profile_forward(hidden_states_layernorm, self.config.KV_dtype)
            # view q, k, v as multi head, split to 2 steps to complete the view
            query           = self.view_q_list[i].profile_forward(query, (batch, input_tokens, 1, self.config.NUM_ATTENTION_HEADS * self.config.HEAD_DIM))
            key             = self.view_k_list[i].profile_forward(key, (batch, input_tokens, 1, self.config.NUM_KEY_VALUE_HEADS * self.config.HEAD_DIM))
            value           = self.view_v_list[i].profile_forward(value, (batch, input_tokens, 1, self.config.NUM_KEY_VALUE_HEADS * self.config.HEAD_DIM))
            query           = self.view_q_list[i].profile_forward(query, (batch, input_tokens, self.config.NUM_ATTENTION_HEADS, self.config.HEAD_DIM))
            key             = self.view_k_list[i].profile_forward(key, (batch, input_tokens, self.config.NUM_KEY_VALUE_HEADS, self.config.HEAD_DIM))
            value           = self.view_v_list[i].profile_forward(value, (batch, input_tokens, self.config.NUM_KEY_VALUE_HEADS, self.config.HEAD_DIM))
            # transpose q, k, v to change the num_heads and input_tokens
            query           = self.transpose_q_list[i].profile_forward(query, dim=(1, 2))
            key             = self.transpose_k_list[i].profile_forward(key, dim=(1, 2))
            value           = self.transpose_v_list[i].profile_forward(value, dim=(1, 2))
            # rope embedding of q, k
            query           = self.rotary_pos_emb_q[i].profile_forward(query, self.config.A_dtype)
            key             = self.rotary_pos_emb_k[i].profile_forward(key, self.config.KV_dtype)
            # concat kv cache
            if kv_cache_len != 0:
                k_cache     = LayerData((batch, self.config.NUM_KEY_VALUE_HEADS, kv_cache_len, self.config.HEAD_DIM), self.config.KV_dtype)
                v_cache     = LayerData((batch, self.config.NUM_KEY_VALUE_HEADS, kv_cache_len, self.config.HEAD_DIM), self.config.KV_dtype)
                key         = self.concat_k_list[i].profile_forward(k_cache, key, dim=2)
                value       = self.concat_v_list[i].profile_forward(v_cache, value, dim=2)
            # transpose v to compute attention
            value           = self.transpose_v_cache_list[i].profile_forward(value, dim=(2, 3))
            # attention
            attn_qkt        = self.attn_qkt_list[i].profile_forward(query, key, self.config.MISC_dtype)
            attn_scores     = self.attn_softmax_list[i].profile_forward(attn_qkt, self.config.A_dtype)
            attn_sv         = self.attn_sv_list[i].profile_forward(attn_scores, value, self.config.A_dtype)
            attn_output     = self.transpose_attn_list[i].profile_forward(attn_sv, dim=(1, 2))
            attn_output     = self.view_post_attn_list[i].profile_forward(attn_output, (batch, input_tokens, 1, self.config.HIDDEN_SIZE))
            attn_output     = self.view_post_attn_list[i].profile_forward(attn_output, (batch, 1, input_tokens, self.config.HIDDEN_SIZE)) # 2 steps to complete the view
            attn_output     = self.o_proj_list[i].profile_forward(attn_output, self.config.MISC_dtype)
            attn_output     = self.post_attn_add_list[i].profile_forward(hidden_states, attn_output, "add", self.config.MISC_dtype)
            # ffn
            post_attn_norm  = self.post_attention_layernorm_list[i].profile_forward(attn_output, self.config.A_dtype)
            up              = self.up_proj_list[i].profile_forward(post_attn_norm, self.config.MISC_dtype)
            gate            = self.gate_proj_list[i].profile_forward(post_attn_norm, self.config.MISC_dtype)
            silu            = self.act_fn_list[i].profile_forward(gate, self.config.MISC_dtype)
            ffn_mul         = self.ffn_mul_list[i].profile_forward(up, silu, "mul", self.config.A_dtype)
            down            = self.down_proj_list[i].profile_forward(ffn_mul, self.config.MISC_dtype)
            hidden_states   = self.ffn_add_list[i].profile_forward(down, attn_output, "add", self.config.MISC_dtype)

        final_layernorm     = self.final_layernorm.profile_forward(hidden_states, self.config.A_dtype)
        logits              = self.lm_head.profile_forward(final_layernorm, self.config.MISC_dtype)

        # print(self.config._profile_state)
        total_compute_amount = sum(self.config._profile_state["compute_amount"])
        total_matrix_compute = 0
        total_vector_compute = 0
        for i, this_layer_type in enumerate(self.config._profile_state["layer_type"]):
            if this_layer_type in ("Linear", "Matmul"):
                total_matrix_compute += self.config._profile_state["compute_amount"][i]
            else:
                total_vector_compute += self.config._profile_state["compute_amount"][i]

        total_read_weight_bytes = sum(self.config._profile_state["read_weight_bytes"])
        total_read_activation_bytes = sum(self.config._profile_state["read_activation_bytes"])
        total_write_activation_bytes = sum(self.config._profile_state["write_activation_bytes"])
        total_profile_info = {
            "total_compute_amount": total_compute_amount,
            "total_matrix_compute_amount": total_matrix_compute,
            "total_vector_compute_amount": total_vector_compute,
            "total_read_weight_bytes": total_read_weight_bytes,
            "total_read_activation_bytes": total_read_activation_bytes,
            "total_write_activation_bytes": total_write_activation_bytes,
            "total_memory_access": total_read_weight_bytes + total_read_activation_bytes + total_write_activation_bytes,
        }
        return total_profile_info


def profile_model_end_to_end(model: LlamaForCausalLM, batch, prefill_size, decode_size):
    assert prefill_size > 0
    assert decode_size >= 0
    # prefill
    prefill_profile_info = model.forward(
        batch           = batch,
        input_tokens    = prefill_size,
        kv_cache_len    = 0,
    )
    # decode
    decode_profile_info = dict()
    for key in prefill_profile_info.keys():
        decode_profile_info[key] = 0

    for out_id in range(decode_size):
        this_decode_profile_info = model.forward(
            batch           = batch,
            input_tokens    = 1,
            kv_cache_len    = prefill_size + out_id,
        )
        for key in prefill_profile_info.keys():
            decode_profile_info[key] += this_decode_profile_info[key]
    
    decode_per_token_profile_info = dict()
    end_to_end_profile_info = dict()
    for key in prefill_profile_info.keys():
        decode_per_token_profile_info[key] = decode_profile_info[key] / decode_size
        end_to_end_profile_info[key] = prefill_profile_info[key] + decode_profile_info[key]

    print("==========================================================")
    print("  Model_name:                           %s"        % (model.config.model_name))
    print("  Weight, Act, KV_cache:        [%6s, %6s, %6s]"   % (model.config.W_dtype, model.config.A_dtype, model.config.KV_dtype))
    print("  Batch, Prefill, Decode:       [%6d, %6d, %6d]"   % (batch, prefill_size, decode_size))
    print("----------------------------------------------------------")
    print("  Prefill_computation:             %13.6f GOPs"    % (prefill_profile_info["total_compute_amount"] / 1e9))
    print("  Prefill_matrix_computation:      %13.6f GOPs"    % (prefill_profile_info["total_matrix_compute_amount"] / 1e9))
    print("  Prefill_vector_computation:      %13.6f GOPs"    % (prefill_profile_info["total_vector_compute_amount"] / 1e9))
    print("  Prefill_memory_access:           %13.6f GB"      % (prefill_profile_info["total_memory_access"] / 1024 / 1024 / 1024))
    print("  --Prefill_read_weight:           %13.6f GB"      % (prefill_profile_info["total_read_weight_bytes"] / 1024 / 1024 / 1024))
    print("  --Prefill_read_act:              %13.6f GB"      % (prefill_profile_info["total_read_activation_bytes"] / 1024 / 1024 / 1024))
    print("  --Prefill_write_act:             %13.6f GB"      % (prefill_profile_info["total_write_activation_bytes"] / 1024 / 1024 / 1024))
    print("  Prefill_intensity:               %13.6f OP/Byte" % (prefill_profile_info["total_compute_amount"] / prefill_profile_info["total_memory_access"]))
    print("----------------------------------------------------------")
    print("  Decode_computation:              %13.6f GOPs"    % (decode_profile_info["total_compute_amount"] / 1e9))
    print("  Decode_matrix_computation:       %13.6f GOPs"    % (decode_profile_info["total_matrix_compute_amount"] / 1e9))
    print("  Decode_vector_computation:       %13.6f GOPs"    % (decode_profile_info["total_vector_compute_amount"] / 1e9))
    print("  Decode_memory_access:            %13.6f GB"      % (decode_profile_info["total_memory_access"] / 1024 / 1024 / 1024))
    print("  --Decode_read_weight:            %13.6f GB"      % (decode_profile_info["total_read_weight_bytes"] / 1024 / 1024 / 1024))
    print("  --Decode_read_act:               %13.6f GB"      % (decode_profile_info["total_read_activation_bytes"] / 1024 / 1024 / 1024))
    print("  --Decode_write_act:              %13.6f GB"      % (decode_profile_info["total_write_activation_bytes"] / 1024 / 1024 / 1024))
    print("  Decode_intensity:                %13.6f OP/Byte" % (decode_profile_info["total_compute_amount"] / decode_profile_info["total_memory_access"]))
    print("----------------------------------------------------------")
    print("  Decode_computation_per_token:    %13.6f GOPs"    % (decode_per_token_profile_info["total_compute_amount"] / 1e9))
    print("  Decode_memory_access_per_token:  %13.6f GB"      % (decode_per_token_profile_info["total_memory_access"] / 1024 / 1024 / 1024))
    print("  --Decode_read_weight_per_token:  %13.6f GB"      % (decode_per_token_profile_info["total_read_weight_bytes"] / 1024 / 1024 / 1024))
    print("  --Decode_read_act_per_token:     %13.6f GB"      % (decode_per_token_profile_info["total_read_activation_bytes"] / 1024 / 1024 / 1024))
    print("  --Decode_write_act_per_token:    %13.6f GB"      % (decode_per_token_profile_info["total_write_activation_bytes"] / 1024 / 1024 / 1024))
    print("  Decode_intensity_per_token:      %13.6f OP/Byte" % (decode_per_token_profile_info["total_compute_amount"] / decode_per_token_profile_info["total_memory_access"]))
    print("----------------------------------------------------------")
    print("  End_to_end_computation:          %13.6f GOPs"    % (end_to_end_profile_info["total_compute_amount"] / 1e9))
    print("  End_to_end_memory_access:        %13.6f GB"      % (end_to_end_profile_info["total_memory_access"] / 1024 / 1024 / 1024))
    print("  --End_to_end_read_weight:        %13.6f GB"      % (end_to_end_profile_info["total_read_weight_bytes"] / 1024 / 1024 / 1024))
    print("  --End_to_end_read_act:           %13.6f GB"      % (end_to_end_profile_info["total_read_activation_bytes"] / 1024 / 1024 / 1024))
    print("  --End_to_end_write_act:          %13.6f GB"      % (end_to_end_profile_info["total_write_activation_bytes"] / 1024 / 1024 / 1024))
    print("  End_to_end_intensity:            %13.6f OP/Byte" % (end_to_end_profile_info["total_compute_amount"] / end_to_end_profile_info["total_memory_access"]))
    print("==========================================================")


def profile_model_single_step(model: LlamaForCausalLM, batch, input_tokens, kv_cache_len):
    profile_info = model.forward(
        batch           = batch,
        input_tokens    = input_tokens,
        kv_cache_len    = kv_cache_len,
    )
    print("==========================================================")
    print("  Model_name:                           %s"        % (model.config.model_name))
    print("  Weight, Act, KV_cache:        [%6s, %6s, %6s]"   % (model.config.W_dtype, model.config.A_dtype, model.config.KV_dtype))
    print("  Batch, Input, KV_cache:       [%6d, %6d, %6d]"   % (batch, input_tokens, kv_cache_len))
    print("----------------------------------------------------------")
    print("  Computation:                     %13.6f GOPs"    % (profile_info["total_compute_amount"] / 1e9))
    print("  Memory_access:                   %13.6f GB"      % (profile_info["total_memory_access"] / 1024 / 1024 / 1024))
    print("  --Read_weight:                   %13.6f GB"      % (profile_info["total_read_weight_bytes"] / 1024 / 1024 / 1024))
    print("  --Read_activation:               %13.6f GB"      % (profile_info["total_read_activation_bytes"] / 1024 / 1024 / 1024))
    print("  --Write_activation:              %13.6f GB"      % (profile_info["total_write_activation_bytes"] / 1024 / 1024 / 1024))
    print("  Intensity:                       %13.6f OP/Byte" % (profile_info["total_compute_amount"] / profile_info["total_memory_access"]))
    print("==========================================================")


if __name__ == "__main__":
    model = LlamaForCausalLM(
        model_name      = "Llama-3.1-70B",
        W_dtype         = "INT4.537667411", # 8 channel 3bit
        A_dtype         = "INT12",
        KV_dtype        = "INT4.925030048076923", # 0 channel 3bit
        MISC_dtype      = "FP16",
    )
    # profile_model_single_step(
    #     model           = model,
    #     batch           = 2,
    #     input_tokens    = 1,
    #     kv_cache_len    = 10,
    # )
    for batch_size in [96]:
        profile_model_end_to_end(
            model           = model,
            batch           = batch_size,
            prefill_size    = 1024,
            decode_size     = 1024,
        )