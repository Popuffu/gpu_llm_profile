import numpy
import math

class Config:
    def __init__(self, Model_name, Hardware_name, Prefill_size, Decode_size):
        assert Prefill_size > 0
        assert Decode_size >= 0

        self.Prefill_size = Prefill_size
        self.Decode_size = Decode_size
        self.Model_name = Model_name
        self.Hardware_name = Hardware_name
        
        if self.Model_name == "LLaMA2-7B":
            self.Embedding_flag = False
            self.Bias_flag = False
            self.Layer_num = 32
            self.Embedding_size = 4096
            self.Vocabulary_size = 32000
            self.Attention_head = 32
            self.Attention_feature_size = self.Embedding_size // self.Attention_head
            self.FFN_times = 11008 / self.Embedding_size
        else:
            raise ValueError("cfg.Model_name not found")
        
        if Hardware_name == "GPU":
            self.Weight_width = 2 # FP16
            self.Activation_width = 2 # FP16
            self.WAKV_bitwidth = "W16A16KV16"
        elif Hardware_name == "FPGA":
            self.Weight_width = 0.5 # W4
            self.Activation_width = 1 # A8
            self.WAKV_bitwidth = "W4A8KV8"
        else:
            raise ValueError("cfg.Hardware_name not found")



class Stage:
    def __init__(self, name, platform="cloud"):
        self.name = name
        self.platform = platform

        self.read_data_amount = 0
        self.compute_amount = 0
        self.write_data_amount = 0

        self.memory_usage = 0
        self.onchip_memory_usage = 0  # 只记录主要线性层或者attention等


def Prefill(cfg):
    Stage_list = list()

    Weight_memory_usage = cfg.Weight_width * (
        cfg.Layer_num * (
            cfg.Embedding_size * cfg.Embedding_size * 3
            + cfg.Embedding_size * 3
            + cfg.Embedding_size * cfg.Embedding_size
            + cfg.Embedding_size
            + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times
            + cfg.Embedding_size * cfg.FFN_times
            + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times
            + cfg.Embedding_size + cfg.Embedding_size * 2
            + cfg.Embedding_size * 2
            )
        + cfg.Embedding_size * 2
        + cfg.Embedding_size * cfg.Vocabulary_size
    )
    KV_cache_init_usage = cfg.Activation_width * cfg.Layer_num * (cfg.Prefill_size * cfg.Embedding_size * 2)

    if cfg.Embedding_flag:
        embedding1 = Stage("Embedding", platform="cloud")
        embedding1.read_data_amount = (cfg.Prefill_size + cfg.Prefill_size * cfg.Embedding_size) * cfg.Activation_width
        embedding1.compute_amount = cfg.Prefill_size * cfg.Embedding_size  # 加位置编码
        embedding1.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        embedding1.memory_usage = Weight_memory_usage + embedding1.write_data_amount

        Stage_list.append(embedding1)
    
    for i in range(cfg.Layer_num):
        RMSLayerNorm1 = Stage("RMSLayerNorm", platform="cloud")
        RMSLayerNorm1.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
                                          + cfg.Embedding_size * cfg.Weight_width * (1 + cfg.Bias_flag))
        RMSLayerNorm1.compute_amount = cfg.Prefill_size * cfg.Embedding_size * 2
        RMSLayerNorm1.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        RMSLayerNorm1.memory_usage = (Weight_memory_usage
                                      + cfg.Activation_width * i * cfg.Prefill_size * cfg.Embedding_size * 2
                                      + RMSLayerNorm1.write_data_amount)
        Stage_list.append(RMSLayerNorm1)

        QKV_MM = Stage("QKV_MM", platform="cloud")
        QKV_MM.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width + (cfg.Embedding_size * cfg.Embedding_size) * cfg.Weight_width) * 3
        QKV_MM.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Embedding_size * 2 * 3   #没考虑bias是device_dict["cloud"].mfu里自带可以加bias模块，方便计算
        QKV_MM.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width * 3
        QKV_MM.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + cfg.Activation_width * cfg.Prefill_size * cfg.Embedding_size
        QKV_MM.onchip_memory_usage = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Embedding_size * cfg.Weight_width + QKV_MM.write_data_amount
        Stage_list.append(QKV_MM)
        
        rope = Stage("rope_embedding_QK", platform="cloud")
        rope.read_data_amount = (2 * cfg.Attention_feature_size * cfg.Prefill_size * cfg.Weight_width
                                 + 2 * cfg.Attention_head * cfg.Attention_feature_size * cfg.Prefill_size * cfg.Activation_width)
        rope.compute_amount = 2 * 2 * cfg.Attention_head * cfg.Attention_feature_size * cfg.Prefill_size
        rope.write_data_amount = 2 * cfg.Attention_head * cfg.Attention_feature_size * cfg.Prefill_size  # 写入memory
        rope.memory_usage = (Weight_memory_usage
                             + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                             + cfg.Activation_width * cfg.Prefill_size * cfg.Embedding_size * 2)
        Stage_list.append(rope)

        QKV_Transpose1 = Stage("V_Transpose", platform="cloud")
        QKV_Transpose1.read_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        QKV_Transpose1.compute_amount = 0
        QKV_Transpose1.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        QKV_Transpose1.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + QKV_Transpose1.write_data_amount
        Stage_list.append(QKV_Transpose1)

        QK_MM = Stage("QK_MM", platform="cloud")
        QK_MM.read_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Attention_feature_size * cfg.Activation_width * 2
        QK_MM.compute_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Attention_feature_size * cfg.Prefill_size * 2
        QK_MM.write_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Prefill_size * cfg.Activation_width
        QK_MM.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + QK_MM.write_data_amount
        QK_MM.onchip_memory_usage = QK_MM.read_data_amount + QK_MM.write_data_amount + cfg.Attention_head * cfg.Prefill_size * cfg.Attention_feature_size * cfg.Activation_width   # 额外存了一份V
        Stage_list.append(QK_MM)

        Softmax = Stage("Softmax", platform="cloud")
        Softmax.read_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Prefill_size * cfg.Activation_width
        Softmax.compute_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Prefill_size
        Softmax.write_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Prefill_size * cfg.Activation_width
        #这个ex、加法是由device_dict["cloud"].vfu和device_dict["cloud"].vfu_device_dict["cloud"].sfu完成的，然后除法由单独由device_dict["cloud"].vfu完成，目前问题在于是否要反复读上来？？？
        Softmax.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + Softmax.write_data_amount
        Stage_list.append(Softmax)

        AttenV = Stage("AttenV", platform="cloud")
        AttenV.read_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Prefill_size * cfg.Activation_width + cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        AttenV.compute_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Prefill_size * cfg.Attention_feature_size * 2
        AttenV.write_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Attention_feature_size * cfg.Activation_width
        AttenV.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + AttenV.write_data_amount
        AttenV.onchip_memory_usage = AttenV.read_data_amount + AttenV.write_data_amount
        Stage_list.append(AttenV)

        Atten_Transpose = Stage("Atten_Transpose", platform="cloud")
        Atten_Transpose.read_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Attention_feature_size * cfg.Activation_width
        Atten_Transpose.compute_amount = 0
        Atten_Transpose.write_data_amount = cfg.Attention_head * cfg.Prefill_size * cfg.Attention_feature_size * cfg.Activation_width
        Atten_Transpose.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + Atten_Transpose.write_data_amount
        Stage_list.append(Atten_Transpose)

        Output_MM = Stage("Output_MM", platform="cloud")
        Output_MM.read_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Embedding_size * cfg.Weight_width
        Output_MM.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Embedding_size * 2
        Output_MM.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        Output_MM.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + Output_MM.write_data_amount
        Output_MM.onchip_memory_usage = Output_MM.read_data_amount + Output_MM.write_data_amount
        Stage_list.append(Output_MM)

        Atten_Element_Add = Stage("Atten_Element_Add", platform="cloud")
        Atten_Element_Add.read_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width + cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        Atten_Element_Add.compute_amount = cfg.Prefill_size * cfg.Embedding_size
        Atten_Element_Add.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        Atten_Element_Add.memory_usage = Weight_memory_usage + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2 + Atten_Element_Add.write_data_amount
        Stage_list.append(Atten_Element_Add)
        
        FFN_RMSLayerNorm0 = Stage("FFN_RMSLayerNorm0", platform="cloud")
        FFN_RMSLayerNorm0.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
                                              + cfg.Embedding_size * cfg.Weight_width)
        FFN_RMSLayerNorm0.compute_amount = cfg.Prefill_size * cfg.Embedding_size * 2
        FFN_RMSLayerNorm0.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        FFN_RMSLayerNorm0.memory_usage = (Weight_memory_usage
                                          + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                                          + FFN_RMSLayerNorm0.write_data_amount)
        Stage_list.append(FFN_RMSLayerNorm0)

        FFN_MM_up = Stage("FFN_MM_up", platform="cloud")
        FFN_MM_up.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
                                      + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * cfg.Weight_width)
        FFN_MM_up.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * 2
        FFN_MM_up.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
        FFN_MM_up.memory_usage = (Weight_memory_usage
                                  + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                                  + FFN_MM_up.write_data_amount)
        FFN_MM_up.onchip_memory_usage = FFN_MM_up.read_data_amount + FFN_MM_up.write_data_amount
        Stage_list.append(FFN_MM_up)

        FFN_MM_gate = Stage("FFN_MM_gate", platform="cloud")
        FFN_MM_gate.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
                                        + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * cfg.Weight_width)
        FFN_MM_gate.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * 2
        FFN_MM_gate.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
        FFN_MM_gate.memory_usage = (Weight_memory_usage
                                    + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                                    + FFN_MM_gate.write_data_amount
                                    + FFN_MM_up.write_data_amount)
        FFN_MM_gate.onchip_memory_usage = FFN_MM_gate.read_data_amount + FFN_MM_gate.write_data_amount
        Stage_list.append(FFN_MM_gate)

        FFN_Element_Mul = Stage("FFN_Element_Mul", platform="cloud")
        FFN_Element_Mul.read_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width * 2
        FFN_Element_Mul.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.FFN_times * 2
        FFN_Element_Mul.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
        FFN_Element_Mul.memory_usage = (Weight_memory_usage
                                        + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                                        + FFN_Element_Mul.write_data_amount)
        Stage_list.append(FFN_Element_Mul)

        FFN_MM_down = Stage("FFN_MM_down", platform="cloud")
        FFN_MM_down.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
                                        + cfg.Embedding_size * cfg.FFN_times * cfg.Embedding_size * cfg.Weight_width)
        FFN_MM_down.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * 2
        FFN_MM_down.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        FFN_MM_down.memory_usage = (Weight_memory_usage
                                    + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                                    + FFN_MM_down.write_data_amount)
        FFN_MM_down.onchip_memory_usage = FFN_MM_down.read_data_amount + FFN_MM_down.write_data_amount
        Stage_list.append(FFN_MM_down)

        FFN_Element_Add = Stage("FFN_Element_Add", platform="cloud")
        FFN_Element_Add.read_data_amount = (cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
                                            + cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width)
        FFN_Element_Add.compute_amount = cfg.Prefill_size * cfg.Embedding_size
        FFN_Element_Add.write_data_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Activation_width
        FFN_Element_Add.memory_usage = (Weight_memory_usage
                                        + cfg.Activation_width * (i + 1) * cfg.Prefill_size * cfg.Embedding_size * 2
                                        + FFN_Element_Add.write_data_amount)
        Stage_list.append(FFN_Element_Add)

    Last_RMSLayerNorm = Stage("Last_RMSLayerNorm", platform="cloud")
    Last_RMSLayerNorm.read_data_amount = cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Weight_width
    Last_RMSLayerNorm.compute_amount = cfg.Embedding_size * 2
    Last_RMSLayerNorm.write_data_amount = cfg.Embedding_size * cfg.Activation_width
    Last_RMSLayerNorm.memory_usage = (Weight_memory_usage
                                      + cfg.Activation_width * cfg.Layer_num * cfg.Prefill_size * cfg.Embedding_size * 2
                                      + Last_RMSLayerNorm.write_data_amount)
    Stage_list.append(Last_RMSLayerNorm)
    
    Last_MM = Stage("Last_MM", platform="cloud")
    Last_MM.read_data_amount = cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Vocabulary_size * cfg.Weight_width
    Last_MM.compute_amount = cfg.Embedding_size * cfg.Vocabulary_size * 2
    Last_MM.write_data_amount = cfg.Vocabulary_size * cfg.Activation_width
    Last_MM.memory_usage = (Weight_memory_usage
                            + cfg.Activation_width * cfg.Layer_num * cfg.Prefill_size * cfg.Embedding_size * 2
                            + Last_MM.write_data_amount)
    Last_MM.onchip_memory_usage = Last_MM.read_data_amount + Last_MM.write_data_amount
    Stage_list.append(Last_MM)
    return Stage_list


def Decode(cfg):
    Stage_list = list()

    Weight_memory_usage = cfg.Weight_width * (
        cfg.Layer_num * (
            cfg.Embedding_size * cfg.Embedding_size * 3
            + cfg.Embedding_size * 3
            + cfg.Embedding_size * cfg.Embedding_size
            + cfg.Embedding_size
            + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times
            + cfg.Embedding_size * cfg.FFN_times
            + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times
            + cfg.Embedding_size + cfg.Embedding_size * 2
            + cfg.Embedding_size * 2
            )
        + cfg.Embedding_size * 2
        + cfg.Embedding_size * cfg.Vocabulary_size
    )
    KV_cache_init_usage = cfg.Activation_width * cfg.Layer_num * (cfg.Prefill_size * cfg.Embedding_size * 2)
    
    for i in range(cfg.Decode_size):
        if cfg.Embedding_flag:
            embedding1 = Stage("Embedding", platform="cloud")
            embedding1.read_data_amount = (1 + 1 * cfg.Embedding_size) * cfg.Activation_width
            embedding1.compute_amount = 1 * cfg.Embedding_size
            embedding1.write_data_amount = 1 * cfg.Embedding_size * cfg.Activation_width
            embedding1.memory_usage = (Weight_memory_usage
                                       + KV_cache_init_usage
                                       + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2
                                       + embedding1.write_data_amount)
            Stage_list.append(embedding1)
        
        for j in range(cfg.Layer_num):
            LayerRMSNorm1 = Stage("LayerRMSNorm1", platform="cloud")
            LayerRMSNorm1.read_data_amount = (1 * cfg.Embedding_size * cfg.Activation_width
                                              + cfg.Embedding_size * cfg.Weight_width)
            LayerRMSNorm1.compute_amount = 1 * cfg.Embedding_size * 2
            LayerRMSNorm1.write_data_amount = 1 * cfg.Embedding_size * cfg.Activation_width
            LayerRMSNorm1.memory_usage = (Weight_memory_usage
                                          + KV_cache_init_usage
                                          + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2
                                          + cfg.Activation_width * j * cfg.Embedding_size * 2
                                          + LayerRMSNorm1.write_data_amount)
            Stage_list.append(LayerRMSNorm1)
            
            QKV_MV = Stage("QKV_MV", platform="cloud")  # 这个地方写回就完成了KV concat
            QKV_MV.read_data_amount = (1 * cfg.Embedding_size * cfg.Activation_width + (cfg.Embedding_size * cfg.Embedding_size ) * cfg.Weight_width) * 3
            # QKV_MV.compute_amount = cfg.Prefill_size * cfg.Embedding_size * cfg.Embedding_size * 2 + cfg.Prefill_size * cfg.Embedding_size
            QKV_MV.compute_amount = 1 * cfg.Embedding_size * cfg.Embedding_size * 2 * 3   #没考虑bias是device_dict["cloud"].mfu里自带可以加bias模块，方便计算
            QKV_MV.write_data_amount = 1 * cfg.Embedding_size * cfg.Activation_width * 3
            QKV_MV.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + cfg.Embedding_size * cfg.Activation_width
            QKV_MV.onchip_memory_usage = 1 * cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Embedding_size * cfg.Weight_width + QKV_MV.write_data_amount
            Stage_list.append(QKV_MV)
            
            rope = Stage("rope_embedding_QK", platform="cloud")
            rope.read_data_amount = (2 * cfg.Attention_feature_size * 1 * cfg.Weight_width  # 编码读取
                                     + 2 * cfg.Attention_head * cfg.Attention_feature_size * 1 * cfg.Activation_width)
            rope.compute_amount = 2 * 2 * cfg.Attention_head * cfg.Attention_feature_size * 1
            rope.write_data_amount = 2 * cfg.Attention_head * cfg.Attention_feature_size * 1  # 写入memory
            rope.memory_usage = (Weight_memory_usage
                                 + KV_cache_init_usage
                                 + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2
                                 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2
                                 + cfg.Embedding_size * cfg.Activation_width)
            Stage_list.append(rope)
            
            QKV_Transpose1 = Stage("V_Transpose", platform="cloud")
            QKV_Transpose1.read_data_amount = (cfg.Prefill_size + i + 1) * cfg.Embedding_size * cfg.Activation_width
            QKV_Transpose1.compute_amount = 0
            QKV_Transpose1.write_data_amount = (cfg.Prefill_size + i + 1) * cfg.Embedding_size * cfg.Activation_width
            QKV_Transpose1.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + cfg.Embedding_size * cfg.Activation_width
            Stage_list.append(QKV_Transpose1)

            QK_MV = Stage("QK_MV", platform="cloud")
            QK_MV.read_data_amount = cfg.Attention_head * (cfg.Prefill_size + i + 2) * cfg.Attention_feature_size * cfg.Activation_width
            QK_MV.compute_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1) * cfg.Attention_feature_size * 2
            QK_MV.write_data_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1) * cfg.Activation_width
            QK_MV.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + QK_MV.write_data_amount
            QK_MV.onchip_memory_usage = QK_MV.read_data_amount + QK_MV.write_data_amount + 1 * cfg.Embedding_size * cfg.Activation_width
            Stage_list.append(QK_MV)

            Softmax = Stage("Softmax", platform="cloud")
            Softmax.read_data_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1) * cfg.Activation_width
            Softmax.compute_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1)
            Softmax.write_data_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1) * cfg.Activation_width
            Softmax.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + Softmax.write_data_amount
            Stage_list.append(Softmax)

            AttenV = Stage("AttenV", platform="cloud")
            AttenV.read_data_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1) * cfg.Activation_width + (cfg.Prefill_size + i + 1) * cfg.Embedding_size * cfg.Activation_width
            AttenV.compute_amount = cfg.Attention_head * (cfg.Prefill_size + i + 1) * cfg.Attention_feature_size * 2
            AttenV.write_data_amount = cfg.Attention_head * cfg.Attention_feature_size * cfg.Activation_width
            AttenV.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + AttenV.write_data_amount
            AttenV.onchip_memory_usage = AttenV.read_data_amount + AttenV.write_data_amount
            Stage_list.append(AttenV)

            Output_MV = Stage("Output_MV", platform="cloud")
            Output_MV.read_data_amount = cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Embedding_size * cfg.Weight_width
            Output_MV.compute_amount = cfg.Embedding_size * cfg.Embedding_size * 2
            Output_MV.write_data_amount = cfg.Embedding_size * cfg.Activation_width
            Output_MV.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + Output_MV.write_data_amount
            Output_MV.onchip_memory_usage = Output_MV.read_data_amount + Output_MV.write_data_amount
            Stage_list.append(Output_MV)

            Atten_Element_Add = Stage("Atten_Element_Add", platform="cloud")
            Atten_Element_Add.read_data_amount = cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Activation_width
            Atten_Element_Add.compute_amount = cfg.Embedding_size
            Atten_Element_Add.write_data_amount = cfg.Embedding_size * cfg.Activation_width
            Atten_Element_Add.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + Atten_Element_Add.write_data_amount
            Stage_list.append(Atten_Element_Add)
            
            FFN_RMSLayerNorm0 = Stage("FFN_RMSLayerNorm0", platform="cloud")
            FFN_RMSLayerNorm0.read_data_amount = (cfg.Embedding_size * cfg.Activation_width
                                                  + cfg.Embedding_size * cfg.Weight_width)
            FFN_RMSLayerNorm0.compute_amount = cfg.Embedding_size * 2
            FFN_RMSLayerNorm0.write_data_amount = cfg.Embedding_size * cfg.Activation_width
            FFN_RMSLayerNorm0.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + FFN_RMSLayerNorm0.write_data_amount
            Stage_list.append(FFN_RMSLayerNorm0)

            FFN_MV_up = Stage("FFN_MV_up", platform="cloud")
            FFN_MV_up.read_data_amount = (cfg.Embedding_size * cfg.Activation_width
                                          + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * cfg.Weight_width)
            FFN_MV_up.compute_amount = cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * 2
            FFN_MV_up.write_data_amount = cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
            FFN_MV_up.onchip_memory_usage = FFN_MV_up.read_data_amount + FFN_MV_up.write_data_amount
            FFN_MV_up.memory_usage = (Weight_memory_usage
                                      + KV_cache_init_usage
                                      + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2
                                      + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2
                                      + FFN_MV_up.write_data_amount)
            Stage_list.append(FFN_MV_up)

            FFN_MV_gate = Stage("FFN_MV_gate", platform="cloud")  # 包含了Silu
            FFN_MV_gate.read_data_amount = (cfg.Embedding_size * cfg.Activation_width
                                            + cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * cfg.Weight_width)
            FFN_MV_gate.compute_amount = 1 * cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * 2
            FFN_MV_gate.write_data_amount = 1 * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
            FFN_MV_gate.onchip_memory_usage = FFN_MV_gate.read_data_amount + FFN_MV_gate.write_data_amount
            FFN_MV_gate.memory_usage = (Weight_memory_usage
                                        + KV_cache_init_usage
                                        + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2
                                        + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2
                                        + FFN_MV_up.write_data_amount
                                        + FFN_MV_gate.write_data_amount)
            Stage_list.append(FFN_MV_gate)

            FFN_Element_Mul = Stage("FFN_Element_Mul", platform="cloud")
            FFN_Element_Mul.read_data_amount = 1 * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width * 2
            FFN_Element_Mul.compute_amount = 1 * cfg.Embedding_size * cfg.FFN_times * 2
            FFN_Element_Mul.write_data_amount = 1 * cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
            FFN_Element_Mul.memory_usage = (Weight_memory_usage
                                            + KV_cache_init_usage
                                            + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2
                                            + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2
                                            + FFN_Element_Mul.write_data_amount
                                            )
            Stage_list.append(FFN_Element_Mul)

            FFN_MV_down = Stage("FFN_MV_down", platform="cloud")
            FFN_MV_down.read_data_amount = (cfg.Embedding_size * cfg.FFN_times * cfg.Activation_width
                                            + cfg.Embedding_size * cfg.FFN_times * cfg.Embedding_size * cfg.Weight_width)
            FFN_MV_down.compute_amount = cfg.Embedding_size * cfg.Embedding_size * cfg.FFN_times * 2
            FFN_MV_down.write_data_amount = cfg.Embedding_size * cfg.Activation_width
            FFN_MV_down.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + FFN_MV_down.write_data_amount
            FFN_MV_down.onchip_memory_usage = FFN_MV_down.read_data_amount + FFN_MV_down.write_data_amount
            Stage_list.append(FFN_MV_down)

            FFN_Element_Add = Stage("FFN_Element_Add", platform="cloud")
            FFN_Element_Add.read_data_amount = cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Activation_width
            FFN_Element_Add.compute_amount = cfg.Embedding_size
            FFN_Element_Add.write_data_amount = cfg.Embedding_size * cfg.Activation_width
            FFN_Element_Add.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * i * cfg.Embedding_size * 2 + cfg.Activation_width * (j + 1) * cfg.Embedding_size * 2 + FFN_Element_Add.write_data_amount
            Stage_list.append(FFN_Element_Add)

        Last_RMSLayerNorm = Stage("Last_RMSLayerNorm", platform="cloud")
        Last_RMSLayerNorm.read_data_amount = (cfg.Embedding_size * cfg.Activation_width
                                              + cfg.Embedding_size * cfg.Weight_width)
        Last_RMSLayerNorm.compute_amount = cfg.Embedding_size * 2
        Last_RMSLayerNorm.write_data_amount = cfg.Embedding_size * cfg.Activation_width
        Last_RMSLayerNorm.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * (
                i + 1) * cfg.Embedding_size * 2 + Last_RMSLayerNorm.write_data_amount
        Stage_list.append(Last_RMSLayerNorm)

        Last_MM = Stage("Last_MM", platform="cloud")
        Last_MM.read_data_amount = cfg.Embedding_size * cfg.Activation_width + cfg.Embedding_size * cfg.Vocabulary_size * cfg.Weight_width
        Last_MM.compute_amount = cfg.Embedding_size * cfg.Vocabulary_size * 2
        Last_MM.write_data_amount = cfg.Vocabulary_size * cfg.Activation_width
        Last_MM.memory_usage = Weight_memory_usage + KV_cache_init_usage + cfg.Activation_width * cfg.Layer_num * (i + 1) * cfg.Embedding_size * 2 + Last_MM.write_data_amount
        Last_MM.onchip_memory_usage = Last_MM.read_data_amount + Last_MM.write_data_amount
        Stage_list.append(Last_MM)
    return Stage_list


def print_inference_info(cfg, prefill_stages, decode_stages):
    Prefill_memory_r_B = 0
    Prefill_memory_w_B = 0
    Prefill_computation_OPs = 0
    Decode_memory_r_B = 0
    Decode_memory_w_B = 0
    Decode_computation_OPs = 0


    for stage in prefill_stages:
        Prefill_memory_r_B += stage.read_data_amount
        Prefill_memory_w_B += stage.write_data_amount
        Prefill_computation_OPs += stage.compute_amount

    for stage in decode_stages:
        Decode_memory_r_B += stage.read_data_amount
        Decode_memory_w_B += stage.write_data_amount
        Decode_computation_OPs += stage.compute_amount

    Prefill_memory_access_B = Prefill_memory_r_B + Prefill_memory_w_B
    Decode_memory_access_B = Decode_memory_r_B + Decode_memory_w_B

    Total_memory_r_B = Prefill_memory_r_B + Decode_memory_r_B
    Total_memory_w_B = Prefill_memory_w_B + Decode_memory_w_B
    Total_memory_access_B = Total_memory_r_B + Total_memory_w_B
    Total_computation_OPs = Prefill_computation_OPs + Decode_computation_OPs

    print("=====================================================")
    print("Model_name:                           %s"        % (cfg.Model_name))
    print("Weight, Activation, KV cache:         %s"        % (cfg.WAKV_bitwidth))
    print("Prefill_size, Decode_size:           [%4d, %4d]" % (cfg.Prefill_size, cfg.Decode_size))
    print("-----------------------------------------------------")
    print("Prefill_computation:             %13.6f GOPs"    % (Prefill_computation_OPs / 1e9))
    print("Prefill_memory_access:           %13.6f GB"      % (Prefill_memory_access_B / 1024 / 1024 / 1024))
    print("--Prefill_memory_read:           %13.6f GB"      % (Prefill_memory_r_B / 1024 / 1024 / 1024))
    print("--Prefill_memory_write:          %13.6f GB"      % (Prefill_memory_w_B / 1024 / 1024 / 1024))
    print("Prefill_intensity:               %13.6f OP/Byte" % (Prefill_computation_OPs / Prefill_memory_access_B))
    print("-----------------------------------------------------")
    print("Decode_computation:              %13.6f GOPs"    % (Decode_computation_OPs / 1e9))
    print("Decode_memory_access:            %13.6f GB"      % (Decode_memory_access_B / 1024 / 1024 / 1024))
    print("--Decode_memory_read:            %13.6f GB"      % (Decode_memory_r_B / 1024 / 1024 / 1024))
    print("--Decode_memory_write:           %13.6f GB"      % (Decode_memory_w_B / 1024 / 1024 / 1024))
    print("Decode_intensity:                %13.6f OP/Byte" % (Decode_computation_OPs / Decode_memory_access_B))
    print("-----------------------------------------------------")
    print("Decode_computation_per_token:    %13.6f GOPs"    % (Decode_computation_OPs / cfg.Decode_size / 1e9))
    print("Decode_memory_access_per_token:  %13.6f GB"      % (Decode_memory_access_B / cfg.Decode_size / 1024 / 1024 / 1024))
    print("--Decode_memory_read_per_token:  %13.6f GB"      % (Decode_memory_r_B / cfg.Decode_size / 1024 / 1024 / 1024))
    print("--Decode_memory_write_per_token: %13.6f GB"      % (Decode_memory_w_B / cfg.Decode_size / 1024 / 1024 / 1024))
    print("Decode_intensity_per_token:      %13.6f OP/Byte" % (Decode_computation_OPs / Decode_memory_access_B))
    print("-----------------------------------------------------")
    print("Total_computation:               %13.6f GOPs"    % (Total_computation_OPs / 1e9))
    print("Total_memory_access:             %13.6f GB"      % (Total_memory_access_B / 1024 / 1024 / 1024))
    print("--Total_memory_read:             %13.6f GB"      % (Total_memory_r_B / 1024 / 1024 / 1024))
    print("--Total_memory_write:            %13.6f GB"      % (Total_memory_w_B / 1024 / 1024 / 1024))
    print("Total_intensity_per_token:       %13.6f OP/Byte" % (Total_computation_OPs / Total_memory_access_B))
    print("=====================================================")


def profile_model(Model_name, Hardware_name, Prefill_size, Decode_size):
    cfg = Config(
        Model_name = Model_name,
        Hardware_name = Hardware_name,
        Prefill_size = Prefill_size,
        Decode_size = Decode_size,
    )
    prefill_stages = Prefill(cfg)
    decode_stages = Decode(cfg)
    print_inference_info(cfg, prefill_stages, decode_stages)


if __name__ == "__main__":
    profile_model(
        Model_name = "LLaMA2-7B",
        Hardware_name = "GPU",
        Prefill_size = 1536,
        Decode_size = 512,
    )

