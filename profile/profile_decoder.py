from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import distributed_utils as dist_utils, utils
from fairseq.modules import gelu, LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.moe import Top1Gate, Top2Gate, TopKGate, MOELayer
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fused_bias_gelu import fused_bias_gelu, has_fused_bias_gelu
from fairseq.modules.transformer_layer import make_experts
from torch import Tensor
from timers import Timers

timers = Timers()
def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)

def _ffn(
    x,
    fc1,
    activation_fn,
    activation_dropout_module,
    fc2,
    dropout_module,
):
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    if has_fused_bias_gelu and activation_fn == gelu:
        x = _linear(x, fc1.weight)
        x = fused_bias_gelu(x, fc1.bias)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    else:
        x = _linear(x, fc1.weight, fc1.bias)
        x = activation_fn(x)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    x = x.view(x_shape)
    x = dropout_module(x)
    return x


class ModelConfig:
    def __init__(self):
        self.decoder_embed_dim = 1024
        self.dropout = 0.1
        self.attention_dropout = 0.0
        self.activation_dropout = 0.0
        self.relu_dropout = 0.0
        self.decoder_embed_dim = 1024 
        self.decoder_output_dim = 1024
        self.decoder_input_dim = 1024 
        self.decoder_ffn_embed_dim = 4096 
        self.decoder_layers = 24
        self.decoder_attention_heads = 16
        self.decoder_normalize_before = True
        self.no_decoder_final_norm = False
        self.adaptive_softmax_cutoff = None
        self.adaptive_softmax_dropout = 0.0
        self.adaptive_softmax_factor = 4.0 
        self.no_token_positional_embeddings = False
        self.share_decoder_input_output_embed = True
        self.quant_noise_pq = 0.0
        self.moe_freq = 2
        self.moe_expert_count = 4
        self.moe_gating_use_fp32 = True
        self.moe_second_expert_policy = 'sampling'
        self.moe_normalize_gate_prob_before_dropping = False
        self.moe_top1_expert = True
        self.moe_train_capacity_token_fraction = 4.0
        self.moe_eval_capacity_token_fraction = 1.0
        self.moe_normalize_expert_grad = 'sqrt_world_size'
        self.activation_fn = 'gelu'

class DecoderLayerProfiler(nn.Module):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, is_moe_layer=False,
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.is_moe_layer = is_moe_layer

        ffn_dim = args.decoder_ffn_embed_dim
        if self.is_moe_layer and getattr(args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            ffn_dim = getattr(args, "alternate_decoder_ffn_embed_dim", 0.0)

        if not self.is_moe_layer or getattr(args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            self.activation_fn = utils.get_activation_fn(
                activation=str(args.activation_fn)
                if getattr(args, "activation_fn", None) is not None
                else "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    capacity_factor=getattr(args, "moe_train_capacity_token_fraction", 1.0),
                    moe_eval_capacity_token_fraction=getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                )
            elif args.moe_topk_expert:
                gate = TopKGate(
                    self.embed_dim,
                    num_experts=args.moe_expert_count,
                    topk=args.topk,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                )                
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                    getattr(args, "moe_batch_prioritized_routing", False),
                )
            experts = make_experts(args, self.embed_dim, ffn_dim, self.dropout_module)
            self.moe_layer = MOELayer(gate, experts, args)


        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        self.args = args

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        # Need input 'x'(seq_len, batch, embed_dim) and 'self_attn_mask'(seq_len, seq_len)
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)    # 1
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x                               # 2

        x, attn = self.self_attn(               # 3
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)              # 4
        x = self.residual_connection(x, residual)  # 5
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x     #6
        if self.normalize_before: 
            x = self.final_layer_norm(x)   #7
        if not self.is_moe_layer or getattr(self.args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:   #8
            x = _ffn(
                x,
                fc1=self.fc1,
                activation_fn=self.activation_fn,
                activation_dropout_module=self.activation_dropout_module,
                fc2=self.fc2,
                dropout_module=self.dropout_module,
            )
            l_aux = None
        else:                                               # 9
            # x - seq_len, batch_size, model_dim
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(self.args, "use_moe_pad_mask", False):
                x, l_aux = self.moe_layer(x, input_padding_mask=self_attn_padding_mask)
            else:
                x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1) # seq_len, batch_size, model_dim
        x = self.residual_connection(x, residual)          #10
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, l_aux
    
    def profile_layer_norm(self, seq_len, batch_size, cfg):
        x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim).cuda()
        grads_y = torch.rand(x.size()).cuda()
        # For warmup
        for i in range(10):
            y = self.self_attn_layer_norm(x)
            y.backward(grads_y)
        # For measuring
        for i in range(100):
            timers("LayerNorm Forward").start()
            y = self.self_attn_layer_norm(x)
            timers("LayerNorm Forward").stop()
            
            timers("LayerNorm Backward").start()
            y.backward(grads_y)
            timers("LayerNorm Backward").stop()
        timers.log(["LayerNorm Forward", "LayerNorm Backward"], normalizer=100)
    
    def profile_self_attn(self, seq_len, batch_size, cfg): 
        attn_mask = torch.zeros(seq_len, seq_len)
        attn_mask[:] = -float('inf')
        attn_mask = torch.triu(attn_mask, diagonal=1).cuda()
        x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim).cuda()
        grads_y = torch.rand(x.size()).cuda()
        # For warmup
        for i in range(10):
            y, attn = self.self_attn(               # 3
                query=x,
                key=x,
                value=x,
                key_padding_mask=None,
                incremental_state=None,
                need_weights=False,
                attn_mask=attn_mask,
            )
            y.backward(grads_y)
        
        # For measuring
        for i in range(100):
            timers("Self-Attention Forward").start()
            y, attn = self.self_attn(               # 3
                query=x,
                key=x,
                value=x,
                key_padding_mask=None,
                incremental_state=None,
                need_weights=False,
                attn_mask=attn_mask,
            )
            timers("Self-Attention Forward").stop()
            
            timers("Self-Attention Backward").start()
            y.backward(grads_y)
            timers("Self-Attention Backward").stop()

        timers.log(["Self-Attention Forward", "Self-Attention Backward"], normalizer=100)

    def profile_dropout(self, seq_len, batch_size, cfg): 
        # For warmup
        x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        grads_y = torch.rand(x.size()).cuda()
        for i in range(10):
            y = self.dropout_module(x)
            y.backward(grads_y)

        # For measuring
        # x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        # grads_y = torch.rand(y.size()).cuda()
        for i in range(100):
            timers("Dropout Forward").start()
            y = self.dropout_module(x)
            timers("Dropout Forward").stop()

            timers("Dropout Backward").start()
            y.backward(grads_y)
            timers("Dropout Backward").stop()
        timers.log(["Dropout Forward", "Dropout Backward"], normalizer=100)

    def profile_residual_connection(self, seq_len, batch_size, cfg): 
        x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        x2 = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        grads_y = torch.rand(x.size()).cuda()
        # For warmup
        for i in range(10):
            y = self.residual_connection(x, x2)
            y.backward(grads_y)
        # For measuring
        for i in range(100):
            timers("Residual_Connection Forward").start()
            y = self.residual_connection(x, x2)
            timers("Residual_Connection Forward").stop()

            timers("Residual_Connection Backward").start()
            y.backward(grads_y)
            timers("Residual_Connection Backward").stop()
        timers.log(["Residual_Connection Forward", "Residual_Connection Backward"], normalizer=100)

    def profile_ffn(self, seq_len, batch_size, cfg): 
        x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        grads_y = torch.rand(x.size()).cuda()
        
        for i in range(10):
            y = _ffn(
                x,
                fc1=self.fc1,
                activation_fn=self.activation_fn,
                activation_dropout_module=self.activation_dropout_module,
                fc2=self.fc2,
                dropout_module=self.dropout_module,
            )
            y.backward(grads_y)

        for i in range(100):
            timers("FFN Forward").start()
            y = _ffn(
                x,
                fc1=self.fc1,
                activation_fn=self.activation_fn,
                activation_dropout_module=self.activation_dropout_module,
                fc2=self.fc2,
                dropout_module=self.dropout_module,
            )
            timers("FFN Forward").stop()

            timers("FFN Backward").start()
            y.backward(grads_y)
            timers("FFN Backward").stop()
        timers.log(["FFN Forward", "FFN Backward"], normalizer=100)

    def profile_moe(self, seq_len, batch_size, cfg): 
        x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        grads_y = torch.rand(x.size()).cuda()

        for i in range(10):
            y = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(cfg, "use_moe_pad_mask", False):
                y, l_aux = self.moe_layer(y, input_padding_mask=None)
            else:
                y, l_aux = self.moe_layer(y)
            y = y.transpose(0, 1) # seq_len, batch_size, model_dim
            y.backward(grads_y)

        for i in range(100):
            timers("MoE Forward").start()
            y = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(cfg, "use_moe_pad_mask", False):
                y, l_aux = self.moe_layer(y, input_padding_mask=None)
            else:
                y, l_aux = self.moe_layer(y)
            y = y.transpose(0, 1) # seq_len, batch_size, model_dim
            timers("MoE Forward").stop()

            timers("MoE Backward").start()
            y.backward(grads_y)
            timers("MoE Backward").stop()
        timers.log(["MoE Forward", "MoE Backward"], normalizer=100)

    def profile_gate(self, seq_len, batch_size, cfg, gate_type):
        x = torch.rand(seq_len*batch_size, cfg.decoder_embed_dim, requires_grad=True).cuda()
        grads_y = None
        if gate_type == "Top1":
            gate = Top1Gate(
                cfg.decoder_embed_dim,
                cfg.moe_expert_count,
                use_fp32=cfg.moe_gating_use_fp32,
                capacity_factor=cfg.moe_train_capacity_token_fraction,
                moe_eval_capacity_token_fraction=cfg.moe_eval_capacity_token_fraction,
            )
        elif gate_type == "TopK":
            gate = TopKGate(
                cfg.decoder_embed_dim,
                num_experts=cfg.moe_expert_count,
                topk=-1,
                use_fp32=cfg.moe_gating_use_fp32,
                moe_eval_capacity_token_fraction=cfg.moe_eval_capacity_token_fraction,
            )                
        else:
            gate = Top2Gate(
                self.embed_dim,
                cfg.moe_expert_count,
                cfg.moe_gating_use_fp32,
                cfg.moe_second_expert_policy,
                cfg.moe_normalize_gate_prob_before_dropping,
                cfg.moe_eval_capacity_token_fraction,
                False,
            )
        gate = gate.cuda()
        for i in range(10):
            _, y, _, _ = gate(x)
            if grads_y == None:
                grads_y = torch.rand(y.size()).cuda()
            y.backward(grads_y)

        for i in range(100):
            timers(gate_type + "Gate Forward").start()
            _, y, _, _ = gate(x)
            timers(gate_type + "Gate Forward").stop()

            timers(gate_type + "Gate Backward").start()
            y.backward(grads_y)
            timers(gate_type + "Gate Backward").stop()
        timers.log([gate_type + "Gate Forward", gate_type + "Gate Backward"], normalizer=100)

if __name__ == "__main__":
    seq_len = 1024
    batch_size = 4
    cfg = ModelConfig()
    # profiler = DecoderLayerProfiler(cfg, is_moe_layer=cfg.moe_freq > 0).cuda()
    profiler = DecoderLayerProfiler(cfg).cuda()
    
    # Simulate the models' input
    input_x = torch.rand(seq_len, batch_size, cfg.decoder_embed_dim).cuda()
    attn_mask = torch.zeros(seq_len, seq_len)
    attn_mask[:] = -float('inf')
    attn_mask = torch.triu(attn_mask, diagonal=1).cuda()

    # x, _, _, _ = profiler.forward(input_x, self_attn_mask = attn_mask)
    profiler.profile_layer_norm(seq_len, batch_size, cfg)
    profiler.profile_self_attn(seq_len, batch_size, cfg)
    profiler.profile_dropout(seq_len, batch_size, cfg)
    profiler.profile_residual_connection(seq_len, batch_size, cfg)
    profiler.profile_ffn(seq_len, batch_size, cfg)
    # profiler.profile_moe(seq_len, batch_size, cfg)
    profiler.profile_gate(seq_len, batch_size, cfg, "Top1")
    profiler.profile_gate(seq_len, batch_size, cfg, "Top2")
    profiler.profile_gate(seq_len, batch_size, cfg, "TopK")
