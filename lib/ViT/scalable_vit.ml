module IWSA = struct
  type t = {
    dim: int;
    num_heads: int;
    scale: float;
    qkv: Tensor.t;
    attn_drop: Tensor.t -> Tensor.t;
    proj: Tensor.t;
    proj_drop: Tensor.t -> Tensor.t;
    ws: int;
    limit_func: Tensor.t;
  }

  let create ~vs ~dim ~num_heads ~qkv_bias ~qk_scale ~attn_drop ~proj_drop ~ws =
    let head_dim = dim / num_heads in
    let scale = Option.value qk_scale ~default:(1. /. sqrt (float_of_int head_dim)) in
    let qkv = Vs.linear vs dim (dim * 3) ~bias:qkv_bias in
    let attn_drop = Vs.dropout ~p:attn_drop in
    let proj = Vs.linear vs dim dim in
    let proj_drop = Vs.dropout ~p:proj_drop in
    let limit_func = Vs.conv2d vs dim dim ~kernel_size:3 ~stride:1 ~padding:1 ~groups:dim in
    { dim; num_heads; scale; qkv; attn_drop; proj; proj_drop; ws; limit_func }
end