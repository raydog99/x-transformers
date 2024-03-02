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
    lim_func: Tensor.t;
  }

  let create ~vs ~dim ~num_heads ~qkv_bias ~qk_scale ~attn_drop ~proj_drop ~ws =
    let head_dim = dim / num_heads in
    let scale = Option.value qk_scale ~default:(1. /. sqrt (float_of_int head_dim)) in
    let qkv = Vs.linear vs dim (dim * 3) ~bias:qkv_bias in
    let attn_drop = Vs.dropout ~p:attn_drop in
    let proj = Vs.linear vs dim dim in
    let proj_drop = Vs.dropout ~p:proj_drop in
    let lim_func = Vs.conv2d vs dim dim ~kernel_size:3 ~stride:1 ~padding:1 ~groups:dim in
    { dim; num_heads; scale; qkv; attn_drop; proj; proj_drop; ws; lim_func }

  let lim t func =
    let y_lim = Vs.apply func [|t|] in
    Tensor.view y_lim ~size:[|Batch, H * W, dim|] |> Tensor.transpose1d ~dim1:1 ~dim2:2
  ;;

  let img2win x H W =
    let B, N, C = Tensor.shape x in
    let h_group, w_group = H / ws, W / ws in
    let x = Tensor.reshape4d x ~size:[|B; h_group; ws; w_group; ws; C|] |> Tensor.transpose ~dim1:2 ~dim2:3 in
    let x = Tensor.reshape5d x ~size:[|B; h_group * w_group; num_heads; ws * ws; C / num_heads|] in
    Tensor.permute x ~dims:[|0; 1; 3; 2; 4|]
  ;;

  let win2img x H W =
    let h_group, w_group = H / ws, W / ws in
    let B = Tensor.shape x.(0) in
    let C = Tensor.shape x.(4) * num_heads in
    let x = Tensor.permute x ~dims:[|0; 1; 3; 2; 4|] in
    let x = Tensor.reshape4d x ~size:[|B; h_group; w_group; ws; ws; num_heads * head_dim|] in
    Tensor.transpose x ~dim1:2 ~dim2:3 |> Tensor.reshape4d ~size:[|B; C; H; W|] |> Tensor.permute ~dims:[|0; 3; 1; 2|]
  ;;
end


module SSA = struct
  type t = {
    dim: int;
    num_heads: int;
    scale: float;
    q: Tensor.t;
    reduction: Tensor.t;
    norm_act: Tensor.t;
    k: Tensor.t;
    v: Tensor.t;
    proj: Tensor.t;
    attn_drop: Tensor.t -> Tensor.t;
    proj_drop: Tensor.t -> Tensor.t;
  }

  let create ~vs ~dim ~num_heads ~qkv_bias ~qk_scale ~attn_drop ~proj_drop ~sr_ratio ~c_ratio =
    let head_dim = dim / num_heads in
    let scale = Option.value qk_scale ~default:(head_dim * c_ratio) ** (-0.5) in
    let c_new = int_of_float (float_of_int dim * c_ratio) in
    let q = Vs.linear vs dim c_new ~bias:qkv_bias in
    let k = Vs.linear vs c_new c_new ~bias:qkv_bias in
    let v = Vs.linear vs c_new dim ~bias:qkv_bias in
    let proj = Vs.linear vs dim dim in
    let attn_drop = Vs.dropout ~p:attn_drop in
    let proj_drop = Vs.dropout ~p:proj_drop in
    let reduction =
      if sr_ratio > 1.0 then
        let reduction_conv1 = Vs.conv2d vs dim dim ~kernel_size:sr_ratio ~stride:sr_ratio ~groups:dim in
        let reduction_conv2 = Vs.conv2d vs dim c_new ~kernel_size:1 ~stride:1 in
        let norm_act = Vs.apply Layer_norm [|reduction_conv1|] in
        fun x -> Vs.apply reduction_conv2 [|Vs.apply norm_act [|x|]|]
      else
        fun x -> x
    in
    { dim; num_heads; scale; q; reduction; norm_act; k; v; proj; attn_drop; proj_drop }
  ;;
end