open Torch

let ds_conv2d dim_in dim_out kernel_size padding ?(stride=1) ?(bias=true) =
  nn_sequential [
    nn_conv2d ~input_dim:dim_in ~output_dim:dim_in ~kernel_size ~padding ~groups:dim_in ~stride ~bias ();
    nn_conv2d ~input_dim:dim_in ~output_dim:dim_out ~kernel_size:(1, 1) ~bias ()
  ]

type layer_norm =
  { g : Tensor.t
  ; b : Tensor.t
  ; eps : float
  }

let layer_norm dim eps =
  { g = nn_parameter (Tensor.ones [1; dim; 1; 1])
  ; b = nn_parameter (Tensor.zeros [1; dim; 1; 1])
  ; eps
  }

let layer_norm_forward x { g; b; eps } =
  let std = Tensor.var x ~dim:1 ~unbiased:false ~keepdim:true |> Tensor.sqrt in
  let mean = Tensor.mean x ~dim:1 ~keepdim:true in
  let normalized = Tensor.((x - mean) / (std + eps) * g + b) in
  normalized

let pre_norm dim fn =
  let norm = layer_norm dim 1e-5 in
  fun x -> fn (layer_norm_forward x norm)

type efficient_self_attention =
  { scale : float
  ; heads : int
  ; to_q : Torch.nn.t
  ; to_kv : Torch.nn.t
  ; to_out : Torch.nn.t
  }

let efficient_self_attention ~dim ~heads ~reduction_ratio =
  let scale = (float_of_int (dim / heads)) ** (-0.5) in
  { scale
  ; heads
  ; to_q = nn_conv2d ~input_dim:dim ~output_dim:dim ~kernel_size:(1, 1) ~bias:false ()
  ; to_kv = nn_conv2d ~input_dim:dim ~output_dim:(dim * 2) ~kernel_size:reduction_ratio ~stride:reduction_ratio ~bias:false ()
  ; to_out = nn_conv2d ~input_dim:dim ~output_dim:dim ~kernel_size:(1, 1) ~bias:false ()
  }

let efficient_self_attention_forward x { scale; heads; to_q; to_kv; to_out } =
  let h, w = Tensor.shape2d x in
  let q, k, v = (to_q x, nn_forward to_kv x |> Tensor.chunk ~chunks:2 ~dim:1) in
  let q, k, v = Tensor.(q, k, v.(0), v.(1)) |> List.map (fun t -> rearrange t ~dims:"b (h c) x y -> (b h) (x y) c" ~h:heads) in
  let sim = einsum "b i d, b j d -> b i j" q k |> Tensor.( * ) scale in
  let attn = Tensor.softmax sim ~dim:(-1) in
  let out = einsum "b i j, b j d -> b i d" attn v |> rearrange ~dims:"(b h) (x y) c -> b (h c) x y" ~h:heads ~x:h ~y:w in
  nn_forward to_out out