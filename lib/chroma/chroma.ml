open Torch

let residual fn =
  fun x ->
    let fx = fn x in
    Tensor.add_ fx x;
    fx

let upsample dim dim_out =
  let dim_out = default dim_out dim in
  nn_sequential [
    nn_upsample ~scale_factor:2 ~mode:`Nearest;
    nn_conv2d ~input_dim:dim ~output_dim:dim_out ~kernel_size:(3, 3) ~padding:(1, 1) ()
  ]

let downsample dim dim_out =
  let dim_out = default dim_out dim in
  nn_conv2d ~input_dim:dim ~output_dim:dim_out ~kernel_size:(4, 4) ~stride:(2, 2) ~padding:(1, 1) ()

type layer_norm =
  { g : Tensor.t
  }

let layer_norm dim =
  { g = Tensor.ones [1; dim; 1; 1] }

let layer_norm_forward x { g } =
  let eps = if Tensor.dtype x = Torch.float32 then 1e-5 else 1e-3 in
  let var = Tensor.var x ~dim:1 ~unbiased:false ~keepdim:true in
  let mean = Tensor.mean x ~dim:1 ~keepdim:true in
  let normalized = Tensor.((x - mean) * ((var + eps) ** 0.5) * g) in
  normalized

let pre_norm dim fn =
  let norm = layer_norm dim in
  fun x ->
    let x = layer_norm_forward x norm in
    fn x

type learned_sinusoidal_pos_emb =
  { weights : Tensor.t
  }

let learned_sinusoidal_pos_emb dim =
  assert (dim mod 2 = 0);
  let half_dim = dim / 2 in
  { weights = Tensor.randn [half_dim] }

let learned_sinusoidal_pos_emb_forward x { weights } =
  let freqs = Tensor.(x * rearrange weights "d -> 1 d" * 2.0 * Float.pi) in
  let sin_part = Tensor.sin freqs in
  let cos_part = Tensor.cos freqs in
  let fouriered = Tensor.cat [x; sin_part; cos_part] ~dim:(-1) in
  fouriered