open Torch

let layer_norm dim =
  let gamma = nn_parameter (Torch.ones [dim]) in
  let beta = nn_buffer (Torch.zeros [dim]) in
  fun x -> F.layer_norm x ~normalized_shape:[|x.shape.(Array.length x.shape - 1)|] ~weight:gamma ~bias:beta

let residual fn x =
  let res = fn x in
  Torch.add res x

let embed_to_latents dim dim_latents =
  let to_latents = nn_linear ~bias:false dim dim_latents in
  fun x ->
    let latents = Torch.forward to_latents x in
    F.normalize latents ~dim:(-1)

let rotary_embedding dim =
  let inv_freq = Tensor.pow (Tensor.of_float 10000.) (Tensor.arange ~start:0 ~end_:dim ~step:2 ~dtype:Float32 / float dim) in
  fun max_seq_len ~device ->
    let seq = Torch.arange ~device ~dtype:inv_freq#dtype (Tensor.of_int max_seq_len) in
    let freqs = Tensor.einsum "i,j->ij" seq inv_freq in
    Tensor.cat ~dim:(-1) [|freqs; freqs|]