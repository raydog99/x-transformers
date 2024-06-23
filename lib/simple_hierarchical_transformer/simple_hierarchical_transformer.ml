open Base
open Torch

let l2norm t =
  let normalized = Tensor.normalize t ~dim:(-1) ~p:2. in
  normalized

let cosine_sim_loss x y =
  let x_norm = l2norm x in
  let y_norm = l2norm y in
  let similarity = Tensor.einsum "bnd,bnd->bn" [|x_norm; y_norm|] in
  Tensor.neg similarity |> Tensor.mean |> Tensor.neg |> Tensor.add_scalar 1.

let log t ~eps =
  Tensor.clamp_min t eps |> Tensor.log

let gumbel_noise t =
  let noise = Tensor.empty_like t |> Tensor.uniform_ ~a:0. ~b:1. in
  Tensor.neg (log (Tensor.neg (log noise)) ~eps:1e-20)

let gumbel_sample t ~temperature ~dim =
  let max_temperature = max temperature 1e-10 in
  let noise = gumbel_noise t in
  let scaled_logits = Tensor.div t max_temperature |> Tensor.add noise in
  Tensor.argmax ~dim scaled_logits ~keepdim:false

let top_k logits ~thres =
  let num_logits = Tensor.size logits.(Array.length logits - 1) in
  let k = int_of_float ((1. -. thres) *. float_of_int num_logits) in
  let (values, indices) = Tensor.topk logits k ~dim:1 ~largest:true ~sorted:true in
  let probs = Tensor.full_like logits (Tensor.finfo logits |> Tensor.finfo_max |> Tensor.neg) in
  Tensor.scatter_ probs ~dim:1 ~index:indices ~src:values;
  probs

module RotaryEmbedding = struct
  let create ~dim ~scale_base ~use_xpos =
    let inv_freq_values =
      Tensor.arange_float 0. (float_of_int dim) 2.
      |> Tensor.pow 10000.
      |> Tensor.pow (-1.)
    in
    let inv_freq = Var_store.new_var inv_freq_values in
    let scale_values =
      Tensor.arange_float 0. (float_of_int dim) 2.
      |> Tensor.add 0.4
      |> Tensor.div (1.4 *. float_of_int dim)
    in
    let scale = Var_store.new_var scale_values in

    let device = inv_freq#device in

    let forward seq_len =
      let t = Tensor.arange ~device (Torch_core.IArray [|seq_len|]) ~type_:(Torch_core.IFloat32) in
      let freqs = Tensor.einsum "i,j->ij" t inv_freq |> Tensor.cat ~dim:(-1) in

      if not use_xpos then
        (freqs, Tensor.ones ~device (Torch_core.IArray [|1|]))
      else
        let power = Tensor.sub t (Tensor.floordiv seq_len 2) |> Tensor.to_float |> Tensor.div scale_base in
        let scale = Tensor.pow scale power |> Tensor.cat ~dim:(-1) in
        (freqs, scale)
    in

    object
      method device = device
      method forward = forward
    end
end