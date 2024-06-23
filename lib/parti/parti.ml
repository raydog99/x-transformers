open Base
open Torch

let log t ~eps:eps_float =
  let eps = Tensor.of_float eps_float ~device:(Device.Cpu 0) in
  Tensor.log (Tensor.add t eps)

let gumbel_noise t =
  let noise = Tensor.zeros_like t |> Tensor.uniform_ ~low:0. ~high:1. in
  Tensor.neg (log (Tensor.neg (log noise)))

let gumbel_sample t ~temperature ~dim =
  let noise = gumbel_noise t in
  let scaled_t = Tensor.div t (Tensor.of_float temperature ~device:(Device.Cpu 0)) in
  Tensor.add scaled_t noise |> Tensor.argmax ~dim

let top_k logits ~thres =
  let num_logits = Tensor.shape logits |> Array.last_exn in
  let k = max (Float.to_int ((1. -. thres) *. float num_logits)) 1 in
  let (vals, inds) = Tensor.topk logits ~k ~dim:(-1) ~largest:true ~sorted:true in
  let probs = Tensor.full_like logits Float.neg_infinity in
  Tensor.scatter probs ~index:inds ~src:vals
  |> fun _ -> probs

let prob_mask_like shape prob device =
  match Float.(prob = 1.) with
  | true -> Tensor.ones shape ~device ~dtype:Bool
  | false ->
    let uniform = Tensor.uniform shape ~low:0. ~high:1. ~device in
    Tensor.(uniform < Tensor.of_float prob ~device)

type layer_norm = { gamma : Tensor.t; beta : Tensor.t }

let layer_norm dim =
  let gamma = Tensor.ones [dim] ~device:(Device.Cpu 0) |> Tensor.to_requires_grad ~requires_grad:true in
  let beta = Tensor.zeros [dim] ~device:(Device.Cpu 0) in
  { gamma; beta }

let layer_norm_forward x { gamma; beta } =
  Tensor.layer_norm x ~normalized_shape:(List.init (Tensor.ndim x) ~f:(fun _ -> dim)) ~weight:gamma ~bias:beta ~eps:1e-5