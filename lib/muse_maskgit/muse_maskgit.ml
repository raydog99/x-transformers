open Torch

let exists (value : 'a option) : bool =
  match value with
  | Some _ -> true
  | None -> false

let default (value : 'a option) (default : 'a) : 'a =
  match value with
  | Some v -> v
  | None -> default

let eval_decorator fn =
  fun model ->
    let was_training = Model.is_training model in
    Model.eval model;
    let out = fn model in
    Model.train model was_training;
    out

let l2norm t =
  Tensor.normalize t ~dim:(-1)

let get_mask_subset_prob mask prob min_mask =
  let batch = Tensor.shape mask |> Array.of_list |> Array.get 0 in
  let seq = Tensor.shape mask |> Array.of_list |> Array.get 1 in
  let device = Tensor.device mask in

  let num_to_mask = Tensor.mul (Tensor.sum ~dim:(-1) mask) prob |> Tensor.clamp_min min_mask in
  let logits = Tensor.rand [|(batch, seq)|] ~device in
  let logits = Tensor.masked_fill logits (Tensor.logical_not mask) (-1.) in

  let randperm = Tensor.argsort logits ~dim:(-1) |> Tensor.argsort ~dim:(-1) |> Tensor.float_of_tensor in

  let num_padding = Tensor.sum ~dim:(-1) (Tensor.logical_not mask) in
  let randperm = Tensor.sub randperm (Tensor.unsqueeze num_padding ~dim:1) in

  let subset_mask = Tensor.lt randperm (Tensor.unsqueeze num_to_mask ~dim:1) in
  Tensor.masked_fill_ subset_mask (Tensor.logical_not mask) false;
  subset_mask