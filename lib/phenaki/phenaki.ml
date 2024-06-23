open Torch

let get_mask_subset_with_prob mask prob =
  let batch = Tensor.shape mask |> Array.of_list |> Array.get 0 in
  let seq_len = Tensor.shape mask |> Array.of_list |> Array.get 1 in
  let device = Tensor.device mask in

  let num_tokens = Tensor.sum ~dim:(-1) mask in
  let num_pads = Tensor.sub seq_len num_tokens in
  let num_masked = Tensor.round (Tensor.mul prob num_tokens) |> Tensor.clamp_min 1. in

  let randperm_indices = Tensor.rand [|(batch, seq_len)|] ~device |> Tensor.argsort ~dim:(-1) in
  let randperm_indices = Tensor.sub randperm_indices (Tensor.unsqueeze num_pads ~dim:1) in
  let randperm_indices = Tensor.masked_fill randperm_indices (Tensor.lt randperm_indices (Tensor.zeros_like randperm_indices) |> Tensor.logical_not ()) seq_len in

  let mask_subset = Tensor.lt randperm_indices (Tensor.unsqueeze num_masked ~dim:1) in
  mask_subset

let eval_decorator fn =
  fun model ->
    let was_training = Model.is_training model in
    Model.eval model;
    let out = fn model in
    Model.train model was_training;
    out

let uniform shape device =
  Tensor.zeros shape ~device |> Tensor.float |> Tensor.uniform_ ~a:0. ~b:1.

let prob_mask_like shape prob device =
  match prob with
  | 1. -> Tensor.ones shape ~device ~dtype:`Bool
  | 0. -> Tensor.zeros shape ~device ~dtype:`Bool
  | _ -> Tensor.zeros shape ~device |> Tensor.float |> Tensor.uniform_ ~a:0. ~b:1. |> Tensor.lt prob

let log t eps =
  Tensor.log (Tensor.add t eps)

let gumbel_noise t =
  let noise = Tensor.zeros_like t |> Tensor.uniform_ ~a:0. ~b:1. in
  Tensor.neg (log (Tensor.neg noise))

let gumbel_sample t ?(temperature=1.) ~dim =
  let gumbel = gumbel_noise t in
  let gumbel_scaled = Tensor.div t (max temperature 1e-10) |> Tensor.add gumbel in
  Tensor.argmax gumbel_scaled ~dim

let top_k logits ?(thres=0.5) =
  let num_logits = Tensor.shape logits |> Array.of_list |> Array.get (-1) in
  let k = max (int_of_float ((1. -. thres) *. float_of_int num_logits)) 1 in
  let (values, indices) = Tensor.topk logits ~k ~dim:(-1) in
  let probs = Tensor.full_like logits (float_of_int min_int) in
  Tensor.scatter probs ~dim:1 ~index:indices ~src:values;
  probs