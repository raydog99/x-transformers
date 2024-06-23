open Base
open Torch

let prob_mask_like t prob =
  Torch.((torch_zeros_like t |> to_float |> uniform_ 0. 1.) < prob)

let mask_with_tokens t token_ids =
  let init_no_mask = Torch.full_like t false in
  List.fold_left
    (fun acc el -> Torch.(acc ||| (t == el)))
    init_no_mask
    token_ids

let get_mask_subset_with_prob mask prob =
  let batch, seq_len, device = Torch.shape mask.(0), Torch.shape mask.(1), Torch.device mask in
  let max_masked = Float.ceil (prob *. seq_len |> float_of_int |> Float.ceil) |> int_of_float in

  let num_tokens = Torch.sum mask ~dim:[-1] ~keepdim:true in
  let mask_excess = Torch.(cumsum mask ~dim:[-1] > (num_tokens * prob |> ceil)) in
  let mask_excess = Torch.(mask_excess.[; :max_masked]) in

  let rand = Torch.rand ~device:(Torch.device_of_tensor mask) [batch; seq_len] |> Torch.masked_fill (mask |> not_ |> Torch.logical_not) (-1e9) in
  let _, sampled_indices = Torch.topk rand max_masked ~dim:-1 in
  let sampled_indices = Torch.(sampled_indices + 1 |> masked_fill_ mask_excess 0) in

  let new_mask = Torch.zeros ~device:[Torch.device_of_tensor mask] [batch; seq_len + 1] in
  Torch.scatter_ new_mask ~dim:-1 ~index:sampled_indices ~src:(Torch.ones_like sampled_indices);
  Torch.bool (new_mask.[; 1:])

let mlm transformer ~mask_prob ~replace_prob ?(num_tokens = None) ~random_token_prob ~mask_token_id ~pad_token_id ~mask_ignore_token_ids =
  object
    method transformer = transformer
    method mask_prob = mask_prob
    method replace_prob = replace_prob
    method num_tokens = num_tokens
    method random_token_prob = random_token_prob
    method pad_token_id = pad_token_id
    method mask_token_id = mask_token_id
    method mask_ignore_token_ids = List.fold_left (fun acc el -> acc |> Set.add el) (Set.singleton pad_token_id) mask_ignore_token_ids
  end