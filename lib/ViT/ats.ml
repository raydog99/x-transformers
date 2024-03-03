open Torch

module AdaptiveTokenSampling = struct
  type t = {
    eps : float;
    output_num_tokens : int;
  }

  let create output_num_tokens ~eps =
    { eps; output_num_tokens }

  let forward { eps; output_num_tokens } attn value mask =
    let heads = Tensor.size attn |> List.tl |> List.hd_exn in
    let device = Tensor.device attn in
    let dtype = Tensor.kind attn in

    let cls_attn = Tensor.narrow attn ~dim:3 ~start:1 ~length:(Tensor.size attn).(3) - 1 in
    let value_norms = Tensor.norm value ~dim:[|-1|] ~keepdim:false in
    let cls_attn = Tensor.einsum "bhn,bhn->bn" cls_attn value_norms in
    let normed_cls_attn = Tensor.div cls_attn (Tensor.add (Tensor.sum cls_attn ~dim:[|-1|] ~keepdim:true) eps) in
    let pseudo_logits = Tensor.log normed_cls_attn in
    let mask_without_cls = Tensor.narrow mask ~dim:1 ~start:1 ~length:(Tensor.size mask).(1) - 1 in
    let mask_value = Tensor.finfo attn |> Tensor.min_value in
    let pseudo_logits = Tensor.masked_fill pseudo_logits (Tensor.logical_not mask_without_cls) mask_value in
    let pseudo_logits = Tensor.repeat pseudo_logits [|output_num_tokens|] ~dim:1 in
    let pseudo_logits = Tensor.add pseudo_logits (Tensor.sample_gumbel ~shape:(Tensor.shape pseudo_logits) ~device ~dtype) in
    let sampled_token_ids = Tensor.argmax ~dim:1 pseudo_logits ~keepdim:false + 1 in
    let unique_sampled_token_ids_list = List.map (Tensor.unique sampled_token_ids ~sorted:true) ~f:(fun t -> t) in
    let unique_sampled_token_ids = Tensor.pad_sequence unique_sampled_token_ids_list ~batch_first:true in
    let new_mask = Tensor.neq unique_sampled_token_ids 0 in
    let new_mask = Tensor.pad new_mask ~padding:(1, 0) ~value:true in
    let unique_sampled_token_ids = Tensor.pad unique_sampled_token_ids ~padding:(1, 0) ~value:0 in
    let expanded_unique_sampled_token_ids = Tensor.repeat unique_sampled_token_ids [|heads|] ~dim:1 in
    let new_attn = Tensor.batched_index_select attn expanded_unique_sampled_token_ids ~dim:2 in
    new_attn, new_mask, unique_sampled_token_ids
end