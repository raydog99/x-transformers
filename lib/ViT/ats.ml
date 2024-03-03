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
    let pseudo_logits = Tensor.masked_fill pseudo_logits (Tensor.logical_not mask_without_cls) mask_value
end