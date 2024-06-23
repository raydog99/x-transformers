open Torch

let exists v = Torch.(v <> none)

let attention
    q k v
    ?(mask : Tensor.t option)
    ?(causal=false)
    ?(attn_bias : Tensor.t option)
    ()
  =
  let scale = Tensor.pow_scalar (Tensor.shape q |> List.last_exn |> Tensor.to_float) (-0.5) in
  let q = Tensor.mul q scale in

  let sim = Layer.einsum "b h i d, b h j d -> b h i j" [q; k] in

  let sim =
    match attn_bias with
    | Some bias -> Tensor.add sim bias
    | None -> sim
  in

  let mask_value = Tensor.finfo sim |> Tensor.dtype |> Tensor.max_value in

  let sim =
    match mask with
    | Some m ->
      if Tensor.ndim m = 2 then
        Tensor.rearrange m "b j -> b 1 1 j"
      else m
      |> Tensor.masked_fill ~mask:(Tensor.logical_not m) ~value:mask_value
      |> Tensor.masked_fill ~mask:(Tensor.ones_like sim |> Tensor.triu (Tensor.shape sim |> List.last_exn - 1)) ~value:mask_value
    | None -> sim
  in

  let sim = Tensor.sub sim (Tensor.amax ~dim:[-1] ~keepdim:true sim |> Tensor.detach) in
  let attn = Tensor.softmax sim ~dim:(-1) in

  let out = Layer.einsum "b h i j, b h j d -> b h i d" [attn; v] in
  out

let summarize_qkv_chunk
    q k v mask attn_bias_chunk causal qk_start_indices dropout
  =
  let q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = qk_start_indices in
  let weight = Layer.einsum "b h i d, b h j d -> b h i j" [q; k] in

  let weight =
    match attn_bias_chunk with
    | Some bias -> Tensor.add weight bias
    | None -> weight
  in

  let mask_value = Tensor.finfo weight |> Tensor.dtype |> Tensor.max_value in

  let weight =
    match mask with
    | Some m ->
      Tensor.rearrange m "b j -> b 1 1 j"
      |> Tensor.masked_fill ~mask:(Tensor.logical_not m) ~value:mask_value
    | None -> weight
  in

  let weight =
    if causal && q_start_index < (k_start_index + k_chunk_size - 1) then
      let causal_mask = Tensor.ones [q_chunk_size; k_chunk_size] ~dtype:Bool |> Tensor.triu (q_start_index - k_start_index + 1) in
      Tensor.masked_fill weight ~mask:causal_mask ~value:mask_value
    else weight
  in

  let weight_max = Tensor.amax ~dim:[-1] ~keepdim:true weight |> Tensor.detach in
  let weight = Tensor.sub weight weight_max in

  let exp_weight = Tensor.exp weight in
  let exp_weight = Layer.dropout exp_weight ~p:dropout in

  let weighted_value = Layer.einsum "b h i j, b h j d -> b h i d" [exp_weight; v] in

  exp_weight |> Tensor.sum ~dim:[-1], weighted_value, Tensor.rearrange weight_max "... 1"

let checkpointed_summarize_qkv_chunk = Layer.partial Layer.checkpoint summarize_qkv_chunk