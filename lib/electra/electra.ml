open Torch

let exists val_ = not (Tensor.equal val_ (Tensor.of_float 0.))

let default val_ d = if exists val_ then val_ else d

let gumbel_noise t eps =
  let u = Tensor.rand_like t in
  Tensor.neg (Tensor.log (Tensor.neg (Tensor.log (Tensor.add u eps)) +. eps))

let gumbel_sample t ~temperature =
  let noise = gumbel_noise t 1e-10 in
  let scaled_t = Tensor.div t temperature in
  let gumbel_dist = Tensor.add scaled_t noise in
  Tensor.argmax gumbel_dist ~dim:(-1) ~keepdim:false

let prob_mask_like t prob =
  let zero_tensor = Tensor.zeros_like t in
  let uniform_tensor = Tensor.rand zero_tensor |> Tensor.mul_ prob in
  Tensor.lt uniform_tensor prob

let mask_with_tokens t token_ids =
  let init_no_mask = Tensor.full_like t false in
  let mask =
    List.fold_left
      (fun acc el -> Tensor.logical_or acc (Tensor.eq t (Tensor.of_int el)))
      init_no_mask token_ids
  in
  mask

let get_mask_subset_with_prob mask prob =
  let batch, seq_len = Tensor.shape mask in
  let max_masked = int_of_float (ceil (prob *. float seq_len)) in

  let num_tokens = Tensor.sum mask ~dim:[-1] ~keepdim:true in
  let mask_excess =
    Tensor.cumsum mask ~dim:[-1] ~dtype:(Torch.Long Torch.Device.Cpu)
    |> Tensor.gt (Tensor.mul num_tokens (Tensor.of_float prob |> Tensor.ceil))
  in
  let mask_excess = Tensor.slice mask_excess ~dim:1 ~start:0 ~end_:max_masked in

  let rand = Tensor.rand batch seq_len in
  let rand = Tensor.masked_fill rand (Tensor.logical_not mask) (Tensor.of_float (-1e9)) in

  let _, sampled_indices = Tensor.topk rand max_masked ~dim:1 ~largest:true ~sorted:true in
  let sampled_indices = Tensor.add sampled_indices (Tensor.of_int 1) in
  let sampled_indices = Tensor.masked_fill_ sampled_indices mask_excess (Tensor.of_int 0) in

  let new_mask = Tensor.zeros_ [batch; seq_len + 1] in
  Tensor.scatter_ new_mask ~dim:(-1) ~index:sampled_indices ~src:(Tensor.of_float 1.);
  Tensor.bool new_mask

module HiddenLayerExtractor = struct
  type t = { net : Torch.nn.Module.t; layer : int }

  let create net layer = { net; layer; hidden = None; hook_registered = false }

  let find_layer t =
    if t.layer >= 0 then
      let children = Torch.nn.Module.children t.net in
      List.nth children t.layer
    else
      let modules = Torch.nn.Module.named_children t.net in
      List.assoc (string_of_int t.layer) modules

  let hook t _ _ output =
    t.hidden <- Some output

  let register_hook t =
    let layer = find_layer t in
    assert (layer <> None);
    let layer = Option.get layer in
    let hook = Torch.nn.Module.register_forward_hook layer t.hook in
    t.hook_registered <- true

  let forward t x =
    if t.layer = -1 then Torch.nn.Module.forward t.net x
    else (
      if not t.hook_registered then register_hook t;
      let _ = Torch.nn.Module.forward t.net x in
      let hidden = t.hidden |> Option.get in
      t.hidden <- None;
      hidden
    )
end