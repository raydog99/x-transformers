open Base
open Torch

let exists (v : 'a option) : bool =
  Option.is_some v

let default (v : 'a option) (d : 'a) : 'a =
  Option.value v ~default:d

let l1norm (t : Tensor.t) : Tensor.t =
  Tensor.normalize t ~p:1 ~dim:(-1)

let l2norm (t : Tensor.t) : Tensor.t =
  Tensor.normalize t ~dim:(-1)

let cosine_sim_loss (x : Tensor.t) (y : Tensor.t) : Tensor.t =
  let l2norm t = l2norm t in
  let x, y = l2norm x, l2norm y in
  let similarity = Tensor.einsum "bnd,bnd->bn" [|x; y|] in
  Tensor.neg (Tensor.mean similarity)

let posemb_sincos_nd
    (t : Tensor.t)
    ?(temperature : int = 10000)
    ?(dtype : Kind.t = Kind.Float)
  : Tensor.t =
  let b, dims, feat_dim, device = Tensor.shape t in
  let seq_len = Tensor.cumprod ~dim:(-1) dims |> Tensor.item in
  let arange = Tensor.arange ~device in
  let num_dims = List.length dims in
  let two_times_num_dims = 2 * num_dims in

  let rounded_feat_dim = feat_dim / num_dims * num_dims in
  let feat_dim_remainder = feat_dim % num_dims in

  let omega = arange (rounded_feat_dim / two_times_num_dims) / (rounded_feat_dim / two_times_num_dims - 1) in
  let omega = 1.0 / (float_of_int temperature ** omega) in

  let meshed = List.map arange dims |> Tensor.meshgrid ~indexing:`ij in

  let pos = List.map (fun m -> Tensor.flatten m |> Tensor.unsqueeze ~dim:(-1)) meshed |> Tensor.cat ~dim:0 in
  let pos = Tensor.mul pos omega |> Tensor.sin, Tensor.cos in
  let pos = Tensor.cat [pos] ~dim:0 in

  let pos = Tensor.rearrange pos ~pattern:"(nf)d -> n(fd)" ~n:seq_len ~f:two_times_num_dims in
  let pos = Tensor.to_type pos ~type_kind:dtype in

  Tensor.pad pos ~padding:(0, feat_dim_remainder)

let mask_with_prob
    (shape : int list)
    (prob : float)
    ?(device : Device.t option)
  : Tensor.t =
  let length = List.hd (List.rev shape) in
  let num_mask = int_of_float (prob *. float_of_int length) in
  let randperm = Tensor.randn shape ~device |> Tensor.argsort ~dim:(-1) in
  Tensor.ge randperm (Tensor.of_int num_mask)

let autoregressive_q_learn
    (model : Module.t)
    (ema_model : Module.t)
    (states : Tensor.t)
    (prompt_len : Tensor.t)
    (next_states : Tensor.t)
    (rewards : Tensor.t)
    ?(eos_id : int option = None)
    ?(discount_gamma : float = 0.998)
  : Tensor.t =
  let seq_len, device = Tensor.shape states |> Tuple2.get1 |> Array.get ~-1, Tensor.device states in
  let gamma = discount_gamma in

  let q_pred_all_actions = Module.forward model states in
  let q_pred = Tensor.batch_select_indices q_pred_all_actions actions in

  let q_target_input = Tensor.pack [Tensor.slice states ~dim:1 ~start:1; next_state] ~dim:"b *" in

  let q_target = Module.forward ema_model q_target_input in
  let q_target = Tensor.max ~dim:(-1) q_target |> Tuple2.get1 in

  let done_reached =
    match eos_id with
    | Some eos_id ->
      let done_mask = Tensor.(eq states (of_int eos_id)) in
      let done_cumsum = Tensor.cumsum ~dim:(-1) done_mask in
      let done_pad = Tensor.pad done_cumsum ~padding:(1, -1) ~value:0 in
      let terminal_mask = Tensor.(done_pad > of_int 0) in
      Tensor.float not terminal_mask
    | None -> Tensor.(ones_like rewards)
  in

  let rewards_masked = Tensor.(rewards * done_reached) in
  let q_target_masked = Tensor.(q_target * (ones_like rewards |> Tensor.logical_not |> Tensor.float)) in

  let losses_without_rewards = Tensor.mse_loss q_pred q_target_masked ~reduction:`None in
  let q_target_with_rewards = Tensor.(rewards_masked + (gamma * q_target)) in
  let losses_with_rewards = Tensor.mse_loss q_pred q_target_with_rewards ~reduction:`None in

  let losses =
    Tensor.where_
      (Tensor.gt rewards (Tensor.zeros_like rewards))
      ~x:losses_with_rewards
      ~y:losses_without_rewards
  in

  let is_action_mask = Tensor.(arange seq_len |> gt (Tensor.unsqueeze prompt_len ~dim:(-1))) in
  Tensor.masked_select losses is_action_mask |> Tensor.mean