open Torch

let autoregressive_q_learn
    (model : Torch.torch_nn_module)
    (ema_model : Torch.torch_nn_module)
    (states : Torch.torch_tensor)
    (prompt_len : Torch.torch_tensor)
    (next_state : Torch.torch_tensor)
    (rewards : Torch.torch_tensor)
    ?(eos_id : int option)
    (discount_gamma : float)
  : Torch.torch_tensor =
  let seq_len = Torch.Tensor.shape states |> Array.get 2 in
  let device = Torch.Tensor.device states in

  let gamma = discount_gamma in

  let q_pred_all_actions = Torch.Module.forward model [|states|] in
  let q_pred = Torch.Tensor.index_select q_pred_all_actions ~dim:1 ~index:actions in

  let q_target_input = Torch.Tensor.cat [|Torch.Tensor.index_select states ~dim:1 ~index:(Torch.Tensor.range 1 (seq_len - 1) ~options:{Torch.Tensor.dtype = Long}); next_state|] ~dim:1 in

  let q_target_all_actions = Torch.Module.forward ema_model [|q_target_input|] in
  let q_target = Torch.Tensor.max q_target_all_actions ~dim:3 |> Torch.Tensor.values in

  let not_terminal = match eos_id with
    | Some eos ->
      let done = Torch.Tensor.eq states (Torch.Tensor.of_int eos) in
      let dones = Torch.Tensor.cumsum ~dim:2 done |> Torch.Tensor.gt (Torch.Tensor.zeros_like done) in
      let dones = Torch.Tensor.pad ~padding:(Torch.Tensor.of_int_list [0; 1; 0; -1]) dones ~value:(Torch.Tensor.zeros_like done) in
      Torch.Tensor.logical_not dones |> Torch.Tensor.to_type Float
    | None -> Torch.Tensor.ones_like states
  in

  let rewards = Torch.Tensor.mul rewards not_terminal in
  let q_target = Torch.Tensor.masked_fill q_target ~mask:dones ~value:(Torch.Tensor.zeros_like q_target) in

  let losses_without_rewards = Torch.Tensor.mse_loss q_pred q_target ~reduction:`None in
  let q_target_with_rewards = Torch.Tensor.add rewards (Torch.Tensor.mul gamma q_target) in
  let losses_with_rewards = Torch.Tensor.mse_loss q_pred q_target_with_rewards ~reduction:`None in

  let losses = Torch.Tensor.where (Torch.Tensor.gt rewards (Torch.Tensor.zeros_like rewards)) ~condition:losses_with_rewards ~x:losses_without_rewards in

  let is_action_mask = Torch.Tensor.gt (Torch.Tensor.range 0 seq_len ~options:{Torch.Tensor.device = device}) (Torch.Tensor.rearrange prompt_len ~pattern:"b -> b 1") in
  let losses = Torch.Tensor.index losses ~index:is_action_mask in
  Torch.Tensor.mean losses