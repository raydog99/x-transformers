open Torch

let slot_attention num_slots dim ~iters ~eps ~hidden_dim =
  let scale = Float.pow (float_of_int dim) (-0.5) in
  let slots_mu = nn_parameter (Tensor.randn [1; 1; dim]) in
  let slots_logsigma = nn_parameter (Tensor.zeros [1; 1; dim]) in
  nn_init.xavier_uniform_ slots_logsigma;

  let to_q = nn_linear ~input_dim:dim ~output_dim:dim () in
  let to_k = nn_linear ~input_dim:dim ~output_dim:dim () in
  let to_v = nn_linear ~input_dim:dim ~output_dim:dim () in

  let gru = nn_gru_cell ~input_dim:dim ~hidden_dim:dim () in

  let hidden_dim = max dim hidden_dim in

  let mlp =
    nn_sequential
      [
        nn_linear ~input_dim:dim ~output_dim:hidden_dim ();
        nn_relu ~inplace:true;
        nn_linear ~input_dim:hidden_dim ~output_dim:dim ();
      ]
  in

  let norm_input = nn_layer_norm [dim] in
  let norm_slots = nn_layer_norm [dim] in
  let norm_pre_ff = nn_layer_norm [dim] in

  fun inputs ?(num_slots = None) () ->
    let b, n, d = Tensor.shape inputs in
    let num_slots = match num_slots with Some ns -> ns | None -> num_slots in
    let mu = Tensor.expand_as slots_mu [|b; num_slots; dim|] in
    let sigma = Tensor.exp slots_logsigma |> Tensor.expand_as [|b; num_slots; dim|] in
    let normal_samples = Tensor.randn [b; num_slots; dim] |> Tensor.to_device inputs in
    let slots = Tensor.(mu + (normal_samples * sigma)) in

    let inputs = norm_input inputs in
    let k = Tensor.(inputs |> to_k) in
    let v = Tensor.(inputs |> to_v) in

    let rec loop slots =
      let slots_prev = slots in
      let slots = norm_slots slots in
      let q = Tensor.(slots |> to_q) in

      let dots = Torch.einsum "bid,bjd->bij" [q; k] |> Tensor.mul_scalar scale in
      let attn = Tensor.(dots |> softmax ~dim:1 ~dtype:D.Long |> to_type D.Float) +. eps in

      let attn = Tensor.(attn / (attn |> Tensor.sum ~dim:-1 ~keepdim:true)) in
      let updates = Torch.einsum "bjd,bij->bid" [v; attn] in

      let slots_flat = Tensor.reshape slots [-1; d] in
      let slots_prev_flat = Tensor.reshape slots_prev [-1; d] in
      let slots_updated = gru updates slots_prev_flat |> Tensor.reshape [|b; -1; d|] in

      let slots = Tensor.(slots_updated + (slots |> mlp |> norm_pre_ff)) in
      if iters > 1 then loop slots else slots
    in

    loop slots