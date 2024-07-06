open Torch

let noise_distribution mean std =
  fun x -> Tensor.(normal ~mean ~std x)

let train_step model data noise_samples optimizer =
  let open Tensor in
  let bce = binary_cross_entropy_with_logits ~reduction:Mean in
  Optimizer.zero_grad optimizer;
  let data_logits = M.forward data in
  let noise = noise_distribution (mean data) (std data) noise_samples in
  let noise_logits = M.forward noise in
  let data_labels = ones (shape data_logits) in
  let noise_labels = zeros (shape noise_logits) in
  let loss =
    bce data_logits data_labels +
    bce noise_logits noise_labels
  in
  backward loss;
  Optimizer.step optimizer;
  loss

let train model data noise_ratio num_epochs learning_rate =
  let optimizer = Optimizer.adam M.parameters ~lr:learning_rate in
  for epoch = 1 to num_epochs do
    let batch_size = Tensor.shape data |> List.hd in
    let noise_samples = Tensor.randn [batch_size * noise_ratio] in
    let loss = train_step model data noise_samples optimizer in
    Printf.printf "Epoch %d: Loss = %f\n" epoch (Tensor.to_float0_exn loss)
  done