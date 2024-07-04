open Torch

module ImplicitScoreMatching = struct
  let loss model x t noise_schedule =
    let open Tensor in
    let x_noisy = Diffusion.forward_process t x noise_schedule in
    let score = ScoreNetwork.forward model x_noisy t in
    let grad_x = grad_outputs x_noisy [one_like x_noisy] ~create_graph:true ~retain_graph:true in
    let grad_score = sum (grad_x * score) in
    let score_norm = Tensor.(sum (pow score (Scalar.f 2.))) in
    Tensor.((Scalar.f 0.5 * score_norm) + grad_score)

  let train model x t noise_schedule optimizer =
    Optimizer.zero_grad optimizer;
    let loss = loss model x t noise_schedule in
    backward loss;
    Optimizer.step optimizer;
    Tensor.to_float0_exn loss

  let sample model initial_x num_steps noise_schedule =
    let rec sample_loop x step =
      if step = 0 then x
      else
        let t = Tensor.of_float0 ((float_of_int step) /. (float_of_int num_steps)) in
        let x_new = Diffusion.reverse_process model x t noise_schedule in
        sample_loop x_new (step - 1)
    in
    sample_loop initial_x num_steps
end