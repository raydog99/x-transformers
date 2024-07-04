open Torch

module DualGAN = struct
  module Generator = struct
    let create () =
      Nn.sequential
        [
          Nn.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:4 ~stride:2 ~padding:1 ();
          Nn.leaky_relu ~negative_slope:0.2 ();
          Nn.conv2d ~in_channels:64 ~out_channels:3 ~kernel_size:4 ~stride:2 ~padding:1 ();
          Nn.tanh ()
        ]

    let forward model x = Nn.Module.forward model x
  end

  module Discriminator = struct
    let create () =
      Nn.sequential
        [
          Nn.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:4 ~stride:2 ~padding:1 ();
          Nn.leaky_relu ~negative_slope:0.2 ();
          Nn.conv2d ~in_channels:64 ~out_channels:1 ~kernel_size:4 ~stride:1 ~padding:0 ();
        ]

    let forward model x = Nn.Module.forward model x
  end

  let cycle_loss = Nn.mse_loss

  let compute_gradient_penalty d real_samples fake_samples =
    let batch_size = Tensor.shape real_samples |> List.hd in
    let alpha = Tensor.rand [batch_size; 1; 1; 1] in
    let interpolates = Tensor.(alpha * real_samples + ((scalar 1. - alpha) * fake_samples)) in
    let interpolates = Tensor.set_requires_grad interpolates true in
    let d_interpolates = Discriminator.forward d interpolates in
    let gradients = Tensor.grad d_interpolates ~inputs:[interpolates] ~create_graph:true ~retain_graph:true in
    let gradients = List.hd gradients in
    let gradients = Tensor.view gradients [batch_size; -1] in
    let gradient_penalty = Tensor.((pow (norm gradients ~p:(Scalar.f 2.) ~dim:[1] - (scalar 1.)) (scalar 2.)) |> mean)
    in
    gradient_penalty

  let train g_ab g_ba d_a d_b ~learning_rate ~n_epochs ~batch_size ~n_critic ~lambda_adv ~lambda_cycle ~lambda_gp =
    let opt_g = Optimizer.adam (Nn.Module.parameters g_ab @ Nn.Module.parameters g_ba) ~lr:learning_rate in
    let opt_d_a = Optimizer.adam (Nn.Module.parameters d_a) ~lr:learning_rate in
    let opt_d_b = Optimizer.adam (Nn.Module.parameters d_b) ~lr:learning_rate in

    for epoch = 1 to n_epochs do
      let imgs_a = Tensor.randn [batch_size; 3; 128; 128] in
      let imgs_b = Tensor.randn [batch_size; 3; 128; 128] in

      Optimizer.zero_grad opt_d_a;
      Optimizer.zero_grad opt_d_b;

      let fake_a = Generator.forward g_ba imgs_b in
      let fake_b = Generator.forward g_ab imgs_a in

      let gp_a = compute_gradient_penalty d_a imgs_a fake_a in
      let gp_b = compute_gradient_penalty d_b imgs_b fake_b in

      let d_a_loss = Tensor.(mean (Discriminator.forward d_a fake_a) - mean (Discriminator.forward d_a imgs_a) + (scalar lambda_gp * gp_a)) in
      let d_b_loss = Tensor.(mean (Discriminator.forward d_b fake_b) - mean (Discriminator.forward d_b imgs_b) + (scalar lambda_gp * gp_b)) in

      let d_loss = Tensor.(d_a_loss + d_b_loss) in
      Tensor.backward d_loss;
      Optimizer.step opt_d_a;
      Optimizer.step opt_d_b;

      if epoch mod n_critic = 0 then begin
        Optimizer.zero_grad opt_g;

        let fake_a = Generator.forward g_ba imgs_b in
        let fake_b = Generator.forward g_ab imgs_a in

        let recov_a = Generator.forward g_ba fake_b in
        let recov_b = Generator.forward g_ab fake_a in

        let g_adv = Tensor.(-(mean (Discriminator.forward d_a fake_a) + mean (Discriminator.forward d_b fake_b))) in
        let g_cycle = Tensor.(cycle_loss recov_a imgs_a + cycle_loss recov_b imgs_b) in
        let g_loss = Tensor.((scalar lambda_adv * g_adv) + (scalar lambda_cycle * g_cycle)) in

        Tensor.backward g_loss;
        Optimizer.step opt_g;

        Printf.printf "Epoch [%d/%d] D_loss: %.4f, G_adv: %.4f, G_cycle: %.4f\n"
          epoch n_epochs
          (Tensor.to_float0_exn d_loss)
          (Tensor.to_float0_exn g_adv)
          (Tensor.to_float0_exn g_cycle);
      end;
    done;
    (g_ab, g_ba, d_a, d_b)

  let sample_images g_ab g_ba =
    let real_a = Tensor.randn [1; 3; 128; 128] in
    let real_b = Tensor.randn [1; 3; 128; 128] in
    let fake_b = Generator.forward g_ab real_a in
    let fake_a = Generator.forward g_ba real_b in
    (real_a, fake_b, real_b, fake_a)
end