open Torch

module CycleGAN = struct
  type t = {
    g_ab : Layer.t;
    g_ba : Layer.t;
    d_a : Layer.t;
    d_b : Layer.t;
    opt_g : Optimizer.t;
    opt_d_a : Optimizer.t;
    opt_d_b : Optimizer.t;
    criterion_gan : Tensor.t -> Tensor.t -> Tensor.t;
    criterion_cycle : Tensor.t -> Tensor.t -> Tensor.t;
    criterion_identity : Tensor.t -> Tensor.t -> Tensor.t;
    lambda_cyc : float;
    lambda_id : float;
  }

  let create_generator vs input_shape n_residual_blocks =
    let c, h, w = input_shape in
    Layer.sequential
      (Layer.conv2d vs c 64 ~kernel_size:7 ~stride:1 ~padding:3
       :: Layer.instance_norm2d vs 64
       :: Layer.relu
       :: Layer.conv2d vs 64 128 ~kernel_size:3 ~stride:2 ~padding:1
       :: Layer.instance_norm2d vs 128
       :: Layer.relu
       :: Layer.conv2d vs 128 256 ~kernel_size:3 ~stride:2 ~padding:1
       :: Layer.instance_norm2d vs 256
       :: Layer.relu
       :: List.init n_residual_blocks (fun _ ->
              Layer.sequential
                [ Layer.conv2d vs 256 256 ~kernel_size:3 ~stride:1 ~padding:1
                ; Layer.instance_norm2d vs 256
                ; Layer.relu
                ; Layer.conv2d vs 256 256 ~kernel_size:3 ~stride:1 ~padding:1
                ; Layer.instance_norm2d vs 256
                ])
       @ [ Layer.conv_transpose2d vs 256 128 ~kernel_size:3 ~stride:2 ~padding:1 ~output_padding:1
         ; Layer.instance_norm2d vs 128
         ; Layer.relu
         ; Layer.conv_transpose2d vs 128 64 ~kernel_size:3 ~stride:2 ~padding:1 ~output_padding:1
         ; Layer.instance_norm2d vs 64
         ; Layer.relu
         ; Layer.conv2d vs 64 c ~kernel_size:7 ~stride:1 ~padding:3
         ; Layer.tanh
         ])

  let create_discriminator vs input_shape =
    let c, h, w = input_shape in
    Layer.sequential
      [ Layer.conv2d vs c 64 ~kernel_size:4 ~stride:2 ~padding:1
      ; Layer.leaky_relu ~negative_slope:0.2
      ; Layer.conv2d vs 64 128 ~kernel_size:4 ~stride:2 ~padding:1
      ; Layer.instance_norm2d vs 128
      ; Layer.leaky_relu ~negative_slope:0.2
      ; Layer.conv2d vs 128 256 ~kernel_size:4 ~stride:2 ~padding:1
      ; Layer.instance_norm2d vs 256
      ; Layer.leaky_relu ~negative_slope:0.2
      ; Layer.conv2d vs 256 512 ~kernel_size:4 ~stride:1 ~padding:1
      ; Layer.instance_norm2d vs 512
      ; Layer.leaky_relu ~negative_slope:0.2
      ; Layer.conv2d vs 512 1 ~kernel_size:4 ~stride:1 ~padding:1
      ]

  let create ~input_shape ~n_residual_blocks ~lr ~beta1 ~beta2 ~lambda_cyc ~lambda_id =
    let vs = Var_store.create ~name:"cyclegan" () in
    let g_ab = create_generator vs input_shape n_residual_blocks in
    let g_ba = create_generator vs input_shape n_residual_blocks in
    let d_a = create_discriminator vs input_shape in
    let d_b = create_discriminator vs input_shape in
    let opt_g = Optimizer.adam (Var_store.all_vars vs) ~lr ~beta1 ~beta2 in
    let opt_d_a = Optimizer.adam (Var_store.all_vars vs) ~lr ~beta1 ~beta2 in
    let opt_d_b = Optimizer.adam (Var_store.all_vars vs) ~lr ~beta1 ~beta2 in
    { g_ab; g_ba; d_a; d_b; opt_g; opt_d_a; opt_d_b
    ; criterion_gan = Tensor.mse_loss
    ; criterion_cycle = Tensor.l1_loss
    ; criterion_identity = Tensor.l1_loss
    ; lambda_cyc; lambda_id
    }

  let train_generators t real_a real_b =
    Optimizer.zero_grad t.opt_g;
    let fake_b = Layer.forward t.g_ab real_a in
    let fake_a = Layer.forward t.g_ba real_b in
    let loss_id_a = t.criterion_identity (Layer.forward t.g_ba real_a) real_a in
    let loss_id_b = t.criterion_identity (Layer.forward t.g_ab real_b) real_b in
    let loss_identity = Tensor.(loss_id_a + loss_id_b) |> Tensor.mul_scalar 0.5 in
    let loss_gan_ab = t.criterion_gan (Layer.forward t.d_b fake_b) (Tensor.ones_like fake_b) in
    let loss_gan_ba = t.criterion_gan (Layer.forward t.d_a fake_a) (Tensor.ones_like fake_a) in
    let loss_gan = Tensor.(loss_gan_ab + loss_gan_ba) |> Tensor.mul_scalar 0.5 in
    let recov_a = Layer.forward t.g_ba fake_b in
    let recov_b = Layer.forward t.g_ab fake_a in
    let loss_cycle_a = t.criterion_cycle recov_a real_a in
    let loss_cycle_b = t.criterion_cycle recov_b real_b in
    let loss_cycle = Tensor.(loss_cycle_a + loss_cycle_b) |> Tensor.mul_scalar 0.5 in
    let loss_g = Tensor.(loss_gan + (f t.lambda_cyc * loss_cycle) + (f t.lambda_id * loss_identity)) in
    Tensor.backward loss_g;
    Optimizer.step t.opt_g;
    loss_g, fake_a, fake_b

  let train_discriminator t opt_d real fake =
    Optimizer.zero_grad opt_d;
    let loss_real = t.criterion_gan (Layer.forward t.d_a real) (Tensor.ones_like real) in
    let loss_fake = t.criterion_gan (Layer.forward t.d_a fake) (Tensor.zeros_like fake) in
    let loss_d = Tensor.(loss_real + loss_fake) |> Tensor.mul_scalar 0.5 in
    Tensor.backward loss_d;
    Optimizer.step opt_d;
    loss_d

  let train_step t real_a real_b =
    let loss_g, fake_a, fake_b = train_generators t real_a real_b in
    let loss_d_a = train_discriminator t t.opt_d_a real_a fake_a in
    let loss_d_b = train_discriminator t t.opt_d_b real_b fake_b in
    loss_g, Tensor.((loss_d_a + loss_d_b) / f 2.)
end