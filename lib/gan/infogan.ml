open Torch

module Generator = struct
  type t = {
    l1: (float, [`D1]) t;
    conv_blocks: (float, [`D4]) t;
    init_size: int;
  }

  let create () =
    let input_dim = !opt.latent_dim + !opt.n_classes + !opt.code_dim in
    let init_size = !opt.img_size / 4 in
    let l1 = Layer.linear ~in_dim:input_dim ~out_dim:(128 * init_size * init_size) () in
    let conv_blocks =
      Layer.sequential
        [Layer.batch_norm2d ~num_features:128 ();
         Layer.upsample ~scale_factor:2. ();
         Layer.conv2d ~in_channels:128 ~out_channels:128 ~kernel_size:3 ~stride:1 ~padding:1 ();
         Layer.batch_norm2d ~num_features:128 ~momentum:0.8 ();
         Layer.leaky_relu ~negative_slope:0.2 ();
         Layer.upsample ~scale_factor:2. ();
         Layer.conv2d ~in_channels:128 ~out_channels:64 ~kernel_size:3 ~stride:1 ~padding:1 ();
         Layer.batch_norm2d ~num_features:64 ~momentum:0.8 ();
         Layer.leaky_relu ~negative_slope:0.2 ();
         Layer.conv2d ~in_channels:64 ~out_channels:!opt.channels ~kernel_size:3 ~stride:1 ~padding:1 ();
         Layer.tanh ()]
    in
    { l1; conv_blocks; init_size }

  let forward t noise labels code =
    let gen_input = Tensor.cat [noise; labels; code] ~dim:(-1) in
    let out = Layer.forward t.l1 gen_input in
    let out = Tensor.view out ~size:[Tensor.shape out |> List.hd; 128; t.init_size; t.init_size] in
    Layer.forward t.conv_blocks out
end

module Discriminator = struct
  type t = {
    conv_blocks: (float, [`D4]) t;
    adv_layer: (float, [`D1]) t;
    aux_layer: (float, [`D1]) t;
    latent_layer: (float, [`D1]) t;
  }

  let discriminator_block in_filters out_filters bn =
    let block = [
      Layer.conv2d ~in_channels:in_filters ~out_channels:out_filters ~kernel_size:3 ~stride:2 ~padding:1 ();
      Layer.leaky_relu ~negative_slope:0.2 ();
      Layer.dropout2d ~p:0.25 ()]
    in
    if bn then Layer.batch_norm2d ~num_features:out_filters ~momentum:0.8 () :: block else block

  let create () =
    let conv_blocks =
      Layer.sequential
        (discriminator_block !opt.channels 16 false @
         discriminator_block 16 32 true @
         discriminator_block 32 64 true @
         discriminator_block 64 128 true)
    in
    let ds_size = !opt.img_size / (2 ** 4) in
    let adv_layer = Layer.linear ~in_dim:(128 * ds_size * ds_size) ~out_dim:1 () in
    let aux_layer =
      Layer.sequential
        [Layer.linear ~in_dim:(128 * ds_size * ds_size) ~out_dim:!opt.n_classes ();
         Layer.softmax ~dim:(-1) ()]
    in
    let latent_layer = Layer.linear ~in_dim:(128 * ds_size * ds_size) ~out_dim:!opt.code_dim () in
    { conv_blocks; adv_layer; aux_layer; latent_layer }

  let forward t img =
    let out = Layer.forward t.conv_blocks img in
    let out = Tensor.view out ~size:[Tensor.shape out |> List.hd; -1] in
    let validity = Layer.forward t.adv_layer out in
    let label = Layer.forward t.aux_layer out in
    let latent_code = Layer.forward t.latent_layer out in
    (validity, label, latent_code)
end

let adversarial_loss = Loss.mse_loss
let categorical_loss = Loss.cross_entropy_loss
let continuous_loss = Loss.mse_loss

let lambda_cat = 1.
let lambda_con = 0.1

let generator = Generator.create ()
let discriminator = Discriminator.create ()

let optimizer_G = Optimizer.adam (Generator.parameters generator) ~lr:!opt.lr
let optimizer_D = Optimizer.adam (Discriminator.parameters discriminator) ~lr:!opt.lr
let optimizer_info = Optimizer.adam (Generator.parameters generator @ Discriminator.parameters discriminator) ~lr:!opt.lr

let static_z = Tensor.zeros [!opt.n_classes * !opt.n_classes; !opt.latent_dim]
let static_label = to_categorical (Array.init (!opt.n_classes * !opt.n_classes) (fun i -> i / !opt.n_classes)) ~num_columns:!opt.n_classes
let static_code = Tensor.zeros [!opt.n_classes * !opt.n_classes; !opt.code_dim]

let sample_image n_row batches_done =
  let z = Tensor.randn [n_row * n_row; !opt.latent_dim] in
  let static_sample = Generator.forward generator z static_label static_code in
  save_image static_sample ~filename:(Printf.sprintf "images/static/%d.png" batches_done) ~nrow:n_row ~normalize:true;
  
  let zeros = Tensor.zeros [n_row * n_row; 1] in
  let c_varied = Tensor.linspace (-1.) 1. n_row |> Tensor.repeat [n_row; 1] in
  let c1 = Tensor.cat [c_varied; zeros] ~dim:1 in
  let c2 = Tensor.cat [zeros; c_varied] ~dim:1 in
  let sample1 = Generator.forward generator static_z static_label c1 in
  let sample2 = Generator.forward generator static_z static_label c2 in
  save_image sample1 ~filename:(Printf.sprintf "images/varying_c1/%d.png" batches_done) ~nrow:n_row ~normalize:true;
  save_image sample2 ~filename:(Printf.sprintf "images/varying_c2/%d.png" batches_done) ~nrow:n_row ~normalize:true