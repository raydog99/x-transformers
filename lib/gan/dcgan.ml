open Torch

let weights_init_normal m =
  match m with
  | Layer.Conv2D { weight; _ } ->
      Tensor.normal_into weight ~mean:0.0 ~std:0.02
  | Layer.BatchNorm2D { weight; bias; _ } ->
      Tensor.normal_into weight ~mean:1.0 ~std:0.02;
      Tensor.zero_ bias
  | _ -> ()

module Generator = struct
  type t = {
    l1: Layer.t;
    conv_blocks: Layer.t;
    init_size: int;
  }

  let create () =
    let init_size = !opt.img_size / 4 in
    let l1 = Layer.linear ~in_dim:!opt.latent_dim ~out_dim:(128 * init_size * init_size) () in
    let conv_blocks =
      Layer.of_list [
        Layer.batch_norm2d ~num_features:128 ();
        Layer.upsample ~scale_factor:2. ();
        Layer.conv2d ~in_channels:128 ~out_channels:128 ~kernel_size:3 ~stride:1 ~padding:1 ();
        Layer.batch_norm2d ~num_features:128 ~momentum:0.8 ();
        Layer.leaky_relu ~negative_slope:0.2 ();
        Layer.upsample ~scale_factor:2. ();
        Layer.conv2d ~in_channels:128 ~out_channels:64 ~kernel_size:3 ~stride:1 ~padding:1 ();
        Layer.batch_norm2d ~num_features:64 ~momentum:0.8 ();
        Layer.leaky_relu ~negative_slope:0.2 ();
        Layer.conv2d ~in_channels:64 ~out_channels:!opt.channels ~kernel_size:3 ~stride:1 ~padding:1 ();
        Layer.tanh ();
      ]
    in
    { l1; conv_blocks; init_size }

  let forward t z =
    let out = Layer.forward t.l1 z in
    let out = Tensor.view out ~size:[-1; 128; t.init_size; t.init_size] in
    Layer.forward t.conv_blocks out
end

module Discriminator = struct
  type t = {
    model: Layer.t;
    adv_layer: Layer.t;
  }

  let discriminator_block in_filters out_filters bn =
    let block = [
      Layer.conv2d ~in_channels:in_filters ~out_channels:out_filters ~kernel_size:3 ~stride:2 ~padding:1 ();
      Layer.leaky_relu ~negative_slope:0.2 ();
      Layer.dropout2d ~p:0.25 ();
    ] in
    if bn then
      block @ [Layer.batch_norm2d ~num_features:out_filters ~momentum:0.8 ()]
    else
      block

  let create () =
    let model =
      Layer.of_list (
        discriminator_block !opt.channels 16 false @
        discriminator_block 16 32 true @
        discriminator_block 32 64 true @
        discriminator_block 64 128 true
      )
    in
    let ds_size = !opt.img_size / (2 ** 4) in
    let adv_layer =
      Layer.of_list [
        Layer.linear ~in_dim:(128 * ds_size * ds_size) ~out_dim:1 ();
        Layer.sigmoid ();
      ]
    in
    { model; adv_layer }

  let forward t img =
    let out = Layer.forward t.model img in
    let out = Tensor.view out ~size:[-1; Tensor.shape out |> List.last] in
    Layer.forward t.adv_layer out
end

let adversarial_loss = Loss.binary_cross_entropy