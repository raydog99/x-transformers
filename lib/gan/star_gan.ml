open Torch

module GeneratorResNet = struct
  type t = {
    model : Nn.t;
    img_shape : int * int * int;
    res_blocks : int;
    c_dim : int;
  }

  let create img_shape res_blocks c_dim =
    let model = Nn.sequential
      [ Nn.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:7 ~stride:1 ~padding:3 ();
        Nn.instance_norm2d 64;
        Nn.relu ();
        (* Downsampling *)
        Nn.conv2d ~in_channels:64 ~out_channels:128 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.instance_norm2d 128;
        Nn.relu ();
        Nn.conv2d ~in_channels:128 ~out_channels:256 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.instance_norm2d 256;
        Nn.relu ();
        (* Residual blocks *)
        List.init res_blocks (fun _ ->
          Nn.sequential [
            Nn.conv2d ~in_channels:256 ~out_channels:256 ~kernel_size:3 ~stride:1 ~padding:1 ();
            Nn.instance_norm2d 256;
            Nn.relu ();
            Nn.conv2d ~in_channels:256 ~out_channels:256 ~kernel_size:3 ~stride:1 ~padding:1 ();
            Nn.instance_norm2d 256;
          ]
        );
        (* Upsampling *)
        Nn.conv_transpose2d ~in_channels:256 ~out_channels:128 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.instance_norm2d 128;
        Nn.relu ();
        Nn.conv_transpose2d ~in_channels:128 ~out_channels:64 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.instance_norm2d 64;
        Nn.relu ();
        Nn.conv2d ~in_channels:64 ~out_channels:3 ~kernel_size:7 ~stride:1 ~padding:3 ();
        Nn.tanh ();
      ] in
    { model; img_shape; res_blocks; c_dim }

  let forward t x c =
    let c = Tensor.view c ~size:[ Tensor.shape c |> List.hd; t.c_dim; 1; 1 ] in
    let c = Tensor.expand c ~size:(Tensor.shape x) in
    let x = Tensor.cat [ x; c ] ~dim:1 in
    Nn.forward t.model x
end

module Discriminator = struct
  type t = {
    model : Nn.t;
    img_shape : int * int * int;
    c_dim : int;
  }

  let create img_shape c_dim =
    let model = Nn.sequential
      [ Nn.conv2d ~in_channels:3 ~out_channels:64 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.leaky_relu ~negative_slope:0.01 ();
        Nn.conv2d ~in_channels:64 ~out_channels:128 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.leaky_relu ~negative_slope:0.01 ();
        Nn.conv2d ~in_channels:128 ~out_channels:256 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.leaky_relu ~negative_slope:0.01 ();
        Nn.conv2d ~in_channels:256 ~out_channels:512 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.leaky_relu ~negative_slope:0.01 ();
        Nn.conv2d ~in_channels:512 ~out_channels:1024 ~kernel_size:4 ~stride:2 ~padding:1 ();
        Nn.leaky_relu ~negative_slope:0.01 ();
        Nn.conv2d ~in_channels:1024 ~out_channels:1 ~kernel_size:3 ~stride:1 ~padding:1 ();
        Nn.conv2d ~in_channels:1024 ~out_channels:c_dim ~kernel_size:4 ~stride:1 ~padding:0 ();
      ] in
    { model; img_shape; c_dim }

  let forward t x =
    let output = Nn.forward t.model x in
    let out_adv = Tensor.slice output ~dim:1 ~start:0 ~end_:1 in
    let out_cls = Tensor.slice output ~dim:1 ~start:1 ~end_:(t.c_dim + 1) in
    (out_adv, out_cls)
end

let compute_gradient_penalty d real_samples fake_samples =
  let batch_size = Tensor.shape real_samples |> List.hd in
  let alpha = Tensor.rand [ batch_size; 1; 1; 1 ] ~device:(Tensor.device real_samples) in
  let interpolates = Tensor.(alpha * real_samples + ((Scalar.f 1. - alpha) * fake_samples)) in
  let interpolates = Tensor.set_requires_grad interpolates true in
  let (d_interpolates, _) = Discriminator.forward d interpolates in
  let gradients = Tensor.grad d_interpolates ~inputs:[interpolates] ~create_graph:true ~retain_graph:true in
  let gradients = Tensor.view gradients ~size:[batch_size; -1] in
  let gradient_penalty = Tensor.(mean (pow (norm gradients ~p:(Scalar.f 2.) ~dim:[1] - (Scalar.f 1.)) (Scalar.f 2.))) in
  gradient_penalty

let sample_images g val_imgs val_labels batches_done =
  let img_samples = ref None in
  for i = 0 to 9 do
    let img = Tensor.select val_imgs 0 i in
    let label = Tensor.select val_labels 0 i in
    let imgs = Tensor.repeat img [5; 1; 1; 1] in
    let labels = Tensor.repeat label [5; 1] in
    let label_changes = [
      [|(0, 1); (1, 0); (2, 0)|];
      [|(0, 0); (1, 1); (2, 0)|];
      [|(0, 0); (1, 0); (2, 1)|];
      [|(3, -1)|];
      [|(4, -1)|];
    ] in
    List.iteri (fun sample_i changes ->
      Array.iter (fun (col, val_) ->
        let new_val = if val_ = -1 then Tensor.(f 1. - (select labels sample_i col))
                      else Tensor.f (float_of_int val_) in
        Tensor.select_set labels sample_i col new_val
      ) changes
    ) label_changes;
    let gen_imgs = GeneratorResNet.forward g imgs labels in
    let gen_imgs = Tensor.cat (List.init 5 (fun i -> Tensor.select gen_imgs i)) ~dim:2 in
    let img_sample = Tensor.cat [img; gen_imgs] ~dim:2 in
    img_samples := match !img_samples with
      | None -> Some img_sample
      | Some samples -> Some (Tensor.cat [samples; img_sample] ~dim:1)
  done;
  Tensor.save_image !img_samples ~filename:(Printf.sprintf "images/%d.png" batches_done) ~normalize:true