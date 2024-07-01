open Torch

module Opt = struct
  let n_epochs = 200
  let batch_size = 64
  let lr = 0.00005
  let latent_dim = 100
  let img_size = 28
  let channels = 1
  let n_critic = 5
  let clip_value = 0.01
  let sample_interval = 400
end

let img_shape = [Opt.channels; Opt.img_size; Opt.img_size]

module Generator = struct
  let block in_feat out_feat normalize =
    let layers = [Layer.linear ~in_features:in_feat ~out_features:out_feat ()]
    in
    let layers =
      if normalize then
        Layer.batch_norm1d ~num_features:out_feat ~eps:0.8 () :: layers
      else layers
    in
    Layer.leaky_relu ~negative_slope:0.2 ~inplace:true () :: layers

  let model =
    Layer.sequential
      (List.concat
         [ block Opt.latent_dim 128 false
         ; block 128 256 true
         ; block 256 512 true
         ; block 512 1024 true
         ; [ Layer.linear ~in_features:1024 ~out_features:(List.fold_left ( * ) 1 img_shape) ()
           ; Layer.tanh () ] ])

  let forward z = Tensor.view (Layer.forward model z) ~size:(-1 :: img_shape)
end

module Discriminator = struct
  let model =
    Layer.sequential
      [ Layer.linear ~in_features:(List.fold_left ( * ) 1 img_shape) ~out_features:512 ()
      ; Layer.leaky_relu ~negative_slope:0.2 ~inplace:true ()
      ; Layer.linear ~in_features:512 ~out_features:256 ()
      ; Layer.leaky_relu ~negative_slope:0.2 ~inplace:true ()
      ; Layer.linear ~in_features:256 ~out_features:1 () ]

  let forward img =
    Layer.forward model (Tensor.view img ~size:[-1; List.fold_left ( * ) 1 img_shape])
end

let generator = Generator.model
let discriminator = Discriminator.model

let optimizer_g = Optimizer.rmsprop generator ~lr:Opt.lr
let optimizer_d = Optimizer.rmsprop discriminator ~lr:Opt.lr

let dataloader =
  Dataset.load_mnist ~train:true ()
  |> Dataset.map ~f:(fun (data, _) -> Tensor.(data * f 2. - f 1.))
  |> Dataset.to_dataloader ~batch_size:Opt.batch_size ~shuffle:true

let rec train epoch batches_done =
  if epoch = Opt.n_epochs then ()
  else
    let rec train_batch i batches_done =
      match Dataset.Batch.next dataloader with
      | None -> batches_done
      | Some real_imgs ->
          let batch_size = Tensor.shape real_imgs |> List.hd in
          Optimizer.zero_grad optimizer_d;
          let z =
            Tensor.randn [batch_size; Opt.latent_dim]
            |> Tensor.to_device ~device:Device.cuda
          in
          let fake_imgs = Generator.forward z |> Tensor.detach in
          let loss_d =
            Tensor.(
              neg (mean (Discriminator.forward real_imgs))
              + mean (Discriminator.forward fake_imgs))
          in
          Tensor.backward loss_d;
          Optimizer.step optimizer_d;
          List.iter
            (fun p ->
              Tensor.clamp_ (Var_store.trainable_vars discriminator).(p)
                ~min:(Scalar.f Opt.clip_value)
                ~max:(Scalar.f Opt.clip_value))
            (List.init
               (Array.length (Var_store.trainable_vars discriminator))
               (fun x -> x));
          if i mod Opt.n_critic = 0 then (
            Optimizer.zero_grad optimizer_g;
            let gen_imgs = Generator.forward z in
            let loss_g = Tensor.neg (Tensor.mean (Discriminator.forward gen_imgs)) in
            Tensor.backward loss_g;
            Optimizer.step optimizer_g;
            Stdio.printf "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
              epoch Opt.n_epochs
              (batches_done mod Dataset.length dataloader)
              (Dataset.length dataloader) (Tensor.to_float loss_d)
              (Tensor.to_float loss_g);
            if batches_done mod Opt.sample_interval = 0 then
              Tensor.narrow gen_imgs ~dim:0 ~start:0 ~length:25
              |> Tensor.to_device ~device:Device.cpu
              |> Imagenet.write_image ~filename:(Printf.sprintf "images/%d.png" batches_done));
          train_batch (i + 1) (batches_done + 1)
    in
    let batches_done = train_batch 0 batches_done in
    train (epoch + 1) batches_done