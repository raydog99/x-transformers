open Torch

let () =
  let parser = Argparse.ArgumentParser.create () in
  Argparse.ArgumentParser.add_argument parser ~name:"--n_epochs" ~typ:Argparse.Int ~default:200
    ~help:"number of epochs of training";
  Argparse.ArgumentParser.add_argument parser ~name:"--batch_size" ~typ:Argparse.Int ~default:64
    ~help:"size of the batches";
  Argparse.ArgumentParser.add_argument parser ~name:"--lr" ~typ:Argparse.Float ~default:0.0002
    ~help:"adam: learning rate";
  Argparse.ArgumentParser.add_argument parser ~name:"--b1" ~typ:Argparse.Float ~default:0.5
    ~help:"adam: decay of first order momentum of gradient";
  Argparse.ArgumentParser.add_argument parser ~name:"--b2" ~typ:Argparse.Float ~default:0.999
    ~help:"adam: decay of first order momentum of gradient";
  Argparse.ArgumentParser.add_argument parser ~name:"--n_cpu" ~typ:Argparse.Int ~default:8
    ~help:"number of cpu threads to use during batch generation";
  Argparse.ArgumentParser.add_argument parser ~name:"--latent_dim" ~typ:Argparse.Int ~default:100
    ~help:"dimensionality of the latent space";
  Argparse.ArgumentParser.add_argument parser ~name:"--img_size" ~typ:Argparse.Int ~default:28
    ~help:"size of each image dimension";
  Argparse.ArgumentParser.add_argument parser ~name:"--channels" ~typ:Argparse.Int ~default:1
    ~help:"number of image channels";
  Argparse.ArgumentParser.add_argument parser ~name:"--sample_interval" ~typ:Argparse.Int ~default:400
    ~help:"interval betwen image samples";
  let opt = Argparse.ArgumentParser.parse_args parser in
  print_endline (Argparse.ArgumentParser.string_of_t opt);

  let img_shape = [| opt.channels; opt.img_size; opt.img_size |] in

  let cuda = Cuda.is_available () in

  (* Generator module *)
  let generator =
    let block in_feat out_feat normalize =
      let layers = [ Layer.linear in_feat out_feat ] in
      if normalize then
        layers |> List.append [ Layer.batch_norm1d out_feat ~affine:true ~track_running_stats:true 0.8 ];
      layers |> List.append [ Layer.leaky_relu ~negative_slope:0.2 ] |> Layer.sequential
    in
    let model =
      [
        block opt.latent_dim 128 false;
        block 128 256 true;
        block 256 512 true;
        block 512 1024 true;
        Layer.linear 1024 (Tensor.size1d_exn (Tensor.of_int (Array.fold_left ( * ) 1 img_shape)));
        Layer.tanh;
      ]
      |> Layer.sequential
    in
    Layer.forward model;
  in

  (* Discriminator module *)
  let discriminator =
    let model =
      Layer.(
        [
          Layer.linear (Tensor.size1d_exn (Tensor.of_int (Array.fold_left ( * ) 1 img_shape))) 512;
          Layer.leaky_relu ~negative_slope:0.2;
          Layer.linear 512 256;
          Layer.leaky_relu ~negative_slope:0.2;
          Layer.linear 256 1;
          Layer.sigmoid;
        ]
        |> Layer.sequential)
    in
    Layer.forward model;
  in

  (* Loss function *)
  let adversarial_loss = Loss.binary_cross_entropy () in

  (* Initialize generator and discriminator *)
  let generator_params = Generator.parameters generator in
  let generator_optimizer = Optimizer.adam generator_params ~learning_rate:opt.lr ~beta1:opt.b1 ~beta2:opt.b2 in
  let discriminator_params = Discriminator.parameters discriminator in
  let discriminator_optimizer =
    Optimizer.adam discriminator_params ~learning_rate:opt.lr ~beta1:opt.b1 ~beta2:opt.b2
  in

  let tensor_type = if cuda then Tensor.cuda FloatType else Tensor.float32 in

  (* Configure data loader *)
  let transform =
    Transform.compose
      [
        Transform.resize [ opt.img_size; opt.img_size ];
        Transform.to_tensor;
        Transform.normalize ~mean:[| 0.5 |] ~std:[| 0.5 |];
      ]
  in
  let dataset = Datasets.mnist ~root:"../data/mnist" ~train:true ~transform () in
  let dataloader =
    Dataloader.make ~dataset ~batch_size:opt.batch_size ~shuffle:true ~num_workers:opt.n_cpu ()
  in

  (* Training loop *)
  for epoch = 0 to opt.n_epochs - 1 do
    Dataloader.iter dataloader ~f:(fun batch ->
        let imgs = Tensor.requires_grad batch#data ~requires_grad:false in

        (* Adversarial ground truths *)
        let valid = Tensor.(ones [ size0 imgs; 1 ] |> fill_ ~value:1.0) in
        let fake = Tensor.(ones [ size0 imgs; 1 ] |> fill_ ~value:0.0) in

        (* Train Generator *)
        Optimizer.zero_grad generator_optimizer;
        let z = Tensor.(randn [ size0 imgs; opt.latent_dim ] |> to_type tensor_type) in
        let gen_imgs = Generator.forward generator z in
        let g_loss = Tensor.(adversarial_loss (Discriminator.forward discriminator gen_imgs) valid) in
        Tensor.backward g_loss;
        Optimizer.step generator_optimizer;

        (* Train Discriminator *)
        Optimizer.zero_grad discriminator_optimizer;
        let real_loss = Tensor.(adversarial_loss (Discriminator.forward discriminator imgs) valid) in
        let gen_imgs = Tensor.detach gen_imgs in
        let fake_loss = Tensor.(adversarial_loss (Discriminator.forward discriminator gen_imgs) fake) in
        let d_loss = Tensor.((real_loss + fake_loss) / F 2.0) in
        Tensor.backward d_loss;
        Optimizer.step discriminator_optimizer;

        Printf.printf "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n%!" epoch opt.n_epochs
          (Batch.index batch) (Dataloader.length dataloader) (Tensor.float_value d_loss)
          (Tensor.float_value g_loss));

    let batches_done = epoch * (Dataloader.length dataloader) in
    if batches_done % opt.sample_interval = 0 then (
      let sample_noise = Tensor.(randn [ 25; opt.latent_dim ] |> to_type tensor_type) in
      let samples = Generator.forward generator sample_noise in
      let filename = Printf.sprintf "images/%d.png" batches_done in
      Torch_vision.Utils.save_image ~filename samples ~nrow:5 ~normalize:true)
  done