open Torch

module type Opt = sig
  val n_epochs : int
  val batch_size : int
  val lr : float
  val b1 : float
  val b2 : float
  val latent_dim : int
  val n_classes : int
  val img_size : int
  val channels : int
  val sample_interval : int
end

module Make (O : Opt) = struct
  let img_shape = (O.channels, O.img_size, O.img_size)

  module Generator = struct
    type t = {
      label_emb : nn;
      model : nn;
    }

    let create () =
      let label_emb = Layer.embedding O.n_classes O.n_classes in
      let block in_feat out_feat normalize =
        let layers = [Layer.linear in_feat out_feat] in
        let layers = if normalize then
          Layer.batch_norm1d out_feat 0.8 :: layers
        else
          layers
        in
        Layer.leaky_relu 0.2 :: layers
      in
      let model = Layer.sequential [
        Layer.sequential (block (O.latent_dim + O.n_classes) 128 false);
        Layer.sequential (block 128 256 true);
        Layer.sequential (block 256 512 true);
        Layer.sequential (block 512 1024 true);
        Layer.linear 1024 (O.channels * O.img_size * O.img_size);
        Layer.tanh;
      ] in
      { label_emb; model }

    let forward t noise labels =
      let label_emb = Layer.apply t.label_emb labels in
      let gen_input = Tensor.cat [label_emb; noise] ~dim:1 in
      let img = Layer.apply t.model gen_input in
      Tensor.view img ~size:[ -1; O.channels; O.img_size; O.img_size ]
  end

  module Discriminator = struct
    type t = {
      label_embedding : nn;
      model : nn;
    }

    let create () =
      let label_embedding = Layer.embedding O.n_classes O.n_classes in
      let model = Layer.sequential [
        Layer.linear (O.n_classes + (O.channels * O.img_size * O.img_size)) 512;
        Layer.leaky_relu 0.2;
        Layer.linear 512 512;
        Layer.dropout 0.4;
        Layer.leaky_relu 0.2;
        Layer.linear 512 512;
        Layer.dropout 0.4;
        Layer.leaky_relu 0.2;
        Layer.linear 512 1;
      ] in
      { label_embedding; model }

    let forward t img labels =
      let img_flat = Tensor.view img ~size:[ Tensor.shape img |> List.hd; -1 ] in
      let label_emb = Layer.apply t.label_embedding labels in
      let d_in = Tensor.cat [img_flat; label_emb] ~dim:1 in
      Layer.apply t.model d_in
  end

  let adversarial_loss = Loss.mse

  let train () =
    let generator = Generator.create () in
    let discriminator = Discriminator.create () in
    let optim_g = Optimizer.adam (Var_store.all_vars generator) ~lr:O.lr ~betas:(O.b1, O.b2) in
    let optim_d = Optimizer.adam (Var_store.all_vars discriminator) ~lr:O.lr ~betas:(O.b1, O.b2) in
    
    let dataset = Dataset.mnist ~train:true () in
    let dataloader = Dataset.to_dataloader dataset ~batch_size:O.batch_size ~shuffle:true in

    for epoch = 1 to O.n_epochs do
      Dataset.iter dataloader ~f:(fun batch ->
        let imgs = Tensor.to_device ~device:Device.cuda (fst batch) in
        let labels = Tensor.to_device ~device:Device.cuda (snd batch) in
        let batch_size = Tensor.shape imgs |> List.hd in
        
        let valid = Tensor.ones [batch_size; 1] ~device:Device.cuda in
        let fake = Tensor.zeros [batch_size; 1] ~device:Device.cuda in

        (* Train Generator *)
        Optimizer.zero_grad optim_g;
        let z = Tensor.randn [batch_size; O.latent_dim] ~device:Device.cuda in
        let gen_labels = Tensor.randint O.n_classes ~low:0 ~size:[batch_size] ~device:Device.cuda in
        let gen_imgs = Generator.forward generator z gen_labels in
        let validity = Discriminator.forward discriminator gen_imgs gen_labels in
        let g_loss = Loss.mse validity valid in
        Tensor.backward g_loss;
        Optimizer.step optim_g;

        (* Train Discriminator *)
        Optimizer.zero_grad optim_d;
        let validity_real = Discriminator.forward discriminator imgs labels in
        let d_real_loss = Loss.mse validity_real valid in
        let validity_fake = Discriminator.forward discriminator gen_imgs gen_labels in
        let d_fake_loss = Loss.mse validity_fake fake in
        let d_loss = Tensor.add (Tensor.div_scalar d_real_loss (Scalar.float 2.)) (Tensor.div_scalar d_fake_loss (Scalar.float 2.)) in
        Tensor.backward d_loss;
        Optimizer.step optim_d;

        Printf.printf "[Epoch %d/%d] [D loss: %f] [G loss: %f]\n"
          epoch O.n_epochs
          (Tensor.to_float0_exn d_loss)
          (Tensor.to_float0_exn g_loss);
      )
    done

  let sample_image n_row batches_done =
    let z = Tensor.randn [n_row * n_row; O.latent_dim] ~device:Device.cuda in
    let labels = Tensor.of_int1 [| Array.init (n_row * n_row) (fun i -> i mod n_row) |] in
    let gen_imgs = Generator.forward generator z labels in
    Torch_vision.Image.write_image gen_imgs ~filename:(Printf.sprintf "images/%d.png" batches_done)
end