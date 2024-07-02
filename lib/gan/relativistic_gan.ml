open Torch

let img_size = 32
let latent_dim = 100
let channels = 1
let batch_size = 64
let n_epochs = 200
let lr = 0.0002
let b1 = 0.5
let b2 = 0.999
let sample_interval = 400
let rel_avg_gan = false

module Generator = struct
  let create () =
    let init_size = img_size / 4 in
    let l1 = Nn.sequential
      [Nn.linear ~in_dim:latent_dim ~out_dim:(128 * init_size * init_size)] in
    let conv_blocks = Nn.sequential
      [Nn.batch_norm2d 128;
       Nn.upsample ~scale_factor:2.0 ();
       Nn.conv2d ~ksize:3 ~stride:1 ~padding:1 ~input_dim:128 ~output_dim:128;
       Nn.batch_norm2d 128;
       Nn.leaky_relu ~negative_slope:0.2 ();
       Nn.upsample ~scale_factor:2.0 ();
       Nn.conv2d ~ksize:3 ~stride:1 ~padding:1 ~input_dim:128 ~output_dim:64;
       Nn.batch_norm2d 64;
       Nn.leaky_relu ~negative_slope:0.2 ();
       Nn.conv2d ~ksize:3 ~stride:1 ~padding:1 ~input_dim:64 ~output_dim:channels;
       Nn.tanh ()] in
    fun xs ->
      let out = Nn.apply l1 xs in
      let out = Tensor.view out [|-1; 128; init_size; init_size|] in
      Nn.apply conv_blocks out

  let model = create ()
end

module Discriminator = struct
  let discriminator_block in_filters out_filters bn =
    let block = [
      Nn.conv2d ~ksize:3 ~stride:2 ~padding:1 ~input_dim:in_filters ~output_dim:out_filters;
      Nn.leaky_relu ~negative_slope:0.2 ();
      Nn.dropout ~p:0.25 ()
    ] in
    if bn then Nn.batch_norm2d out_filters :: block else block

  let create () =
    let model = Nn.sequential (
      discriminator_block channels 16 false @
      discriminator_block 16 32 true @
      discriminator_block 32 64 true @
      discriminator_block 64 128 true
    ) in
    let ds_size = img_size / (2 * 2 * 2 * 2) in
    let adv_layer = Nn.linear ~in_dim:(128 * ds_size * ds_size) ~out_dim:1 in
    fun xs ->
      let out = Nn.apply model xs in
      let out = Tensor.view out [|-1; 128 * ds_size * ds_size|] in
      Nn.apply adv_layer out

  let model = create ()
end

let adversarial_loss = Nn.binary_cross_entropy_with_logits

let optimizer_g = Optimizer.adam (Generator.model) ~learning_rate:lr ~beta1:b1 ~beta2:b2
let optimizer_d = Optimizer.adam (Discriminator.model) ~learning_rate:lr ~beta1:b1 ~beta2:b2

let dataloader = 
  Mnist_helper.read_files ()
  |> Mnist_helper.batched_samples ~batch_size
  |> Seq.map (fun (images, _) -> 
       Tensor.(images 
         |> to_type ~type_:Float
         |> view ~size:[-1; channels; img_size; img_size]
         |> div_scalar (Scalar.f 127.5) 
         |> sub_scalar (Scalar.f 1.)))

let train () =
  for epoch = 1 to n_epochs do
    Seq.iter (fun real_imgs ->
      let batch_size = Tensor.shape real_imgs |> Array.get 0 in
      let z = Tensor.randn [|batch_size; latent_dim|] in
      
      (* Train Generator *)
      Optimizer.zero_grad optimizer_g;
      let gen_imgs = Generator.model z in
      let real_pred = Discriminator.model real_imgs |> Tensor.detach in
      let fake_pred = Discriminator.model gen_imgs in
      
      let g_loss = 
        if rel_avg_gan then
          adversarial_loss (Tensor.sub fake_pred (Tensor.mean real_pred ~dim:[|0|] ~keepdim:true))
            (Tensor.ones_like fake_pred)
        else
          adversarial_loss (Tensor.sub fake_pred real_pred) (Tensor.ones_like fake_pred)
      in
      
      Tensor.backward g_loss;
      Optimizer.step optimizer_g;
      
      (* Train Discriminator *)
      Optimizer.zero_grad optimizer_d;
      let real_pred = Discriminator.model real_imgs in
      let fake_pred = Discriminator.model (Tensor.detach gen_imgs) in
      
      let (real_loss, fake_loss) = 
        if rel_avg_gan then
          (adversarial_loss (Tensor.sub real_pred (Tensor.mean fake_pred ~dim:[|0|] ~keepdim:true))
             (Tensor.ones_like real_pred),
           adversarial_loss (Tensor.sub fake_pred (Tensor.mean real_pred ~dim:[|0|] ~keepdim:true))
             (Tensor.zeros_like fake_pred))
        else
          (adversarial_loss (Tensor.sub real_pred fake_pred) (Tensor.ones_like real_pred),
           adversarial_loss (Tensor.sub fake_pred real_pred) (Tensor.zeros_like fake_pred))
      in
      
      let d_loss = Tensor.div (Tensor.add real_loss fake_loss) (Scalar.f 2.) in
      Tensor.backward d_loss;
      Optimizer.step optimizer_d;
      
      Printf.printf "[Epoch %d/%d] [D loss: %f] [G loss: %f]\n"
        epoch n_epochs
        (Tensor.float_value d_loss)
        (Tensor.float_value g_loss);
      
    ) dataloader
  done