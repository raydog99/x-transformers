open Torch

type beta_vae = {
  mutable num_iter : int;
  latent_dim : int;
  beta : int;
  gamma : float;
  loss_type : string;
  c_max : Tensor.t;
  c_stop_iter : float;
  encoder : Torch.nn.layer Sequential.t;
  fc_mu : Torch.nn.linear;
  fc_var : Torch.nn.linear;
  decoder_input : Torch.nn.linear;
  decoder : Torch.nn.layer Sequential.t;
  final_layer : Torch.nn.layer Sequential.t;
}

let create_beta_vae
    ?(hidden_dims=[|32; 64; 128; 256; 512|])
    ~in_channels
    ~latent_dim
    ?(beta=4)
    ?(gamma=1000.)
    ?(max_capacity=25)
    ?(capacity_max_iter=1e5)
    ?(loss_type="B") () =

  let modules_encoder = List.map (fun h_dim ->
    Torch.nn.(
      Sequential [
        Conv2d.automatic in_channels h_dim ~kernel_size:(3,3) ~stride:(2,2) ~padding:(1,1);
        BatchNorm2d h_dim;
        LeakyReLU;
      ]
    )
  ) (Array.to_list hidden_dims) in

  let encoder = Torch.nn.Sequential.of_list modules_encoder in
  let fc_mu = Torch.nn.linear (hidden_dims.(Array.length hidden_dims - 1) * 4) latent_dim in
  let fc_var = Torch.nn.linear (hidden_dims.(Array.length hidden_dims - 1) * 4) latent_dim in

  let modules_decoder = List.mapi (fun i h_dim ->
    let prev_dim = if i = 0 then latent_dim else hidden_dims.(i - 1) in
    Torch.nn.(
      Sequential [
        ConvTranspose2d.automatic prev_dim h_dim ~kernel_size:(3,3) ~stride:(2,2) ~padding:(1,1) ~output_padding:(1,1);
        BatchNorm2d h_dim;
        LeakyReLU;
      ]
    )
  ) (Array.to_list hidden_dims) in

  let decoder = Torch.nn.Sequential.of_list modules_decoder in
  let final_layer =
    Torch.nn.(
      Sequential [
        ConvTranspose2d.automatic hidden_dims.(Array.length hidden_dims - 1) hidden_dims.(Array.length hidden_dims - 1)
          ~kernel_size:(3,3) ~stride:(2,2) ~padding:(1,1) ~output_padding:(1,1);
        BatchNorm2d hidden_dims.(Array.length hidden_dims - 1);
        LeakyReLU;
        Conv2d.automatic hidden_dims.(Array.length hidden_dims - 1) ~out_channels:3 ~kernel_size:(3,3) ~padding:(1,1);
        Tanh;
      ]
    )
  in

  {
    num_iter = 0;
    latent_dim;
    beta;
    gamma;
    loss_type;
    c_max = Torch.tensor_of_float [|(float max_capacity)|];
    c_stop_iter = capacity_max_iter;
    encoder;
    fc_mu;
    fc_var;
    decoder_input = Torch.nn.linear latent_dim (hidden_dims.(Array.length hidden_dims - 1) * 4);
    decoder;
    final_layer;
  }

let encode beta_vae input =
  let result = Torch.forward_sequence beta_vae.encoder input in
  let result = Tensor.flatten result ~start_dim:1 in
  let mu = Torch.forward_linear beta_vae.fc_mu result in
  let log_var = Torch.forward_linear beta_vae.fc_var result in
  [mu; log_var]

let decode beta_vae z =
  let result = Torch.forward_linear beta_vae.decoder_input z in
  let result = Tensor.view result ~size:[|-1; 512; 2; 2|] in
  let result = Torch.forward_sequence beta_vae.decoder result in
  let result = Torch.forward_sequence beta_vae.final_layer result in
  result

let reparameterize mu logvar =
  let std = Tensor.exp (Tensor.mul logvar 0.5) in
  let eps = Tensor.randn_like std in
  Tensor.mul eps std + mu

let forward beta_vae input =
  let mu, log_var = encode beta_vae input in
  let z = reparameterize mu log_var in
  [decode beta_vae z; input; mu; log_var]

let loss_function beta_vae args kwargs =
  beta_vae.num_iter <- beta_vae.num_iter + 1;
  let recons = List.hd args in
  let input = List.hd (List.tl args) in
  let mu = List.hd (List.tl (List.tl args)) in
  let log_var = List.hd (List.tl (List.tl (List.tl args))) in
  let kld_weight = Torch.get_item kwargs "M_N" |> Tensor.to_float in

  let recons_loss = Tensor.mse_loss recons input in

  let kld_loss =
    Tensor.mean (Tensor.mul (Tensor.sum (Tensor.add (Tensor.sub (Tensor.add log_var (Tensor.mul mu mu)) (Tensor.exp log_var)) 1.0)) (-0.5)) in

  let loss =
    if beta_vae.loss_type = "H" then
      Tensor.add recons_loss (Tensor.mul (Tensor.mul (Tensor.from_float beta_vae.beta) kld_weight) kld_loss)
    else if beta_vae.loss_type = "B" then
      let c = Torch.clamp (Tensor.div (Tensor.mul beta_vae.c_max (Tensor.div (Tensor.from_float beta_vae.num_iter) beta_vae.c_stop_iter)) beta_vae.c_max) 0.0 (Torch.get_item beta_vae.c_max 0) in
      Tensor.add recons_loss (Tensor.mul (Tensor.mul (Tensor.from_float beta_vae.gamma) kld_weight) (Tensor.abs (Tensor.sub kld_loss c)))
    else
      failwith "Undefined loss type."
  in
  (Dict [("loss", loss); ("Reconstruction_Loss", recons_loss); ("KLD", kld_loss)])

let sample beta_vae num_samples current_device kwargs =
  let z = Tensor.randn [|num_samples; beta_vae.latent_dim|] in
  let z = Tensor.to_device z current_device in
  let samples = decode beta_vae z in
  samples

let generate beta_vae x kwargs =
  forward beta_vae x |> List.hd