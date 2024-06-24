open Torch

type info_vae = {
  latent_dim : int;
  reg_weight : int;
  kernel_type : string;
  z_var : float;
  alpha : float;
  beta : float;
  encoder : Torch.nn.layer Sequential.t;
  fc_mu : Torch.nn.linear;
  fc_var : Torch.nn.linear;
  decoder_input : Torch.nn.linear;
  decoder : Torch.nn.layer Sequential.t;
  final_layer : Torch.nn.layer Sequential.t;
}

let create_info_vae
    ?(hidden_dims=[|32; 64; 128; 256; 512|])
    ~in_channels
    ~latent_dim
    ?(alpha=(-0.5))
    ?(beta=5.0)
    ?(reg_weight=100)
    ?(kernel_type="imq")
    ?(latent_var=2.0)
    () =

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
    latent_dim;
    reg_weight;
    kernel_type;
    z_var = latent_var;
    alpha;
    beta;
    encoder;
    fc_mu;
    fc_var;
    decoder_input = Torch.nn.linear latent_dim (hidden_dims.(Array.length hidden_dims - 1) * 4);
    decoder;
    final_layer;
  }

let encode info_vae input =
  let result = Torch.forward_sequence info_vae.encoder input in
  let result = Tensor.flatten result ~start_dim:1 in
  let mu = Torch.forward_linear info_vae.fc_mu result in
  let log_var = Torch.forward_linear info_vae.fc_var result in
  [mu; log_var]

let decode info_vae z =
  let result = Torch.forward_linear info_vae.decoder_input z in
  let result = Tensor.view result ~size:[|-1; 512; 2; 2|] in
  let result = Torch.forward_sequence info_vae.decoder result in
  let result = Torch.forward_sequence info_vae.final_layer result in
  result

let reparameterize mu logvar =
  let std = Tensor.exp (Tensor.mul logvar 0.5) in
  let eps = Tensor.randn_like std in
  Tensor.mul eps std + mu

let compute_kernel info_vae x1 x2 =
  let d = Tensor.size x1.(0) in
  let n = Tensor.size x1.(0)