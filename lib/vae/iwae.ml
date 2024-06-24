open Torch

type iwae = {
  latent_dim : int;
  num_samples : int;
  encoder : Torch.nn.layer Sequential.t;
  fc_mu : Torch.nn.linear;
  fc_var : Torch.nn.linear;
  decoder_input : Torch.nn.linear;
  decoder : Torch.nn.layer Sequential.t;
  final_layer : Torch.nn.layer Sequential.t;
}

let create_iwae
    ?(hidden_dims=[|32; 64; 128; 256; 512|])
    ~in_channels
    ~latent_dim
    ~num_samples
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
    num_samples;
    encoder;
    fc_mu;
    fc_var;
    decoder_input = Torch.nn.linear latent_dim (hidden_dims.(Array.length hidden_dims - 1) * 4);
    decoder;
    final_layer;
  }

let encode iwae input =
  let result = Torch.forward_sequence iwae.encoder input in
  let result = Tensor.flatten result ~start_dim:1 in
  let mu = Torch.forward_linear iwae.fc_mu result in
  let log_var = Torch.forward_linear iwae.fc_var result in
  [mu; log_var]

let decode iwae z =
  let B = Tensor.size z.(0) in
  let D = Tensor.size z.(2) in
  let result = Tensor.view z ~size:[|B * iwae.num_samples; D|] in
  let result = Torch.forward_linear iwae.decoder_input result in
  let result = Tensor.view result ~size:[|-1; 512; 2; 2|] in
  let result = Torch.forward_sequence iwae.decoder result in
  let result = Torch.forward_sequence iwae.final_layer result in
  let result = Tensor.view result ~size:[|B; iwae.num_samples; result.size(1); result.size(2); result.size(3)|] in
  result

let reparameterize mu logvar =
  let std = Tensor.exp (Tensor.mul logvar 0.5) in
  let eps = Tensor.randn_like std in
  Tensor.mul eps std + mu

let forward iwae input =
  let mu, log_var = encode iwae input in
  let mu = Tensor.repeat mu [|1; iwae.num_samples; 1|] |> Tensor.permute ~dims:[|1; 0; 2|] in
  let log_var = Tensor.repeat log_var [|1; iwae.num_samples; 1|] |> Tensor.permute ~dims:[|1; 0; 2|] in
  let z = reparameterize mu log_var in
  let eps = Tensor.div (Tensor.sub z mu) log_var in
  [decode iwae z; input; mu; log_var; z; eps]

let loss_function iwae args kwargs =
  let recons = args.(0) in
  let input = args.(1) in
  let mu = args.(2) in
  let log_var = args.(3) in
  let z = args.(4) in
  let eps = args.(5) in
  let input = Tensor.repeat input [|iwae.num_samples; 1; 1; 1; 1|] |> Tensor.permute ~dims:[|1; 0; 2; 3; 4|] in
  let kld_weight = kwargs.["M_N"] in
  let log_p_x_z = Tensor.pow (Tensor.sub recons input) 2. |> Tensor.flatten ~start_dim:2 |> Tensor.mean ~dtype:Float in
  let kld_loss = Tensor.mul (Tensor.sum (Tensor.add (Tensor.pow mu 2.) log_var) ~dtype:Float ~dims:[2]) (-0.5) in
  let log_weight = Tensor.add (Tensor.div log_p_x_z kld_weight) kld_loss in
  let weight = F.softmax log_weight ~dim:2 in
  let loss = Tensor.mean (Tensor.mul weight log_weight) ~dtype:Float ~dims:0 in
  {loss= loss; Reconstruction_Loss= Tensor.mean log_p_x_z ~dtype:Float; KLD= Tensor.neg kld_loss}

let sample iwae num_samples current_device kwargs =
  let z = Tensor.randn [num_samples; 1; iwae.latent_dim] in
  let z = z.to current_device in
  decode iwae z |> Tensor.squeeze