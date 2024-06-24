open Torch

type two_stage_vae =
  { encoder : Torch_nn.Sequential.t
  ; fc_mu : Torch_nn.Linear.t
  ; fc_var : Torch_nn.Linear.t
  ; decoder_input : Torch_nn.Linear.t
  ; decoder : Torch_nn.Sequential.t
  ; final_layer : Torch_nn.Sequential.t
  ; latent_dim : int
  ; encoder2 : Torch_nn.Sequential.t
  ; fc_mu2 : Torch_nn.Linear.t
  ; fc_var2 : Torch_nn.Linear.t
  ; decoder2 : Torch_nn.Sequential.t
  }

let create_two_stage_vae in_channels latent_dim ?(hidden_dims=[|32; 64; 128; 256; 512|]) ?(hidden_dims2=[|1024; 1024|]) () =
  let modules_encoder =
    List.fold_left (fun acc h_dim ->
        let conv_layer =
          Torch_nn.(
            sequential
              [ conv2d ~stride:2 ~padding:1 in_channels h_dim ~kernel_size:(3, 3)
              ; batch_norm2d h_dim
              ; leaky_relu ()
              ])
        in
        let () = in_channels := h_dim in
        acc @ [conv_layer]
      ) [] hidden_dims
  in
  let encoder = Torch_nn.sequential modules_encoder in
  let fc_mu = Torch_nn.linear (List.hd (List.rev hidden_dims) * 4) latent_dim in
  let fc_var = Torch_nn.linear (List.hd (List.rev hidden_dims) * 4) latent_dim in

  let modules_decoder =
    List.fold_left (fun acc h_dim ->
        let conv_transpose_layer =
          Torch_nn.(
            sequential
              [ conv_transpose2d ~stride:2 ~padding:1 ~output_padding:1 h_dim (List.hd acc) ~kernel_size:(3, 3)
              ; batch_norm2d (List.hd acc)
              ; leaky_relu ()
              ])
        in
        acc @ [conv_transpose_layer]
      ) [List.hd hidden_dims * 4] (List.tl (List.rev hidden_dims))
  in
  let decoder = Torch_nn.sequential modules_decoder in

  let final_layer =
    Torch_nn.(
      sequential
        [ conv_transpose2d ~stride:2 ~padding:1 ~output_padding:1 (List.hd hidden_dims) (List.hd hidden_dims) ~kernel_size:(3, 3)
        ; batch_norm2d (List.hd hidden_dims)
        ; leaky_relu ()
        ; conv2d (List.hd hidden_dims) 3 ~kernel_size:(3, 3) ~padding:1
        ; tanh ()
        ])
  in

  let encoder2 =
    List.fold_left (fun acc h_dim ->
        let linear_layer =
          Torch_nn.(
            sequential
              [ linear (List.hd acc) h_dim
              ; batch_norm1d h_dim
              ; leaky_relu ()
              ])
        in
        acc @ [linear_layer]
      ) [latent_dim] hidden_dims2
    |> Torch_nn.sequential
  in
  let fc_mu2 = Torch_nn.linear (List.hd (List.rev hidden_dims2)) latent_dim in
  let fc_var2 = Torch_nn.linear (List.hd (List.rev hidden_dims2)) latent_dim in

  let decoder2 =
    List.fold_left (fun acc h_dim ->
        let linear_layer =
          Torch_nn.(
            sequential
              [ linear (List.hd acc) h_dim
              ; batch_norm1d h_dim
              ; leaky_relu ()
              ])
        in
        acc @ [linear_layer]
      ) [latent_dim] (List.tl (List.rev hidden_dims2))
    |> Torch_nn.sequential
  in

  { encoder
  ; fc_mu
  ; fc_var
  ; decoder_input = Torch_nn.linear latent_dim (List.hd hidden_dims * 4)
  ; decoder
  ; final_layer
  ; latent_dim
  ; encoder2
  ; fc_mu2
  ; fc_var2
  ; decoder2
  }

let encode model input =
  let result = Torch_forward.model_forward model.encoder [|input|] in
  let result = Torch_tensor.flatten result ~start_dim:1 in
  let mu = Torch_nn.forward model.fc_mu [|result|] in
  let log_var = Torch_nn.forward model.fc_var [|result|] in
  [|mu; log_var|]

let decode model z =
  let result = Torch_nn.forward model.decoder_input [|z|] in
  let result = Torch_tensor.view result ~size:[|-1; 512; 2; 2|] in
  let result = Torch_forward.model_forward model.decoder [|result|] in
  let result = Torch_forward.model_forward model.final_layer [|result|] in
  result

let reparameterize mu logvar =
  let std = Torch_tensor.exp (Torch_tensor.mul_scalar logvar 0.5) in
  let eps = Torch_tensor.randn_like std in
  Torch_tensor.(eps * std + mu)

let forward model input =
  let mu, log_var = encode model input in
  let z = reparameterize mu log_var in
  [|decode model z; input; mu; log_var|]

let loss_function model recons input mu log_var ~kld_weight =
  let recons_loss = Torch_nn.functional.mse_loss recons input in
  let kld_loss =
    Torch_tensor.(
      mean (neg (mul_scalar (sum (add (sub (add log_var (pow_scalar mu 2.) (exp log_var)) 1.) 0.5) 0.5)) 1) 0)
  in
  let loss = Torch_tensor.add recons_loss (Torch_tensor.mul_scalar kld_loss kld_weight) in
  [|loss; recons_loss; neg kld_loss|]

let sample model num_samples current_device =
  let z = Torch_tensor.randn [|num_samples; model.latent_dim|] |> Torch_tensor.to_device current_device in
  decode model z