open Torch

module VDVAE = struct
  type t = {
    encoder: Layer.t;
    mu_layer: Layer.t;
    logvar_layer: Layer.t;
    decoder: Layer.t;
    latent_dim: int;
    vs: Var_store.t;
  }

  let create_encoder vs input_dim hidden_dims =
    let layers = List.fold_left (fun (prev_dim, acc) dim ->
      (dim, Layer.sequential [
        Layer.linear vs ~input_dim:prev_dim ~output_dim:dim;
        Layer.relu;
        Layer.batch_norm1d vs dim;
      ] :: acc)
    ) (input_dim, []) hidden_dims in
    Layer.sequential (List.rev (snd layers))

  let create_decoder vs latent_dim hidden_dims output_dim =
    let layers = List.fold_left (fun (prev_dim, acc) dim ->
      (dim, Layer.sequential [
        Layer.linear vs ~input_dim:prev_dim ~output_dim:dim;
        Layer.relu;
        Layer.batch_norm1d vs dim;
      ] :: acc)
    ) (latent_dim, []) (List.rev hidden_dims) in
    Layer.sequential (
      List.rev (snd layers) @ [
        Layer.linear vs ~input_dim:(fst layers) ~output_dim;
        Layer.sigmoid;
      ]
    )

  let create ~input_dim ~latent_dim ~hidden_dims =
    let vs = Var_store.create ~name:"vdvae" () in
    let encoder = create_encoder vs input_dim hidden_dims in
    let last_hidden = List.hd (List.rev hidden_dims) in
    let mu_layer = Layer.linear vs ~input_dim:last_hidden ~output_dim:latent_dim in
    let logvar_layer = Layer.linear vs ~input_dim:last_hidden ~output_dim:latent_dim in
    let decoder = create_decoder vs latent_dim hidden_dims input_dim in
    { encoder; mu_layer; logvar_layer; decoder; latent_dim; vs }

  let reparameterize mu logvar =
    let std = Tensor.exp (Tensor.mul logvar (Tensor.f 0.5)) in
    let eps = Tensor.randn_like std in
    Tensor.(mu + (eps * std))

  let forward t x =
    let h = Layer.forward t.encoder x in
    let mu = Layer.forward t.mu_layer h in
    let logvar = Layer.forward t.logvar_layer h in
    let z = reparameterize mu logvar in
    let recon = Layer.forward t.decoder z in
    (recon, mu, logvar)

  let loss t x recon mu logvar =
    let recon_loss = Tensor.binary_cross_entropy recon x ~reduction:Reduction.Sum in
    let kl_div = Tensor.(
      sum (exp logvar + square mu - logvar - ones_like mu)
      |> mul_scalar (-0.5)
    ) in
    Tensor.(recon_loss + kl_div)

  let train t ~learning_rate ~num_epochs ~batch_size data =
    let optimizer = Optimizer.adam (Var_store.all_vars t.vs) ~learning_rate in
    for epoch = 1 to num_epochs do
      let total_loss = ref 0. in
      List.iter (fun batch ->
        let recon, mu, logvar = forward t batch in
        let loss = loss t batch recon mu logvar in
        total_loss := !total_loss +. (Tensor.float_value loss);
        Optimizer.backward_step optimizer ~loss
      ) data;
      Printf.printf "Epoch %d, Loss: %f\n" epoch (!total_loss /. float (List.length data))
    done

  let reconstruct t x =
    let recon, _, _ = forward t x in
    recon

  let generate t num_samples =
    let z = Tensor.randn [num_samples; t.latent_dim] in
    Layer.forward t.decoder z
end