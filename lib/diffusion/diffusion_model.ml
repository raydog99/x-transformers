open Torch

module DiffusionModel = struct
  type t = {
    spatial_width : int;
    n_colors : int;
    trajectory_length : int;
    n_temporal_basis : int;
    n_hidden_dense_lower : int;
    n_hidden_dense_lower_output : int;
    n_hidden_dense_upper : int;
    n_hidden_conv : int;
    n_layers_conv : int;
    n_layers_dense_lower : int;
    n_layers_dense_upper : int;
    n_t_per_minibatch : int;
    n_scales : int;
    step1_beta : float;
    uniform_noise : float;
    mlp : regression.MLP_conv_dense.t;
    temporal_basis : Tensor.t;
    beta_arr : Tensor.t;
  }

  let generate_beta_arr dm step1_beta =
    let trajectory_length = dm.trajectory_length in
    let min_beta_val = 1e-6 in
    let min_beta_values = Tensor.ones1 [|trajectory_length|] +. step1_beta in
    let min_beta = Tensor.of_float1 min_beta_values in
    let beta_perturb_coefficients = Tensor.zeros1 [|dm.n_temporal_basis|] in
    let beta_perturb = Tensor.dot dm.temporal_basis beta_perturb_coefficients in
    let beta_baseline = Tensor.linspace 1.0 2.0 trajectory_length in
    let beta_baseline_offset = Util.logit_np beta_baseline |> Tensor.of_float1 in
    let beta_arr = Tensor.nnet_sigmoid (beta_perturb + beta_baseline_offset) in
    let beta_arr = min_beta + (beta_arr * (1.0 - min_beta - 1e-5)) in
    Tensor.reshape beta_arr [|trajectory_length; 1|]

  let get_t_weights dm t =
    let n_seg = dm.trajectory_length in
    let t_compare = Tensor.arange ~options:[|`Device (Tensor.device t)|] Float 0 n_seg 1 in
    let diff = Tensor.abs (Tensor.unsqueeze t 1 - t_compare) in
    let t_weights = Tensor.max2 (Tensor.join 1 [|Tensor.(0.0 - diff + 1.0); Tensor.zeros2 [|n_seg; 1|]|]) ~dim:1 in
    Tensor.reshape t_weights [-1; 1]

  let get_beta_forward dm t =
    let t_weights = get_t_weights dm t in
    Tensor.dot t_weights dm.beta_arr

  let get_mu_sigma dm x_noisy t =
    let z = Regression.MLP_conv_dense.apply dm.mlp x_noisy in
    let mu_coeff, beta_coeff = temporal_readout dm z t in
    let beta_forward = get_beta_forward dm t in
    let beta_coeff_scaled = Tensor.div beta_coeff (Tensor.sqrt (Tensor.of_int dm.trajectory_length)) in
    let beta_reverse = Tensor.nnet_sigmoid (beta_coeff_scaled + Util.logit beta_forward) in
    let mu = Tensor.((x_noisy * sqrt (1.0 - beta_forward)) + (mu_coeff * sqrt beta_forward)) in
    let sigma = Tensor.sqrt beta_reverse in
    mu, sigma

  let generate_forward_diffusion_sample dm x_noiseless =
    let x_noiseless = Tensor.reshape x_noiseless [| -1; dm.n_colors; dm.spatial_width; dm.spatial_width |] in
    let n_images = Tensor.size x_noiseless.(0) in
    let t = Tensor.floor (Tensor.randint [1] 1 dm.trajectory_length) in
    let t_weights = get_t_weights dm t in
    let n = Tensor.randn_like x_noiseless in
    let beta_forward = get_beta_forward dm t in
    let alpha_forward = Tensor.(1.0 - beta_forward) in
    let alpha_arr = Tensor.(1.0 - dm.beta_arr) in
    let alpha_cum_forward_arr = Tensor.cumprod alpha_arr ~dim:0 in
    let alpha_cum_forward = Tensor.dot t_weights alpha_cum_forward_arr in
    let beta_cumulative = Tensor.(1.0 - alpha_cum_forward) in
    let beta_cumulative_prior_step = Tensor.(1.0 - (alpha_cum_forward / alpha_forward)) in
    let x_uniformnoise = Tensor.(x_noiseless + (rand [n_images; dm.n_colors; dm.spatial_width; dm.spatial_width] - 0.5) * dm.uniform_noise) in
    let x_noisy = Tensor.(x_uniformnoise * sqrt alpha_cum_forward + n * sqrt (1.0 - alpha_cum_forward)) in
    let mu1_scl = Tensor.sqrt (alpha_cum_forward / alpha_forward) in
    let mu2_scl = Tensor.inv_sqrt alpha_forward in
    let cov1 = Tensor.(1.0 - (alpha_cum_forward / alpha_forward)) in
    let cov2 = Tensor.(beta_forward / alpha_forward) in
    let lam = Tensor.inv cov1 +. Tensor.inv cov2 in
    let mu = Tensor.((x_uniformnoise * mu1_scl / cov1) + (x_noisy * mu2_scl / cov2)) / lam in
    let sigma = Tensor.sqrt (1.0 /. lam) in
    mu, sigma, x_noisy, t

  let get_beta_full_trajectory dm =
    let alpha_arr = Tensor.(1.0 - dm.beta_arr) in
    let beta_full_trajectory = Tensor.(1.0 - exp (Tensor.sum (Tensor.log alpha_arr))) in
    beta_full_trajectory

  let get_negL_bound dm mu sigma mu_posterior sigma_posterior =
    let kl = Tensor.(
      log sigma - log sigma_posterior +
      ((sigma_posterior ** 2) + (mu_posterior - mu) ** 2) / (2.0 * sigma ** 2) - 0.5
    ) in
    let h_startpoint = Tensor.(0.5 * (1.0 + log (2.0 * pi)) + 0.5 * log dm.beta_arr.(0)) in
    let h_endpoint = Tensor.(0.5 * (1.0 + log (2.0 * pi)) + 0.5 * log (get_beta_full_trajectory dm)) in
    let h_prior = Tensor.(0.5 * (1.0 + log (2.0 * pi)) + 0.5 * log 1.0) in
    let negL_bound = Tensor.((kl * float dm.trajectory_length) + h_startpoint - h_endpoint + h_prior) in
    let negL_gauss = Tensor.(0.5 * (1.0 + log (2.0 * pi)) + 0.5 * log 1.0) in
    let negL_diff = Tensor.(negL_bound - negL_gauss) in
    let l_diff_bits = Tensor.(negL_diff / log 2.0) in
    Tensor.mean l_diff_bits ~dim:0 ~dtype:Dtype.Float64

  let cost_single_t dm x_noiseless =
    let mu_posterior, sigma_posterior, x_noisy, t = generate_forward_diffusion_sample dm x_noiseless in
    let mu, sigma = get_mu_sigma dm x_noisy t in
    get_negL_bound dm mu sigma mu_posterior sigma_posterior

  let internal_state dm x_noiseless =
    let mu_posterior, sigma_posterior, x_noisy, t = generate_forward_diffusion_sample dm x_noiseless in
    let mu, sigma = get_mu_sigma dm x_noisy t in
    let mu_diff = Tensor.(mu - mu_posterior) in
    let logratio = Tensor.log (Tensor.(sigma / sigma_posterior)) in
    [mu_diff; logratio; mu; sigma; mu_posterior; sigma_posterior; x_noiseless; x_noisy]

  let cost dm x_noiseless =
    let cost = ref 0.0 in
    for _ = 1 to dm.n_t_per_minibatch do
      cost := !cost +. cost_single_t dm x_noiseless |> Tensor.to_float1
    done;
    Tensor.of_float1 [|!cost /. float dm.n_t_per_minibatch|]

  let temporal_readout dm z t =
    let n_images = Tensor.size z.(0) in
    let t_weights = get_t_weights dm t in
    let z = Tensor.reshape z [|n_images; dm.spatial_width; dm.spatial_width; dm.n_colors; 2; dm.n_temporal_basis|] in
    let coeff_weights = Tensor.dot dm.temporal_basis t_weights in
    let mu_coeff = Tensor.(z.(...,:,0) |> Tensor.dimshuffle [0; 3; 1; 2]) in
    let beta_coeff = Tensor.(z.(...,:,1) |> Tensor.dimshuffle [0; 3; 1; 2]) in
    mu_coeff, beta_coeff
end