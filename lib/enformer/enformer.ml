open Torch

let maybeSyncBatchnorm ?(is_distributed : bool option) () =
  let is_distributed = match is_distributed with
    | Some is_dist -> is_dist
    | None -> Dist.is_initialized () && Dist.get_world_size () > 1
  in
  if is_distributed then nn.SyncBatchNorm else nn.BatchNorm1d

let poisson_loss pred target =
  Tensor.mean ((pred - target * log pred))

let pearson_corr_coef x y ?(dim=1) ~reduce_dims:reduce_dims =
  let x_centered = x - Tensor.mean ~dim:dim ~keepdim:true x in
  let y_centered = y - Tensor.mean ~dim:dim ~keepdim:true y in
  Tensor.mean (F.cosine_similarity x_centered y_centered ~dim:dim) ~dim:reduce_dims

let get_positional_features_exponential positions features seq_len ?(min_half_life=3.) () =
  let max_range = log (float_of_int seq_len) / log 2. in
  let half_life = Tensor.exp (-.log 2. / half_life * positions.abs () |> Tensor.to_type torch_float) in
  let positions = Tensor.abs positions in
  Tensor.exp (-.log 2. / half_life * positions)

let get_positional_features_central_mask positions features seq_len () =
  let center_widths = Tensor.pow 2. (Tensor.arange 1. (float_of_int features +. 1.) |> Tensor.to_device positions.device) - 1. in
  let center_widths = center_widths |> Tensor.to_type torch_float in
  let positions = Tensor.abs positions in
  Tensor.gt (center_widths |> Tensor.unsqueeze ~dim:0) positions |> Tensor.to_type torch_float

let gamma_pdf x concentration rate =
  let log_unnormalized_prob = Tensor.xlogy (concentration - 1.) x - rate * x in
  let log_normalization = Tensor.lgamma concentration - concentration * Tensor.log rate in
  Tensor.exp (log_unnormalized_prob - log_normalization)