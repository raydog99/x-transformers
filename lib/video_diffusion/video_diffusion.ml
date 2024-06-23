open Base
open Torch
open Torch.Nn

module RelativePositionBias = struct
  type t = {
    num_buckets : int;
    max_distance : int;
    relative_attention_bias : Tensor.t;
  }

  let create ~heads ~num_buckets ~max_distance =
    let relative_attention_bias = nn_embedding num_buckets heads in
    { num_buckets; max_distance; relative_attention_bias }

  let _relative_position_bucket relative_position ~num_buckets ~max_distance =
    let open Tensor in
    let ret = zeros_like relative_position in
    let n = neg relative_position in

    let num_buckets' = num_buckets / 2 in
    let ret = ret + (n < zeros_like n |> to_type Int64 |> mul_scalar (Int64.of_int num_buckets')) in
    let n = abs n in

    let max_exact = num_buckets' / 2 in
    let is_small = n < full_like max_exact (float_of_int max_exact) in

    let val_if_large =
      min_
        (max_exact + (log (to_float n / to_float max_exact) / log (float_of_int (max_distance / max_exact)) * float_of_int (num_buckets - max_exact)) |> to_type Int64)
        (full_like n (float_of_int (num_buckets - 1)) |> to_type Int64) in

    let ret = ret + where is_small n val_if_large in
    ret

  let forward t n device =
    let open Tensor in
    let q_pos = arange ~start:(F 0.) ~end_:(F (float_of_int n)) ~dtype:Int64 ~device |> to_type Int64 in
    let k_pos = arange ~start:(F 0.) ~end_:(F (float_of_int n)) ~dtype:Int64 ~device |> to_type Int64 in
    let rel_pos = (rearrange k_pos "j" ["1"; "j"] - rearrange q_pos "i" ["i"; "1"]) |> to_type Int64 in
    let rp_bucket = _relative_position_bucket rel_pos ~num_buckets:t.num_buckets ~max_distance:t.max_distance in
    let values = index_select t.relative_attention_bias 0 rp_bucket in
    rearrange values "i j h" ["h"; "i"; "j"]
end

module EMA = struct
  type t = {
    beta : float;
  }

  let create beta = { beta }

  let update_model_average t ma_model current_model =
    List.iter2
      (fun current_params ma_params ->
        let old_weight = Tensor.data ma_params in
        let up_weight = Tensor.data current_params in
        ma_params.data <- update_average t old_weight up_weight)
      (current_model # parameters)
      (ma_model # parameters)

  let update_average t old new =
    match old with
    | None -> new
    | Some old -> old * t.beta +. (1. -. t.beta) *. new
end