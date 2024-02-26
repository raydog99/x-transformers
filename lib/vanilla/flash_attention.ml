open Base
open Tensor

module FlashAttention : sig
  type t

  val forward :
    t ->
    Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t option -> bool ->
    int -> int -> Tensor.t
end = struct
  type t

  let forward t q k v mask causal q_bucket_size k_bucket_size =
    let device = Tensor.device q in
    let max_neg_value = Float.neg_infinity in
    let qk_len_diff = max 0 (Tensor.size k |> List.last_exn - Tensor.size q |> List.last_exn) in

    let o = Tensor.zeros_like q in
    let all_row_sums = Tensor.zeros (Tensor.shape q |> List.init ~f:Fn.id @ [1]) ~device in
    let all_row_maxes = Tensor.full (Tensor.shape q |> List.init ~f:Fn.id @ [1]) ~fill:max_neg_value ~device in

    let scale = Float.pow (Float.of_int (Tensor.size q |> List.last_exn)) (-0.5) in

    let num_row_tiles = Float.ceil (Float.of_int (Tensor.size q |> List.last_exn) /. Float.of_int q_bucket_size) |> Int.of_float in
    let num_col_tiles = Float.ceil (Float.of_int (Tensor.size k |> List.last_exn) /. Float.of_int k_bucket_size) |> Int.of_float in

    let mask =
      match mask with
      | None -> List.init num_col_tiles ~f:(fun _ -> None)
      | Some m ->
        if Tensor.ndim m = 2 then
          Some (Tensor.unsqueeze m ~dim:1 |> Tensor.unsqueeze ~dim:1)
        else
          Some (List.init num_row_tiles ~f:(fun _ -> Some (Tensor.split m ~dim:(-2) ~size:[q_bucket_size])))
    in

    let row_splits =
      List.zip_exn
        (Tensor.split q ~dim:(-2) ~size:[q_bucket_size])
        (Tensor.split o ~dim:(-2) ~size:[q_bucket_size])
        (List.map mask ~f:(function
          | None -> None
          | Some row_mask ->
            if Tensor.size row_mask |> List.last_exn = 1 then
              Some (List.init num_col_tiles ~f:(fun _ -> row_mask))
            else
              Some (List.map row_mask ~f:(fun col_mask -> Tensor.split col_mask ~dim:(-1) ~size:[k_bucket_size])))
        )
        (Tensor.split all_row_sums ~dim:(-2) ~size:[q_bucket_size])
        (Tensor.split all_row_maxes ~dim:(-2) ~size:[q_bucket_size])
    in

    List.iteri row_splits ~f:(fun ind (qc, oc, row_mask, row_sums, row_maxes) ->
      let q_start_index = ind * q_bucket_size - qk_len_diff in

      let col_splits =
        List.zip_exn
          (Tensor.split k ~dim:(-2) ~size:[k_bucket_size])
          (Tensor.split v ~dim:(-2) ~size:[k_bucket_size])
          row_mask
      in

      List.iter col_splits ~f:(fun (kc, vc, col_mask) ->
        let k_start_index = k_ind * k_bucket_size in

        let attn_weights = Tensor.matmul qc kc * scale in

        (match col_mask with
         | None -> ()
         | Some mask -> Tensor.masked_fill_ attn_weights ~mask:(Tensor.logical_not mask) ~value:max_neg_value);

        if causal && q_start_index < (k_start_index + k_bucket_size - 1) then
          let causal_mask = Tensor.triu (Tensor.ones [Tensor.size qc |> List.last_exn; Tensor.size kc |> List.last_exn] ~dtype:(T Bool) ~device) ~diagonal:(q_start_index - k_start_index + 1) in
          Tensor.masked_fill_ attn_weights ~mask:causal_mask ~value:max_neg_value;

        let block_row_maxes = Tensor.amax attn_weights ~dim:(-1) ~keepdim:true in
        let new_row_maxes = Tensor.maximum block_row_maxes row_maxes in

        let exp_weights = Tensor.exp (attn_weights - new_row_maxes) in

        (match col_mask with
         | None -> ()
         | Some mask -> Tensor.masked_fill_ exp_weights ~mask:(Tensor.logical_not mask) ~value:0.);

        let block_row_sums = Tensor.sum exp_weights ~dim:(-1) ~keepdim:true |> Tensor.clamp ~min:EPSILON in

        let exp_values = Tensor.matmul exp_weights vc in

        let exp_row_max_diff = Tensor.exp (row_maxes - new_row_maxes) in

        let new_row_sums = exp_row_max_diff * row_sums + block_row_sums in

        Tensor.mul_ oc exp_row_max_diff;
        Tensor.add_ oc exp_values;

        Tensor.copy_ row_maxes new_row_maxes;
        Tensor.copy_ row_sums new_row_sums
      );

      Tensor.div_ oc row_sums
    );

    let lse = Tensor.log all_row_sums + all_row_maxes in

    (t, causal, scale, mask, q_bucket_size, k_bucket_size), q, k, v, o, lse
end