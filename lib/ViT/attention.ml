open Torch

module MultiheadAttention = struct
  type t = {
    num_embeddings = Config.num_embeddings;
    num_heads = Config.num_heads;
    attention_dropout = Config.attention_dropout;
  }

  let attention ~query ~key ~value ~attention_Mask ~is_training = 
    let bias =
      Tensor.ones [ 1; 1; num_embeddings; num_embeddings ] ~kind:(T Float)
      |> Tensor.tril ~diagonal:0
      |> Tensor.view ~size:[ 1; 1; num_embeddings; num_embeddings ]
    let v_shape = Tensor.size value in
    let attention_weight = Tensor.matmul query key in
    let attention_weight =
      match scale (* Scale in dot-product attn for exploding gradients -- better name? *)
        | True -> Tensor.div1 attention_weight (List.last_exn v_shape |> FLoat.of_int |> Float.sqrt |> Scalar.f)
        | False -> attention_weight in
    let num_heads, sequence_length = 
      match Tensor.size attention_weight with
      (* Checks number of dimensions, extract third and fourth elements *)
      | _ :: _ :: num_heads :: sequence_length :: _ -> num_heads, sequence_length
      | _ -> failwith "Unexpected number of dimensions" in
    let attention_bias = 
      Tensor.narrow bias ~dim:2 ~start:(sequence_length - num_heads) ~length:num_heads
      |> Tensor.narrow ~dim:3 ~start:0 ~length:num_heads in
    let attention_weight = Tensor.((attention_weight * attention_bias) + f 1e4 * (b - f 1.)) in
    let attention_weight = 
    match attention_mask with 
    | None -> attention_weight
    | Some mask -> Tensor.(attention_weight + mask) in
    let attention_weight = Tensor.softmax attention_weight ~dim:(-1) ~dtype:(T Float)
    |> Tensor.dropout ~p:attention_dropout ~is_training in
    Tensor.matmul attention_weight value
  ;;

  let split_heads xs ~k =
    let batch_size = Tensor.size xs |> List.hd_exn in
    let dims = if k then [0; 2; 3; 1] else [0; 2; 1; 3] in
    Tensor.view xs ~size: [batch_size; -1; num_heads; dim_per_head]
    |> Tensor.permute ~dims
;;

    let create vs config ~scale =
      fun xs ~layer_past ~attention_mask ~is_training ->
        let batch_size = Tensor.size xs |> List.hd_exn in
        let query, key, value = 
          match Layer.forward c_attn xs |> Tensor.split ~split_size:num_embeddings ~dim:2 with
          | query :: key :: value :: _ ->
            split_heads query ~k:false, split_heads key ~k:true, split_heads value ~k:false
          | _ -> failwith "Unexpected split size"in
        let key, value = 
          match layer_past with
          | None -> key, value
          | Some p ->
            let key = Tensor.cat 
            [ Tensor.get p 0 |> Tensor.transpose ~dim0:(-2) ~dim1:(-1); key ] ~dim(-1) in
            let value = Tensor.cat
            [ Tensor.get p 1; value ] ~dim:(-2) in
            key, value in
        let present = Tensor.stack
        [Tensor.transpose key ~dim0:(-2) ~dim1:(-1); value] ~dim:0 in
        let a =
          attention ~query ~key ~value ~attention_mask ~is_training
          |> Tensor.transpose ~dim0:1 ~dim1:2
          |> Tensor.continguous
          |> Tensor.view ~size: [ batch_size; -1; num_heads * dim_per_head ]
          |> Layer.forward c_proj
          |> Tensor.dropout ~p:residual_p ~is_training
    in
    a, present
;;
end