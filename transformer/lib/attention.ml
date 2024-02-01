module Attention = struct
  let attn vs config ~scale =
    let { Config.n_embd; n_head; attn_p; resid_p; _ } = config in
    let bias =
      Tensor.ones [ 1; 1; n_embd; n_embd ] ~kind:(T Float)
      |> Tensor.tril ~diagonal:0
      |> Tensor.view ~size:[ 1; 1; n_embd; n_embd ]
    in
    let dim_per_head = n_embd / n_head in
    let attention ~query ~key ~value ~attention_mask ~is_training =
      let v_shape = Tensor.size value in
      let w = Tensor.matmul query key in
      let w =
        if scale
        then
          Tensor.div1 w (List.last_exn v_shape |> Float.of_int |> Float.sqrt |> Scalar.f)
        else w
      in
      let nd, ns =
        match Tensor.size w with
        | _ :: _ :: nd :: ns :: _ -> nd, ns
        | _ -> failwith "unexpected number of dimensions"
      in
      let b =
        Tensor.narrow bias ~dim:2 ~start:(ns - nd) ~length:nd
        |> Tensor.narrow ~dim:3 ~start:0 ~length:nd
      in
      let w = Tensor.((w * b) + (f 1e4 * (b - f 1.))) in
      let w =
        match attention_mask with
        | None -> w
        | Some mask -> Tensor.(w + mask)
      in
      let w =
        Tensor.softmax w ~dim:(-1) ~dtype:(T Float)
        |> Tensor.dropout ~p:attn_p ~is_training
      in
      Tensor.matmul w value
    in
    let split_heads xs ~k =
      let batch_size = Tensor.size xs |> List.hd_exn in
      let dims = if k then [ 0; 2; 3; 1 ] else [ 0; 2; 1; 3 ] in
      Tensor.view xs ~size:[ batch_size; -1; n_head; dim_per_head ]
      |> Tensor.permute ~dims
    in
    fun xs ~layer_past ~attention_mask ~is_training ->
      let batch_size = Tensor.size xs |> List.hd_exn in
      let query, key, value =
        match Layer.forward c_attn xs |> Tensor.split ~split_size:n_embd ~dim:2 with
        | query :: key :: value :: _ ->
          split_heads query ~k:false, split_heads key ~k:true, split_heads value ~k:false
        | _ -> failwith "unexpected split size"
      in
      let key, value =
        match layer_past with
        | None -> key, value
        | Some p ->
          let key =
            Tensor.cat
              [ Tensor.get p 0 |> Tensor.transpose ~dim0:(-2) ~dim1:(-1); key ]
              ~dim:(-1)
          in
          let value = Tensor.cat [ Tensor.get p 1; value ] ~dim:(-2) in
          key, value
      in
      let present =
        Tensor.stack [ Tensor.transpose key ~dim0:(-2) ~dim1:(-1); value ] ~dim:0
      in
      let a =
        attention ~query ~key ~value ~attention_mask ~is_training
        |> Tensor.transpose ~dim0:1 ~dim1:2
        |> Tensor.contiguous
        |> Tensor.view ~size:[ batch_size; -1; n_head * dim_per_head ]
        |> Layer.forward c_proj
        |> Tensor.dropout ~p:resid_p ~is_training
      in
      a, present
end