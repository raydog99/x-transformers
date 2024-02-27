open Base
open Torch

module NaViT = struct
  type t = {
    model_config : ModelConfig.t;
    training_config : TrainingConfig.t;
    patch_embedding : PatchEmbedding.t;
    positional_embedding : PositionalEmbedding.t;
    vit : ViT.t;
    attn_pool_queries : Tensor.t;
    attn_pool : Attention.t;
    to_latent : nn;
    mlp_head : nn;
  }

  let create ~vs ~config =
    let {
      Config.image_size; patch_size; num_classes; dim; depth; heads; mlp_dim; channels;
      dim_head; dropout; emb_dropout; token_dropout_prob
    } = config in

    let patch_embedding = PatchEmbedding.create ~vs ~config in
    let positional_embedding = PositionalEmbedding.create ~vs ~config in
    let vit = ViT.create ~vs ~config in

    let attn_pool_queries = 
      Tensor.empty
      |> Tensor.normal_ ~std:1.
      |> Tensor.requires_grad_ ~requires_grad:false
      |> Var_store.var vs
    in

    let attn_pool = Attention.create ~vs ~config in

    let to_latent =
      let layer_norm = LayerNorm.create ~vs ~config:(Config.layer_norm_dim dim) in
      let linear = Linear.create ~vs ~config:(Config.linear_dim dim num_classes) in
      nn [ layer_norm; linear ]
    in

    let mlp_head =
      let layer_norm = LayerNorm.create ~vs ~config:(Config.layer_norm_dim dim) in
      let linear = Linear.create ~vs ~config:(Config.linear_dim dim num_classes ~bias:false) in
      nn [ layer_norm; linear ]
    in

    { model_config = config; training_config = training_config;
      patch_embedding; positional_embedding; transformer;
      attn_pool_queries; attn_pool; to_latent; mlp_head }
  ;;

  let forward t input_images ~group_images ~group_max_seq_len =
    let p, c, device, has_token_dropout =
      t.patch_embedding.patch_size, t.patch_embedding.channels, t.device, Option.is_some t.model_config.calc_token_dropout in

    let arange = Torch.arange1 ~device in
    let pad_sequence = Torch.nn.functional.pad_sequence ~batch_first:true in

    (* Auto pack if specified *)
    let batched_images =
      if group_images then
        group_images_by_max_seq_len
          input_images
          ~patch_size:p
          ~calc_token_dropout:t.model_config.calc_token_dropout
          ~max_seq_len:group_max_seq_len
      else
        input_images
    in

    (* Process images into variable length sequences with attention mask *)
    let num_images, batched_sequences, batched_positions, batched_image_ids =
      List.fold_left
        ~f:(fun (num_images_acc, seq_acc, pos_acc, ids_acc) images ->
            let num_images = num_images_acc + 1 in
            let sequences, positions, image_ids =
              List.fold_left
                ~f:(fun (seq_acc, pos_acc, ids_acc) (image_id, image) ->
                    let image_dims = Tensor.shape image in
                    let ph, pw = Tensor.map ~f:(fun d -> Int.div d p) image_dims in

                    let pos =
                      Tensor.meshgrid [arange ph; arange pw]
                      |> Tensor.stack ~dim:2
                      |> Tensor.view ~size:[-1; 2]
                    in

                    let seq =
                      Tensor.view image ~size:[c; ph; p; pw]
                      |> Tensor.permute ~dims:[1; 3; 0; 2]
                      |> Tensor.view ~size:[-1; c * p * p]
                    in

                    let seq_len = Tensor.size seq |> List.last_exn in

                    let seq, pos =
                      match t.model_config.calc_token_dropout with
                      | Some token_dropout ->
                        let token_dropout_prob = token_dropout image_dims in
                        let num_keep = max 1 (Int.of_float (Float.of_int seq_len * (1. -. token_dropout_prob))) in
                        let keep_indices = Tensor.normal_ ~std:1. |> Tensor.topk ~k:num_keep ~dim:0 in

                        Tensor.index_select seq ~dim:0 ~index:keep_indices,
                        Tensor.index_select pos ~dim:0 ~index:keep_indices
                      | None -> seq, pos
                    in

                    Tensor.cat seq_acc seq ~dim:0,
                    Tensor.cat pos_acc pos ~dim:0,
                    Tensor.cat ids_acc (Tensor.full_like seq (Scalar.of_int image_id)) ~dim:0
                )
                ~init:(Tensor.empty [], Tensor.empty [], Tensor.empty [])
                images
            in

            num_images, Tensor.cat seq_acc ~dim:0, Tensor.cat pos_acc ~dim:0, Tensor.cat ids_acc ~dim:0
        )
        ~init:(0, Tensor.empty [], Tensor.empty [], Tensor.empty [])
        batched_images