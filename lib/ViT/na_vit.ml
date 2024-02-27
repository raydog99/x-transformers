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

    let batched_images =
      if group_images then
        group_images_by_max_seq_len
          input_images
          ~patch_size:p
          ~calc_token_dropout:t.model_config.calc_token_dropout
          ~max_seq_len:group_max_seq_len
      else
        input_images