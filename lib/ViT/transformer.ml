open Base
open Torch

module Transformer = struct
  type t = {
    modelConfig : ModelConfig.t;
    trainingConfig : TrainingConfig.t;
  }

  let create ~vs ~config =
    let { Config.num_layers; num_embeddings; num_heads; dim_feedforward;
          layer_norm_epsilon; residual_dropout; embedding_dropout; attention_dropout } = modelConfig in
    let encoderStack = Encoder.create ~vs:(Var_store.vs( / Int.to_string layer_idx) ~config) in
    let decoderStack = Decoder.create ~vs:(Var_store.vs( / Int.to_string layer_idx) ~config) in
    {device = vs.device; config; attention_layer; positional_encoding; layers }
;;

  let input_ids ~layer_past ~attention_mask ~token_type_ids ~position_ids ~is_training =
    let input_shape = Tensor.size input_ids in
    let seq_len = List.last_exn input_shape in
    let input_embeddings = Layer.forward wte input_ids in
    let layer_past_len =
      Option.value.map
        layer_past
        ~f:(fun (`layer layer_past) -> List.nth_exn (Tensor.size layer_past.(0)) 3)
        ~default:0 in
    let position_ids = 
      match position_ids with
      | Some position_ids -> position_ids
      | None ->
        Tensor.arange 1
        ~start:(Scalar.i layer_past_len)
        ~end_:(Scalar.i (seq_len + layer_past_len))
        ~options:(T Int64, Var_store.device vs) in
    let attention_mask =
      Option.map attention_mask ~f:(fun attention_mask ->
        let attention_mask =
          Tensor.flatten attention_mask
          |> Tensor.unsqueeze ~dim:1
          |> Tensor.unsqueeze ~dim:2 in
        Tensor.((attention_mask - f 1.) * f 1e4) in
    let position_embeddings =
    match token_type_ids with
    | None -> Tensor.zeros_like position_embeddings
    | Some -> token_type_ids -> Layer.forward wte token_type_ids in
    let hidden_state =
      Tensor.(input_embeddings + position_embeddings + token_type_embeddings)
      |> Tensor.dropout ~p.config.Config.embedding_dropout ~is_training in
    let output = Layer.forward ln_f hidden_state in
    output, `layer (Array.of_list presents)
;;
end