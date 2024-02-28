open Base
open Torch

module TwinsSVT = struct
  type t = {
    num_classes : int;
    s1_emb_dim : int;
    s1_patch_size : int;
    s1_local_patch_size : int;
    s1_global_k : int;
    s1_depth : int;
    s2_emb_dim : int;
    s2_patch_size : int;
    s2_local_patch_size : int;
    s2_global_k : int;
    s2_depth : int;
    s3_emb_dim : int;
    s3_patch_size : int;
    s3_local_patch_size : int;
    s3_global_k : int;
    s3_depth : int;
    s4_emb_dim : int;
    s4_patch_size : int;
    s4_local_patch_size : int;
    s4_global_k : int;
    s4_depth : int;
    peg_kernel_size : int;
    dropout : float;
  }

  let create ~num_classes ~s1_emb_dim ~s1_patch_size ~s1_local_patch_size ~s1_global_k ~s1_depth
            ~s2_emb_dim ~s2_patch_size ~s2_local_patch_size ~s2_global_k ~s2_depth
            ~s3_emb_dim ~s3_patch_size ~s3_local_patch_size ~s3_global_k ~s3_depth
            ~s4_emb_dim ~s4_patch_size ~s4_local_patch_size ~s4_global_k ~s4_depth
            ~peg_kernel_size ~dropout =
    { num_classes; s1_emb_dim; s1_patch_size; s1_local_patch_size; s1_global_k; s1_depth;
      s2_emb_dim; s2_patch_size; s2_local_patch_size; s2_global_k; s2_depth;
      s3_emb_dim; s3_patch_size; s3_local_patch_size; s3_global_k; s3_depth;
      s4_emb_dim; s4_patch_size; s4_local_patch_size; s4_global_k; s4_depth;
      peg_kernel_size; dropout }
  
  let forward { num_classes; s1_emb_dim; s1_patch_size; s1_local_patch_size; s1_global_k; s1_depth;
                s2_emb_dim; s2_patch_size; s2_local_patch_size; s2_global_k; s2_depth;
                s3_emb_dim; s3_patch_size; s3_local_patch_size; s3_global_k; s3_depth;
                s4_emb_dim; s4_patch_size; s4_local_patch_size; s4_global_k; s4_depth;
                peg_kernel_size; dropout } x =
    let rec group_by_key_prefix_and_remove_prefix prefix d =
      let rec loop acc_prefix acc_d = function
        | [] -> acc_prefix, acc_d
        | (key, value) :: rest ->
          if String.is_prefix key ~prefix then
            let new_key = String.drop_prefix key (String.length prefix) in
            loop (new_key :: acc_prefix) ((new_key, value) :: acc_d) rest
          else
            loop acc_prefix ((key, value) :: acc_d) rest
      in
      loop [] [] d
    in

    let rec group_dict_by_key cond d =
      let rec loop acc_cond1 acc_cond2 = function
        | [] -> acc_cond1, acc_cond2
        | (key, value) :: rest ->
          let match_cond = cond key in
          let ind = if match_cond then acc_cond1 else acc_cond2 in
          loop (if match_cond then (key, value) :: acc_cond1 else acc_cond1)
               (if match_cond then acc_cond2 else (key, value) :: acc_cond2) rest
      in
      loop [] [] d
    in

    let rec create_layers prefix dim depth local_patch_size global_k has_local =
      let config, kwargs =
        group_by_key_prefix_and_remove_prefix prefix [
          ("emb_dim", dim);
          ("patch_size", local_patch_size);
          ("local_patch_size", local_patch_size);
          ("global_k", global_k);
          ("depth", depth);
          ("has_local", has_local);
        ]
      in
      let is_last = String.equal prefix "s4" in
      let dim_next = List.Assoc.find_exn config "emb_dim" in

      let patch_embedding =
        Layer_norm.create ~dim ~dim_out:dim_next
        |> Conv2d.create ~kernel_size:s1_patch_size ~stride:1 ~padding:0 ~bias:false
      in
      let transformer1 =
        Transformer.create ~dim:dim_next ~depth:1 ~local_patch_size ~global_k ~dropout ~has_local
      in
      let peg =
        PEG.create ~dim:dim_next ~kernel_size:peg_kernel_size
      in
      let transformer2 =
        Transformer.create ~dim:dim_next ~depth:depth ~local_patch_size ~global_k ~dropout ~has_local
      in

      let layer =
        Sequential.of_list
          [ patch_embedding; transformer1; peg; transformer2 ]
      in
      dim_next, layer
    in

    let layers =
      let _, s1_layer = create_layers "s1" s1_emb_dim s1_depth s1_local_patch_size s1_global_k true in
      let _, s2_layer = create_layers "s2" s2_emb_dim s2_depth s2_local_patch_size s2_global_k true in
      let _, s3_layer = create_layers "s3" s3_emb_dim s3_depth s3_local_patch_size s3_global_k true in
      let _, s4_layer = create_layers "s4" s4_emb_dim s4_depth s4_local_patch_size s4_global_k true in
      [ s1_layer; s2_layer; s3_layer; s4_layer ]
    in

    forward network x
  ;;
end