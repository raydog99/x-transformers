open Torch
open Base

type module_type = {
  embedding_wrapper : EmbeddingWrapper.t;
  rank1_edit_module : Rank1EditModule.t;
}

let get_finetune_parameters text_image_model =
  let params = ref [] in
  Torch.iter_modules text_image_model ~f:(fun module_ ->
    match module_ with
    | EmbeddingWrapper ew -> params := ew#parameters () @ !params
    | Rank1EditModule rm -> params := rm#parameters () @ !params
    | _ -> ()
  );
  !params

let get_finetune_optimizer ?(lr=1e-4) ?(wd=1e-2) ?(betas=(0.9, 0.99)) ?(eps=1e-8) ?(kwargs=[]) text_image_model =
  let params = get_finetune_parameters text_image_model in
  assert (List.length params > 0) ~msg:"no finetuneable parameters found";
  let total_params = List.map params ~f:T.numel |> List.sum in
  Printf.printf "optimizing %d parameters\n" total_params;

  let has_weight_decay = Float.(wd > 0.) in
  let adam_klass = if has_weight_decay then Torch.AdamW else Torch.Adam in
  let adam_kwargs = [("lr", lr); ("betas", T.tensor betas); ("eps", eps)] in

  let adam_kwargs =
    if has_weight_decay then
      List.append adam_kwargs [("weight_decay", wd)]
    else
      adam_kwargs
  in

  adam_klass params ~kwargs:(Array.of_list adam_kwargs)