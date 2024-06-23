open Torch

let set_module_requires_grad_ (module_ : Module.t) (requires_grad : bool) : unit =
  Module.parameters module_
  |> List.iter (fun param -> param.requires_grad <- requires_grad)

let freeze_all_layers_ (module_ : Module.t) : unit =
  set_module_requires_grad_ module_ false

let x_transformer_blocks (transformer : TransformerWrapper.t) : Module.t list =
  let blocks = ref [] in
  transformer.attn_layers.layers
  |> List.iter (fun layer -> blocks := layer.(Array.length layer - 1) :: !blocks);
  List.rev !blocks |> List.tl |> List.rev

type hidden_position =
  | Output

let recorder
    ?(outputs : 'a list option)
    ?(forward_hook_get_hidden : hidden_position = Output)
    ?(modules : Module.t list option) () =

  let output = match outputs with
    | Some out -> out
    | None -> [] in

  let get_output_fn = get_block_output_from_hook_outputs forward_hook_get_hidden in

  let recorder_fn = fun args ->
    match modules with
    | Some mods -> mods := args.(0) :: !mods
    | None -> ();
    let hidden = get_output_fn args in
    hidden |> Tensor.detach |> List.append output in
  recorder_fn