open Torch

let exists v = Torch.is_not_none v

let default v d = if exists v then v else d

let modulate_with_rotation x m =
  let open Complex in
  if Torch.dtype m = ComplexFloat || Torch.dtype m = ComplexDouble then
    let m_abs = Torch.abs m in
    let rot = Complex.((cos m_abs) + (sin m_abs *: I)) in
    x * rot
  else
    failwith "Expected complex tensor for modulation"

let complex_attention_real q k v attend ?mask () =
  assert (List.for_all (fun t -> Torch.dtype t = ComplexFloat || Torch.dtype t = ComplexDouble) [q; k; v]);
  let q_real = Torch.view_as_real q in
  let k_real = Torch.view_as_real k in
  let v_real = Torch.view_as_real v in
  let q_rearranged = Torch.rearrange q_real ~dims:"... d c -> ... (d c)" in
  let k_rearranged = Torch.rearrange k_real ~dims:"... d c -> ... (d c)" in
  let v_rearranged = Torch.rearrange v_real ~dims:"... d c -> ... (d c)" in
  let o = attend q_rearranged k_rearranged v_rearranged ?mask () in
  let o_rearranged = Torch.rearrange o ~dims:"... (d c) -> ... d c" ~c:2 in
  Torch.view_as_complex o_rearranged