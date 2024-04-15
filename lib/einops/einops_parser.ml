type transformation_pattern = {
  left: tensor_pattern;
  right: tensor_pattern;
}

type tensor_pattern = {
  delim: string option;
  group: axis_group;
}

type axis_group =
  | AG_Paren of axis_group
  | AG_List of string option * axis * axis_group option
  | AG_Single of axis

type axis = string

let recognize_token buf =
  let (name, _) = Lexing.read_token buf in
  match name with
  | "->" -> `ARROW
  | "," -> `COMMA
  | "..." -> `ELLIPSIS
  | "unitary_axis" -> `UNITARY_AXIS
  | name -> `NAMED_AXIS (name)
  | _ -> raise (Parsing.Unexpected (Lexing.lexeme_start_p buf, "Unexpected token"))

let rec parse_transformation_pattern buf =
  let (left, _) = parse_tensor_pattern buf in
  match recognize_token buf with
  | `ARROW ->
      let (right, _) = parse_tensor_pattern buf in
      (left, right)
  | _ -> raise (Parsing.Expected (`ARROW, Lexing.lexeme_start_p buf))

and parse_tensor_pattern buf =
  let delim =
    match recognize_token buf with
    | `COMMA | `EOF` -> None
    | _ -> Some (Lexing.read_string buf)
  in
  let group = parse_axis_group buf in
  (delim, group)

and parse_axis_group buf =
  match recognize_token buf with
  | `PAREN` ->
      Lexing.read_token buf;
      let group = parse_axis_group buf in
      match recognize_token buf with
      | `RPAREN` -> AG_Paren group
      | _ -> raise (Parsing.Expected (`RPAREN, Lexing.lexeme_start_p buf))
  | _ ->
      let delim =
        match recognize_token buf with
        | `COMMA | `EOF` -> None
        | _ -> Some (Lexing.read_string buf)
      in
      let axis = parse_axis buf in
      let rest =
        match recognize_token buf with
        | `COMMA` -> Some (parse_axis_group buf)
        | `EOF` -> None
        | _ -> raise (Parsing.Expected (`COMMA` | `EOF`, Lexing.lexeme_start_p buf))
      in
      AG_List (delim, axis, rest)

and parse_axis buf =
  match recognize_token buf with
  | `ELLIPSIS` -> "..."
  | `UNITARY_AXIS` -> "1"
  | `NAMED_AXIS (name)` -> name
  | _ -> raise (Parsing.Unexpected (Lexing.lexeme_start_p buf, "Unexpected token"))

let main () =
  let lexbuf = Lexing.from_string "<x, 1> -> y" in
  try
    let (pattern, _) = parse_transformation_pattern lexbuf in
    print_string (Printexc.to_string (pattern));
  with exn ->
    fprintf stderr "%s\n" (Printexc.to_string exn)