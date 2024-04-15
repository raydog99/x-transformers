let rec lex buf =
  let pos = Lexing.lexeme_start_p buf in
  match Lexing.read_char buf with
  | ' ' => classify (" ")
  | ',' => classify ","
  | '(' | ')' => classify (string_of_char (Lexing.read_char buf))
  | '1' => classify "unitary_axis"
  | '.' | '.'. '.' => classify "..."
  | c when Char.is_alpha c =>
      let name = Lexing.read_string buf in
      classify ("named_axis," ^ name)
  | c when Char.is_digit c =>
      let digits = Lexing.read_string buf in
      if String.for_all (fun char -> Char.is_digit char) digits then
        classify ("anonymous_axis," ^ digits)
      else
        raise (Lexing.Unexpected (pos, "Invalid character in anonymous axis"))
  | _ ->
      raise (Lexing.Unexpected (pos, "Unexpected character"))

  | exception Lexing.Eof ->
      classify End_of_file