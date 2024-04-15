%token  "->" "," "..." "unitary_axis" 
%nonterminal transformation_pattern tensor_pattern axis_group axis named_axis anonymous_axes

%start transformation_pattern

%left ","

%action

transformation_pattern (left, right) = { left; right }

tensor_pattern delim group = { delim; group }

axis_group "(" delim ")" group = group
| delim axis rest = { delim; axis; rest }
| axis rest = { Empty; axis; rest }

axis "..." = "..."
| "unitary_axis" = "1"
| name = "named_axis," ^ name

%ignore EOF