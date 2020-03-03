function transformer = gher_dineof_transformer (input, output)

# This function transforms specified datafile to format suitable with gher dineof

  dataset = load(input);
  gwrite(output, dataset.data);

endfunction